"""CPU-only target adaptation with separate fit, select, confirm and report phases."""
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from newsforge.external.aggregate import paired_interval
from newsforge.external.common import DAY, HOUR, START, stage
from newsforge.external.evaluate import frozen_features, load_models, query_diagnostics
from newsforge.external.prepare import connect_readonly
from newsforge.v2.temporal import _MemoryMonitor
from newsforge.v3.residual import PairwiseResidual
from .common import (CONFIG, DELTAS, METHODS, MODEL_NAMES, PHASES, REPRESENTATION_CUTOFF, SEEDS,
                     check_run, finish_phase, initialize, phase_receipt, read_json,
                     sha256, write_new_json)
from .replay import (build_ema_snapshots, load_original_queries, replay_query,
                     select_phase_queries, slow_reference_pool)
from .representation import fit_local_representation
from .scoring import (confirmation_passes, ensemble, load_adapted_models,
                      rank_methods, sample_pairs, select_delta)


def load_cache(run, phase):
    path = Path(run) / 'prepare' / phase
    return {name: np.load(path / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            for name in ('queries', 'pools', 'sizes', 'history_lengths', 'english', 'local')}


def _cache_phase(connection, first_seen, snapshots, english, local, coverage, queries, out, log):
    out.mkdir(exist_ok=False)
    n = len(queries)
    np.save(out / 'queries.npy', queries, allow_pickle=False)
    shapes = {'pools': ((n, 100), np.int32), 'sizes': ((n,), np.int16),
              'history_lengths': ((n,), np.int16), 'english': ((n, 100, 13), np.float32),
              'local': ((n, 100, 13), np.float32)}
    arrays = {name: np.lib.format.open_memmap(out / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
              for name, (shape, dtype) in shapes.items()}
    for array in arrays.values():
        array[:] = 0
    diagnostics = {'english': {}, 'local': {}}
    last_log = time.perf_counter()
    for row, (_, stamp, user, target) in enumerate(queries):
        history, pool = replay_query(connection, first_seen, snapshots, int(user), int(stamp))
        size = len(pool)
        arrays['pools'][row, :size] = pool
        arrays['sizes'][row] = size
        arrays['history_lengths'][row] = min(len(history), 20)
        if size:
            arrays['english'][row, :size] = frozen_features(english, history, pool)
            arrays['local'][row, :size] = frozen_features(local, history, pool)
        # A fallback item lies outside the recent first-observed window.
        fallback = int(np.count_nonzero(first_seen[pool] < int(stamp) // HOUR * HOUR - 24 * HOUR))
        for name in diagnostics:
            values = query_diagnostics(first_seen, coverage[name], history, pool, int(target), int(stamp), fallback)
            values['target_outside_pool'] = int(target not in pool)
            for key, value in values.items():
                diagnostics[name][key] = diagnostics[name].get(key, 0) + value
        if time.perf_counter() - last_log > 15:
            log('feature_cache_progress', phase=out.name, completed=row + 1, total=n)
            last_log = time.perf_counter()
    for array in arrays.values():
        array.flush()
        array._mmap.close()
    return {'requests': n, 'diagnostics': diagnostics,
            'queries_sha256': sha256(out / 'queries.npy')}


def prepare(repo, base, run):
    repo, base, run = Path(repo), Path(base), Path(run)
    initialize(repo, base, run)
    with stage(run, 'prepare') as (out, log), threadpool_limits(limits=1), _MemoryMonitor() as memory:
        began = time.perf_counter()
        connection = connect_readonly(base / 'prepare/events.sqlite')
        counts = connect_readonly(base / 'evaluate/reference_counts.sqlite')
        try:
            first_seen = np.load(base / 'prepare/first_seen.npy', allow_pickle=False)
            english = np.load(base / 'prepare/features.npy', mmap_mode='r', allow_pickle=False)
            local, representation = fit_local_representation(connection, first_seen, out / 'representation',
                                                             REPRESENTATION_CUTOFF)
            if not np.array_equal(local[:, 96:], english[:, 96:]):
                raise ValueError('local representation changed the matched category suffix')
            write_new_json(out / 'representation.json', representation)
            coverage = {'english': np.load(base / 'prepare/coverage.npy', allow_pickle=False),
                        'local': np.load(out / 'representation/coverage.npy', allow_pickle=False)}
            snapshots = build_ema_snapshots(counts, len(first_seen))
            np.save(out / 'snapshots.npy', snapshots, allow_pickle=False)
            phases = {}
            for phase, (lo, hi, limit) in PHASES.items():
                queries = select_phase_queries(connection, phase, lo, hi, limit)
                if not len(queries):
                    raise ValueError('the frozen phase contains no requests')
                phases[phase] = _cache_phase(connection, first_seen, snapshots, english, local,
                                             coverage, queries, out / phase, log)
                phases[phase]['available_requests'] = connection.execute(
                    'SELECT COUNT(*) FROM events WHERE timestamp>=? AND timestamp<?', (lo, hi)).fetchone()[0]
                phases[phase]['budget'] = limit
                phases[phase]['sample_shortfall'] = max(0, limit - len(queries))
            queries = load_original_queries(connection)
            original_queries = np.load(base / 'evaluate/queries.npy', allow_pickle=False)
            if not np.array_equal(queries, original_queries) or len(queries) != 15000:
                raise ValueError('previously observed evaluation query identities changed')
            phases['evaluate'] = _cache_phase(connection, first_seen, snapshots, english, local,
                                               coverage, queries, out / 'evaluate', log)
            for name in ('pools', 'sizes'):
                old = np.load(base / 'evaluate' / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                new = np.load(out / 'evaluate' / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                if not np.array_equal(old, new):
                    raise ValueError('original evaluation candidates changed')
        finally:
            connection.close()
            counts.close()
        check_run(repo, base, run)
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob('*')
                       if p.is_file() and p.name != 'progress.jsonl')
        result = finish_phase(run, 'prepare', files, phases=phases, representation=representation,
                              original_query_and_candidate_equality=True,
                              elapsed_seconds=time.perf_counter() - began, memory=memory.report())
        log('preparation_complete', requests=sum(p['requests'] for p in phases.values()))
        return result


def fit(repo, base, run):
    run = Path(run)
    check_run(repo, base, run)
    prep = phase_receipt(run, 'prepare')
    cache = load_cache(run, 'fit')
    with stage(run, 'fit') as (out, log), threadpool_limits(limits=1), _MemoryMonitor() as memory:
        began = time.perf_counter()
        statistics = {}
        for seed in SEEDS:
            pairs, counts = sample_pairs(cache['queries'], cache['pools'], cache['sizes'],
                                         cache['history_lengths'], seed)
            if not len(pairs):
                raise ValueError('no eligible target-domain training pairs')
            pair_file = out / f'pairs-seed{seed}.npy'
            np.save(pair_file, pairs, allow_pickle=False)
            bindings = {'manifest_sha256': sha256(run / 'manifest.json'),
                        'prepare_receipt_sha256': sha256(run / 'prepare/receipt.json'),
                        'pair_indices_sha256': sha256(pair_file), 'seed': seed,
                        'label_semantics': 'next observed read versus unobserved candidates, not exposure negatives'}
            for name in MODEL_NAMES:
                features = cache['english' if name == 'english' else 'local']
                positive = features[pairs[:, 0], pairs[:, 1]]
                negative = features[pairs[:, 0], pairs[:, 2]]
                variant = 'no_recent' if name == 'local_no_recent' else 'full13'
                model = PairwiseResidual(variant).fit(positive, negative, bindings=bindings)
                model.save(out / 'models' / f'{name}-seed{seed}.npz')
                statistics[f'{name}-seed{seed}'] = {**counts, 'fit_metadata': model.fit_metadata,
                                                    'pair_indices_sha256': bindings['pair_indices_sha256']}
                log('model_fit_complete', model=name, seed=seed, pairs=len(pairs))
        check_run(repo, base, run)
        names = sorted(p.relative_to(out).as_posix() for p in out.rglob('*')
                       if p.is_file() and p.name != 'progress.jsonl')
        return finish_phase(run, 'fit', names, models=statistics,
                            prepare_receipt_sha256=sha256(run / 'prepare/receipt.json'),
                            elapsed_seconds=time.perf_counter() - began, memory=memory.report())


def _predict(cache, models, frozen, delta, passed):
    n = len(cache['queries'])
    orders = np.full((n, len(METHODS), 100), -1, dtype=np.int16)
    scores = np.zeros((n, len(METHODS), 100), dtype=np.float64)
    hits = np.zeros((n, len(METHODS)), dtype=np.uint8)
    hits100 = np.zeros_like(hits)
    reciprocal = np.zeros((n, len(METHODS)), dtype=np.float64)
    for row, query in enumerate(cache['queries']):
        size = int(cache['sizes'][row])
        order, score = rank_methods(cache['english'][row, :size], cache['local'][row, :size],
                                    int(cache['history_lengths'][row]), models, frozen, delta, passed)
        orders[row, :, :size], scores[row, :, :size] = order, score
        pool = cache['pools'][row, :size]
        for method in range(len(METHODS)):
            locations = np.flatnonzero(pool[order[method]] == query[3])
            hits100[row, method] = bool(len(locations))
            if len(locations) and locations[0] < 10:
                hits[row, method] = 1
                reciprocal[row, method] = 1. / (int(locations[0]) + 1)
    return dict(orders=orders, scores=scores, hits10=hits, hits100=hits100, reciprocal10=reciprocal)


COMPARISONS = {'adapted_local_minus_baseline': (4, 0), 'deployed_minus_baseline': (7, 0),
    'adapted_english_minus_matched_frozen_english': (3, 2),
    'adapted_local_minus_adapted_english': (4, 3),
    'adapted_local_minus_no_recent': (4, 5), 'adapted_local_minus_no_gate': (4, 6),
    'adapted_local_minus_legacy_frozen_english': (4, 1)}


def summarize(queries, predictions):
    hits, h100, reciprocal = (predictions[k] for k in ('hits10', 'hits100', 'reciprocal10'))
    n = len(queries)
    if not n or not np.all(h100 == h100[:, :1]):
        raise ValueError('empty evaluation or candidate-set metric drift')
    result = {'requests': n, 'users': len(np.unique(queries[:, 2])), 'methods': {}, 'comparisons': {}}
    for i, name in enumerate(METHODS):
        result['methods'][name] = {'hits10': int(hits[:, i].sum()), 'denominator': n,
            'hit_rate10': float(hits[:, i].mean()), 'hits100': int(h100[:, i].sum()),
            'hit_rate100': float(h100[:, i].mean()), 'mrr10': float(reciprocal[:, i].mean())}
    for name, (left, right) in COMPARISONS.items():
        difference = hits[:, left].astype(np.int16) - hits[:, right]
        result['comparisons'][name] = {**paired_interval(queries[:, 2], difference),
            'wins': int(np.count_nonzero(difference > 0)), 'losses': int(np.count_nonzero(difference < 0)),
            'ties': int(np.count_nonzero(difference == 0))}
    return result


def tune(repo, base, run):
    run = Path(run)
    check_run(repo, base, run)
    phase_receipt(run, 'prepare'); phase_receipt(run, 'fit')
    cache = load_cache(run, 'tune')
    models = load_adapted_models(run)
    with stage(run, 'tune') as (out, log), threadpool_limits(limits=1), _MemoryMonitor() as memory:
        began = time.perf_counter()
        grid = {}
        for delta in DELTAS:
            hits = 0
            for row, query in enumerate(cache['queries']):
                size = int(cache['sizes'][row])
                n = int(cache['history_lengths'][row])
                original = np.arange(size)
                residual = ensemble(cache['local'][row, :size], models['local']) if size and n else np.zeros(size)
                scores = -original + delta * (n / (n + 5.)) * residual
                order = np.lexsort((original, -scores))
                hits += int(query[3] in cache['pools'][row, :size][order[:10]])
            grid[str(delta)] = {'hits10': hits, 'requests': len(cache['queries']),
                                'hit_rate10': hits / len(cache['queries'])}
        delta = select_delta(grid)
        selection = {'delta': delta, 'grid': grid, 'primary_recipe': 'adapted_local',
                     'selection_rule': 'maximum tune hits10, tie smaller delta',
                     'fit_receipt_sha256': sha256(run / 'fit/receipt.json'),
                     'prepare_receipt_sha256': sha256(run / 'prepare/receipt.json')}
        write_new_json(out / 'selection.json', selection)
        log('selection_complete', delta=delta)
        check_run(repo, base, run)
        return finish_phase(run, 'tune', ['selection.json'], selection=selection,
                            prepare_receipt_sha256=selection['prepare_receipt_sha256'],
                            fit_receipt_sha256=selection['fit_receipt_sha256'],
                            elapsed_seconds=time.perf_counter() - began, memory=memory.report())


def confirm(repo, base, run):
    run = Path(run)
    check_run(repo, base, run)
    phase_receipt(run, 'prepare'); phase_receipt(run, 'fit'); phase_receipt(run, 'tune')
    selection = read_json(run / 'tune/selection.json')
    cache = load_cache(run, 'confirm')
    with stage(run, 'confirm') as (out, log), threadpool_limits(limits=1), _MemoryMonitor() as memory:
        began = time.perf_counter()
        predictions = _predict(cache, load_adapted_models(run), load_models(repo)['full13'], selection['delta'], False)
        # Only the declared primary comparison can decide this gate.
        difference = predictions['hits10'][:, 4].astype(np.int16) - predictions['hits10'][:, 0]
        comparison = paired_interval(cache['queries'][:, 2], difference)
        passed = confirmation_passes(selection['delta'], comparison)
        decision = {'gate_passed': passed, 'delta': selection['delta'], 'primary_comparison': comparison,
                    'requests': len(cache['queries']),
                    'baseline_hits10': int(predictions['hits10'][:, 0].sum()),
                    'adapted_local_hits10': int(predictions['hits10'][:, 4].sum()),
                    'deployed_recipe': 'adapted_local' if passed else 'baseline',
                    'selection_sha256': sha256(run / 'tune/selection.json'),
                    'earliest_service_timestamp': START + 4 * DAY}
        for name, array in predictions.items():
            np.save(out / (name + '.npy'), array, allow_pickle=False)
        write_new_json(out / 'decision.json', decision)
        log('confirmation_complete', gate_passed=passed)
        check_run(repo, base, run)
        return finish_phase(run, 'confirm', [*(name + '.npy' for name in predictions), 'decision.json'],
                            decision=decision, selection_sha256=sha256(run / 'tune/selection.json'),
                            prepare_receipt_sha256=sha256(run / 'prepare/receipt.json'),
                            fit_receipt_sha256=sha256(run / 'fit/receipt.json'),
                            elapsed_seconds=time.perf_counter() - began, memory=memory.report())


def _audit(connection, first_seen, english, local, cache, predictions, models, frozen, delta, passed):
    # Separate raw SQL aggregation, no optimized snapshot table or cached features.
    hourly = list(connection.execute('SELECT timestamp/3600,item,COUNT(*) FROM events GROUP BY timestamp/3600,item ORDER BY timestamp/3600,item'))
    indices = np.linspace(0, len(cache['queries']) - 1, 128, dtype=np.int64)
    maximum = 0.
    for index in indices:
        _, stamp, user, target = cache['queries'][index]
        _, pool = slow_reference_pool(connection, first_seen, int(user), int(stamp), hourly_counts=hourly)
        tail = list(connection.execute('SELECT item FROM events WHERE user=? AND timestamp<? ORDER BY timestamp DESC,member COLLATE BINARY DESC,line DESC LIMIT 20', (int(user), int(stamp))))
        history = np.array([r[0] for r in reversed(tail)], dtype=np.int64)
        size = int(cache['sizes'][index])
        if not np.array_equal(pool, cache['pools'][index, :size]):
            raise ValueError('independent candidate audit mismatch')
        eng, loc = frozen_features(english, history, pool), frozen_features(local, history, pool)
        values = []
        for matrix, collection in ((eng, frozen), (eng, models['english']),
                                    (loc, models['local']), (loc, models['local_no_recent'])):
            manual = [np.clip(((matrix[:, m.feature_indices].astype(np.float64) - m.mean) / m.scale) @ m.weights, -1, 1)
                      for m in collection]
            values.append(np.mean(np.stack(manual), axis=0))
        n = len(history)
        gate = n / (n + 5.)
        adjustments = (np.zeros(size), 10 * gate * values[0], delta * gate * values[0],
                       delta * gate * values[1], delta * gate * values[2],
                       delta * gate * values[3], delta * values[2] if n else np.zeros(size))
        scores = np.stack([-np.arange(size, dtype=np.float64) + (a if n else 0) for a in adjustments])
        scores = np.concatenate((scores, (scores[4] if passed else scores[0])[None]))
        order = np.stack([np.lexsort((np.arange(size), -s)) for s in scores])
        maximum = max(maximum, float(np.max(np.abs(scores - predictions['scores'][index, :, :size]), initial=0.)))
        if maximum > 1e-8 or not np.array_equal(order, predictions['orders'][index, :, :size]):
            raise ValueError('independent scoring audit mismatch')
        for method in range(len(METHODS)):
            if int(target in pool[order[method, :10]]) != predictions['hits10'][index, method]:
                raise ValueError('independent hit audit mismatch')
    return {'passed': True, 'queries': len(indices), 'score_max_absolute_error': maximum,
            'candidate_and_ranking_equality': 'exact', 'score_atol': 1e-8,
            'shared_definition': 'original content_features and fitted local article vectors',
            'independent': ['raw-event SQL counts', 'from-zero EMA', 'user history and candidates', 'linear scoring and stable sorting', 'hits']}


def evaluate(repo, base, run, reports):
    repo, base, run, reports = map(Path, (repo, base, run, reports))
    if reports.exists():
        raise FileExistsError('public output must not exist')
    manifest = check_run(repo, base, run, verify_base=True)
    phases = {phase: phase_receipt(run, phase) for phase in ('prepare', 'fit', 'tune', 'confirm')}
    decision = read_json(run / 'confirm/decision.json')
    cache = load_cache(run, 'evaluate')
    with stage(run, 'evaluate') as (out, log), threadpool_limits(limits=1), _MemoryMonitor() as memory:
        began = time.perf_counter()
        models, frozen = load_adapted_models(run), load_models(repo)['full13']
        predictions = _predict(cache, models, frozen, decision['delta'], decision['gate_passed'])
        # Exact per-request legacy reconciliation, not only matching headline totals.
        for name in ('hits10', 'hits100', 'reciprocal10'):
            old = np.load(base / 'evaluate' / (name + '.npy'), allow_pickle=False)
            if not np.array_equal(predictions[name][:, 1], old[:, 1]):
                raise ValueError('legacy frozen-model predictions changed')
        for row, query in enumerate(cache['queries']):
            size = int(cache['sizes'][row])
            pool = cache['pools'][row, :size]
            for method in range(len(METHODS)):
                order = predictions['orders'][row, method, :size]
                if not np.array_equal(np.sort(order), np.arange(size)):
                    raise ValueError('saved ranking is not a permutation')
                where = np.flatnonzero(pool[order] == query[3])
                hit = bool(len(where) and where[0] < 10)
                rr = 1. / (int(where[0]) + 1) if hit else 0.
                if (int(hit) != predictions['hits10'][row, method]
                        or int(bool(len(where))) != predictions['hits100'][row, method]
                        or rr != predictions['reciprocal10'][row, method]):
                    raise ValueError('all-request metric reconstruction failed')
        connection = connect_readonly(base / 'prepare/events.sqlite')
        try:
            audit = _audit(connection, np.load(base / 'prepare/first_seen.npy', allow_pickle=False),
                np.load(base / 'prepare/features.npy', mmap_mode='r', allow_pickle=False),
                np.load(run / 'prepare/representation/features.npy', mmap_mode='r', allow_pickle=False),
                cache, predictions, models, frozen, decision['delta'], decision['gate_passed'])
        finally:
            connection.close()
        overall = summarize(cache['queries'], predictions)
        daily = {}
        for day in range(3):
            lo = START + (4 + day) * DAY
            mask = (cache['queries'][:, 1] >= lo) & (cache['queries'][:, 1] < lo + DAY)
            daily[f'2017-01-{5 + day:02}'] = summarize(cache['queries'][mask],
                                                     {k: v[mask] for k, v in predictions.items()})
        for name, array in predictions.items():
            np.save(out / (name + '.npy'), array, allow_pickle=False)
        write_new_json(out / 'audit.json', audit)
        check_run(repo, base, run, verify_base=True)
        receipt = finish_phase(run, 'evaluate', [*(name + '.npy' for name in predictions), 'audit.json'],
            confirm_receipt_sha256=sha256(run / 'confirm/receipt.json'),
            elapsed_seconds=time.perf_counter() - began, memory=memory.report())
        report = {'schema_version': 1, 'passed': True, 'scope': manifest['scope'],
            'new_unseen_external_test': False, 'online_ab_test': False, 'target_domain_fitting': True,
            'config': CONFIG, 'sources': manifest['sources'], 'source_commit': manifest['source_commit'],
            'environment': manifest['environment'], 'manifest_sha256': sha256(run / 'manifest.json'),
            'base_receipts': manifest['base_receipts'], 'base_files': manifest['base_files'],
            'representation': phases['prepare']['representation'],
            'phase_data': phases['prepare']['phases'], 'fit': phases['fit']['models'],
            'selection': phases['tune']['selection'], 'confirmation': decision,
            'overall': overall, 'by_day': daily, 'audit': audit,
            'all_request_metrics_recomputed': True, 'legacy_per_request_equality': True,
            'stage_receipts': {p: sha256(run / p / 'receipt.json') for p in (*phases, 'evaluate')},
            'per_query_files': receipt['files'],
            'resources': {p: {key: rec.get(key) for key in ('elapsed_seconds', 'memory')}
                          for p, rec in {**phases, 'evaluate': receipt}.items()},
            'limits': ['previously observed Jan5-7 benchmark; adaptation motivated by its earlier negative result',
                       'unobserved alternatives are not known exposure negatives or calibrated CTR',
                       'paired user-cluster intervals condition on realized dates; exploratory contrasts unadjusted',
                       'same delta for ablations is a matched-strength comparison, not independently tuned optima',
                       'confirmation failure retains the baseline; no later evaluation-based promotion']}
        reports.mkdir(parents=True, exist_ok=False)
        write_new_json(reports / 'evaluation.json', report)
        write_new_json(reports / 'audit.json', audit)
        lines = ['# Adressa 目标域适配：完整执行结果', '',
                 '已观察基准上的后续研究；目标域重新拟合，不是新的冻结迁移或线上实验。', '',
                 f"确认门通过：{decision['gate_passed']}；delta={decision['delta']}；实际策略={decision['deployed_recipe']}。", '',
                 '| 方法 | Top10 / 请求 | HR@10 | HR@100 | MRR@10 |', '|---|---:|---:|---:|---:|']
        for name, value in overall['methods'].items():
            lines.append(f"| {name} | {value['hits10']} / {value['denominator']} | {value['hit_rate10']:.6f} | {value['hit_rate100']:.6f} | {value['mrr10']:.6f} |")
        lines += ['', '| 对照 | 差值（pp） | 95% CI（pp） | 获益 / 受损请求 |', '|---|---:|---|---:|']
        for name, value in overall['comparisons'].items():
            lines.append(f"| {name} | {100*value['difference']:+.4f} | [{100*value['ci95'][0]:+.4f}, {100*value['ci95'][1]:+.4f}] | {value['wins']} / {value['losses']} |")
        lines += ['', '完整逐日结果、训练样本、覆盖、选择、确认、来源与限制见 [evaluation.json](evaluation.json)。',
                  '128 请求独立候选与评分审计通过；所有请求保存排名与指标重算通过；原冻结模型逐请求结果完全复现。', '']
        with (reports / 'RESULTS.md').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(lines))
        log('evaluation_complete', requests=len(cache['queries']), gate_passed=decision['gate_passed'])
        return report
