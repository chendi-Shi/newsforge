"""Fixed experiment configuration and immutable phase bindings."""
from pathlib import Path
import subprocess

from newsforge.external.common import (DAY, HOUR, START, check_files, environment,
    file_records, frozen_bindings, read_json, sha256, write_new_json)

PROTOCOL = 'docs/ADRESSA_ADAPTATION_PROTOCOL.md'
SEEDS = (17, 29, 43)
DELTAS = (0, 2, 5, 10)
PHASES = {
    'fit': (START + DAY, START + 2 * DAY - HOUR, 30000),
    'tune': (START + 2 * DAY, START + 3 * DAY - HOUR, 5000),
    'confirm': (START + 3 * DAY, START + 4 * DAY - HOUR, 5000),
}
REPRESENTATION_CUTOFF = START + DAY - HOUR
MODEL_NAMES = ('english', 'local', 'local_no_recent')
METHODS = ('baseline', 'legacy_frozen_english_delta10', 'matched_frozen_english',
           'adapted_english', 'adapted_local', 'adapted_local_no_recent',
           'adapted_local_no_gate', 'deployed')
CONFIG = {'schema_version': 1, 'seeds': list(SEEDS), 'delta_grid': list(DELTAS),
          'phase_ranges_and_budgets': {k: list(v) for k, v in PHASES.items()},
          'representation_cutoff': REPRESENTATION_CUTOFF,
          'alternatives': 8, 'replacement': False, 'numeric_threads': 1,
          'bootstrap_repetitions': 1000, 'bootstrap_seed': 2026,
          'gate': 'selected_delta>0 and confirm_difference>0 and confirm_ci95_lower>0',
          'evaluation': 'exact previously observed original 15000 queries',
          'methods': list(METHODS)}


def source_bindings(repo):
    repo = Path(repo)
    names = [p.relative_to(repo).as_posix() for p in (repo / 'newsforge/adaptation').glob('*.py')]
    names += [PROTOCOL, 'requirements-raw-windows.lock', 'newsforge/v3/residual.py',
              'newsforge/v2/ranker.py', 'newsforge/v2/temporal.py', 'newsforge/v2/benchmark.py']
    names += [p.relative_to(repo).as_posix() for p in (repo / 'newsforge/external').glob('*.py')]
    return {name: sha256(repo / name) for name in sorted(names)}


def initialize(repo, base, run):
    repo, base, run = Path(repo), Path(base), Path(run)
    if run.exists():
        raise FileExistsError('adaptation run must not already exist')
    prep = read_json(base / 'prepare/receipt.json')
    evaluation = read_json(base / 'evaluate/receipt.json')
    if evaluation['prepare_receipt_sha256'] != sha256(base / 'prepare/receipt.json'):
        raise ValueError('base preparation/evaluation mismatch')
    used = {'prepare': ('events.sqlite', 'features.npy', 'first_seen.npy', 'coverage.npy'),
            'evaluate': ('reference_counts.sqlite', 'queries.npy', 'pools.npy', 'sizes.npy',
                         'hits10.npy', 'hits100.npy', 'reciprocal10.npy')}
    records = {}
    for phase, names in used.items():
        receipt = prep if phase == 'prepare' else evaluation
        selected = {name: receipt['files'][name] for name in names}
        check_files(base / phase, selected)
        records[phase] = selected
    bindings = frozen_bindings(repo)
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True)
    if dirty.strip():
        raise ValueError('commit reviewed source and protocol before formal initialization')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    run.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'source_commit': commit, 'config': CONFIG,
                'sources': source_bindings(repo), 'environment': environment(),
                'base_receipts': {phase: sha256(base / phase / 'receipt.json')
                                  for phase in ('prepare', 'evaluate')},
                'base_files': records, 'frozen_mind_files': bindings,
                'existing_external_result_sha256': sha256(repo / 'reports/external/adressa_v1/evaluation.json'),
                'scope': 'Retrospectively motivated target-domain adaptation; old evaluation benchmark already observed.'}
    write_new_json(run / 'manifest.json', manifest)
    return manifest


def check_run(repo, base, run, *, verify_base=False):
    manifest = read_json(Path(run) / 'manifest.json')
    if manifest['config'] != CONFIG or manifest['sources'] != source_bindings(repo):
        raise ValueError('adaptation protocol, configuration or source changed')
    if manifest['environment'] != environment():
        raise ValueError('adaptation runtime changed')
    if sha256(Path(repo) / 'reports/external/adressa_v1/evaluation.json') != manifest['existing_external_result_sha256']:
        raise ValueError('original external result changed')
    for phase, expected in manifest['base_receipts'].items():
        if sha256(Path(base) / phase / 'receipt.json') != expected:
            raise ValueError('base receipt changed')
    if verify_base:
        for phase, records in manifest['base_files'].items():
            check_files(Path(base) / phase, records)
        if frozen_bindings(repo) != manifest['frozen_mind_files']:
            raise ValueError('frozen MIND inputs changed')
    return manifest


def phase_receipt(run, phase):
    run = Path(run)
    record = read_json(run / phase / 'receipt.json')
    if record['manifest_sha256'] != sha256(run / 'manifest.json') or record['status'] != 'passed':
        raise ValueError('phase manifest or status mismatch')
    for key, relative in (('prepare_receipt_sha256', 'prepare/receipt.json'),
                          ('fit_receipt_sha256', 'fit/receipt.json'),
                          ('selection_sha256', 'tune/selection.json'),
                          ('confirm_receipt_sha256', 'confirm/receipt.json')):
        if key in record and record[key] != sha256(run / relative):
            raise ValueError('phase dependency changed: ' + key)
    check_files(run / phase, record['files'])
    return record


def finish_phase(run, phase, names, **values):
    run = Path(run)
    record = {'schema_version': 1, 'status': 'passed',
              'manifest_sha256': sha256(run / 'manifest.json'),
              'files': file_records(run / phase, names), **values}
    write_new_json(run / phase / 'receipt.json', record)
    return record
