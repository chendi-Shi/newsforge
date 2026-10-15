"""Matched-strength component comparisons and confirmation-controlled serving."""
from pathlib import Path
import numpy as np
from newsforge.v3.residual import PairwiseResidual
from .common import METHODS, MODEL_NAMES, SEEDS


def load_adapted_models(run):
    return {name: [PairwiseResidual.load(Path(run) / 'fit/models' / f'{name}-seed{seed}.npz')
                   for seed in SEEDS] for name in MODEL_NAMES}


def ensemble(features, models):
    if len(models) != 3:
        raise ValueError('exactly three seed models are required')
    return np.mean(np.stack([np.clip(model.predict_raw(features), -1., 1.)
                             for model in models]), axis=0)


def rank_methods(english, local, history_length, models, frozen, delta, gate_passed):
    english, local = np.asarray(english), np.asarray(local)
    if (english.ndim != 2 or english.shape[1] != 13 or local.shape != english.shape
            or not np.isfinite(english).all() or not np.isfinite(local).all()):
        raise ValueError('matched finite [candidates,13] feature matrices required')
    if (isinstance(history_length, (bool, np.bool_))
            or not isinstance(history_length, (int, np.integer)) or not 0 <= history_length <= 20):
        raise ValueError('history length must be an integer in [0,20]')
    if delta not in (0, 2, 5, 10) or isinstance(delta, (bool, np.bool_)):
        raise ValueError('delta must belong to the frozen grid')
    if not isinstance(gate_passed, (bool, np.bool_)):
        raise ValueError('gate decision must be boolean')
    original = np.arange(len(english), dtype=np.int64)
    scores = np.broadcast_to(-original.astype(np.float64), (len(METHODS), len(original))).copy()
    if len(original) and history_length:
        factor = history_length / (history_length + 5.)
        old = ensemble(english, frozen)
        eng = ensemble(english, models['english'])
        loc = ensemble(local, models['local'])
        recent = ensemble(local, models['local_no_recent'])
        scores[1] += 10 * factor * old
        scores[2] += delta * factor * old
        scores[3] += delta * factor * eng
        scores[4] += delta * factor * loc
        scores[5] += delta * factor * recent
        scores[6] += delta * loc
    scores[7] = scores[4] if gate_passed else scores[0]
    orders = np.stack([np.lexsort((original, -score)) for score in scores]).astype(np.int16)
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite ranking scores')
    return orders, scores


def sample_pairs(queries, pools, sizes, history_lengths, seed):
    """Choose shared unobserved alternatives, never manufacture exposure labels."""
    queries, pools = np.asarray(queries), np.asarray(pools)
    sizes, history_lengths = np.asarray(sizes), np.asarray(history_lengths)
    n = len(queries)
    if (queries.shape != (n, 4) or pools.ndim != 2 or len(pools) != n
            or sizes.shape != (n,) or history_lengths.shape != (n,)):
        raise ValueError('training cache shape mismatch')
    rng = np.random.default_rng(seed)
    pairs, counts = [], {'requests': n, 'empty_history': 0, 'target_outside_pool': 0,
                         'no_alternatives': 0, 'eligible_requests': 0}
    for row, query in enumerate(queries):
        size = int(sizes[row])
        if not 0 <= size <= pools.shape[1]:
            raise ValueError('invalid candidate size')
        pool = pools[row, :size]
        if len(np.unique(pool)) != size or np.any(pool <= 0):
            raise ValueError('candidate IDs must be positive and unique')
        if not history_lengths[row]:
            counts['empty_history'] += 1
            continue
        positions = np.flatnonzero(pool == query[3])
        if not len(positions):
            counts['target_outside_pool'] += 1
            continue
        positive = int(positions[0])
        choices = np.flatnonzero(np.arange(size) != positive)
        if not len(choices):
            counts['no_alternatives'] += 1
            continue
        chosen = rng.choice(choices, size=min(8, len(choices)), replace=False)
        pairs.extend((row, positive, int(negative)) for negative in chosen)
        counts['eligible_requests'] += 1
    result = np.asarray(pairs, dtype=np.int32).reshape(-1, 3)
    counts['pairs'] = len(result)
    return result, counts


def select_delta(grid_results):
    if set(grid_results) != {'0', '2', '5', '10'}:
        raise ValueError('selection requires the complete fixed delta grid')
    return min((0, 2, 5, 10), key=lambda delta: (-grid_results[str(delta)]['hits10'], delta))


def confirmation_passes(delta, comparison):
    return bool(delta > 0 and comparison['difference'] > 0 and comparison['ci95'][0] > 0)
