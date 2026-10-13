"""MIND impression-macro metrics; no sampled negatives in evaluation."""
import numpy as np
from scipy.stats import rankdata

METRICS = ["auc", "mrr", "ndcg5", "ndcg10"]


def impression_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("labels/scores must be equal-length, nonempty vectors")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("finite scores and binary labels required")
    positives = labels.sum()
    negatives = len(labels) - positives
    # AUC uses average ranks for ties, matching sklearn/official evaluator.
    auc = ((rankdata(scores)[labels == 1].sum() - positives * (positives + 1) / 2)
           / (positives * negatives)) if positives and negatives else np.nan
    # Tie policy is deterministic input order; score-unique models match official.
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    mrr = (ranked / np.arange(1, len(labels) + 1)).sum() / positives if positives else np.nan
    ndcg = []
    for k in [5, 10]:
        discounts = 1 / np.log2(np.arange(2, min(k, len(labels)) + 2))
        ideal = discounts[:min(int(positives), k)].sum()
        ndcg.append(float((ranked[:k] * discounts).sum() / ideal) if ideal else np.nan)
    return np.array([auc, mrr, *ndcg], dtype=np.float64)


def summarize(values):
    return {name: {"mean": float(np.nanmean(values[:, i])),
                   "eligible_impressions": int(np.isfinite(values[:, i]).sum())}
            for i, name in enumerate(METRICS)}


def user_cluster_interval(values_a, values_b, users, repeats=500, seed=2026):
    """Paired user bootstrap for the macro-impression mean metric difference."""
    _, inverse = np.unique(users, return_inverse=True)
    groups = int(inverse.max()) + 1
    delta = values_b - values_a
    sums = np.column_stack([np.bincount(inverse, weights=np.nan_to_num(delta[:, i]), minlength=groups)
                            for i in range(delta.shape[1])])
    counts = np.column_stack([np.bincount(inverse, weights=np.isfinite(delta[:, i]), minlength=groups)
                              for i in range(delta.shape[1])])
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(repeats):
        chosen = rng.integers(0, groups, size=groups)
        replicates.append(sums[chosen].sum(0) / np.maximum(counts[chosen].sum(0), 1))
    interval = np.quantile(replicates, [.025, .975], axis=0)
    return {name: {"delta": float(np.nanmean(delta[:, i])),
                   "ci95": interval[:, i].tolist()} for i, name in enumerate(METRICS)}
