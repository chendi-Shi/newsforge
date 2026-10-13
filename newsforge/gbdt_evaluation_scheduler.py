"""Internal-only diagnostic for batching the unchanged GBDT evaluator by length."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from . import evaluation_scheduler
from .evaluate import select_validation_rows
from .evaluation_scheduler import _profile, scheduling_plan
from .gbdt import evaluate_gbdt
from .metrics import METRICS
from .store import Store


def evaluate_scheduled_gbdt(store, artifact, rows, *, batch_size=64, output=None, allow_test=False):
    """Change only batch membership; preserve every candidate and restore row order."""
    started = time.perf_counter()
    rows = np.asarray(rows, dtype=np.int64)
    # Reuse the frozen stable-order helper, but not its ranker-specific batch-32
    # slot accounting: the unchanged GBDT evaluator still uses batch size 64.
    order, _ = scheduling_plan(store, rows, kind="ranker", batch_size=batch_size)
    lengths = np.asarray(store.offsets[rows + 1] - store.offsets[rows], dtype=np.int64)
    def slots(values):
        return sum(int(values[start:start + batch_size].max()) * len(values[start:start + batch_size])
                   for start in range(0, len(values), batch_size))
    original_slots, scheduled_slots = slots(lengths), slots(lengths[order])
    report, ordered_values = evaluate_gbdt(store, artifact, rows[order], batch_size=batch_size,
                                          allow_test=allow_test)
    values = np.empty_like(ordered_values)
    values[order] = ordered_values
    if output is not None:
        np.save(output, values)
    actual = int(lengths.sum())
    report["evaluation_scheduling"] = {
        "strategy": "stable_candidate_count", "gbdt_only": True, "batch_size": batch_size,
        "impressions": len(rows), "candidate_exposures": actual,
        "original_candidate_slots": original_slots, "scheduled_candidate_slots": scheduled_slots,
        "original_padding_slots": original_slots - actual, "scheduled_padding_slots": scheduled_slots - actual,
        "same_population": True, "same_candidates_within_each_impression": True,
        "metric_rows_restored_to_original_order": True, "wrapper_seconds": time.perf_counter() - started}
    return report, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/gbdt.joblib"))
    parser.add_argument("--rows", type=int, choices=(512, 12000), default=512)
    args = parser.parse_args()
    available = psutil.virtual_memory().available
    if available < 1024 * 2**20:
        raise MemoryError(f"GBDT diagnostic waits for at least 1 GiB available RAM; current={available}")
    store = Store(args.data)
    artifact = joblib.load(args.checkpoint)
    fixed_rows = select_validation_rows(store, 12000)
    rows = fixed_rows[:args.rows]
    if len(rows) != args.rows or not np.all(store.split[rows] == 1):
        raise ValueError("GBDT diagnostic requires only the fixed internal-validation sample")
    with threadpool_limits(1):
        (original, original_values), original_cost = _profile(lambda:
            evaluate_gbdt(store, artifact, rows, batch_size=64, allow_test=False))
        (scheduled, scheduled_values), scheduled_cost = _profile(lambda:
            evaluate_scheduled_gbdt(store, artifact, rows, batch_size=64, allow_test=False))
    differences = {}
    for column, metric in enumerate(METRICS):
        first, second = original_values[:, column], scheduled_values[:, column]
        finite = np.isfinite(first) & np.isfinite(second)
        difference = np.abs(first[finite] - second[finite])
        differences[metric] = {"finite_pairs": int(finite.sum()),
            "same_nan_mask": bool(np.array_equal(np.isnan(first), np.isnan(second))),
            "max_absolute_difference": float(difference.max(initial=0)),
            "pairs_differing_more_than_1e_12": int((difference > 1e-12).sum()),
            "aggregate_mean_difference": scheduled["metrics"][metric]["mean"] - original["metrics"][metric]["mean"]}
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    code = Path(__file__).resolve().parent
    passed = bool(np.allclose(original_values, scheduled_values, atol=1e-12, rtol=0, equal_nan=True))
    result = {"scope": "fixed internal validation only; original GBDT model/features/evaluator unchanged",
        "rows": len(rows), "fixed_validation_rows": len(fixed_rows),
        "row_selection": "first rows in the sorted fixed seed-2026 internal-validation sample",
        "selected_row_indices_sha256": hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
        "checkpoint_sha256": sha(args.checkpoint), "data_manifest_sha256": sha(args.data / "manifest.json"),
        "evaluation_source_sha256": {name: sha(code / name) for name in ("gbdt.py", "store.py", "evaluate.py", "metrics.py")},
        "scheduler_source_sha256": sha(__file__), "order_helper_source_sha256": sha(evaluation_scheduler.__file__),
        "feature_mode": artifact.get("feature_mode", "similarity"), "feature_count": len(artifact["feature_names"]),
        "per_impression_metric_equivalence_atol": 1e-12, "metric_equivalence_passed": passed,
        "metric_differences": differences, "original": original_cost, "scheduled": scheduled_cost,
        "scheduling": scheduled["evaluation_scheduling"], "environment": {"threads": 1, "system_available_memory_before": available},
        "limitations": ["One original-then-scheduled pair on a shared desktop; load and cache warmth are uncontrolled.",
            "RSS is sampled in the actual process at 20 ms and includes mapped pages; it is not an exact allocation peak.",
            "Comparison is per-impression metrics, not feature matrices or classifier probabilities.",
            "No official-dev evaluation, model fitting or checkpoint selection occurred."]}
    output = Path(f"reports/gbdt_evaluation_scheduler_validation_{args.rows}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit("GBDT scheduling metrics differ; do not adopt this schedule")


if __name__ == "__main__":
    main()
