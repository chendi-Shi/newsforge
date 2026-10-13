"""Candidate-count scheduling around the unchanged full-candidate evaluator.

Only ranker batches are reordered. Returned/saved metric rows always follow
the caller's original row order. The diagnostic CLI reads internal-validation
labels only and never evaluates official dev.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import threading
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

from .evaluate import evaluate, select_validation_rows
from .metrics import METRICS


def scheduling_plan(store, rows, *, kind, batch_size=128):
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or not len(rows):
        raise ValueError("evaluation rows must be a nonempty integer vector")
    if np.any(rows < 0) or np.any(rows + 1 >= len(store.offsets)) or batch_size < 1:
        raise ValueError("invalid evaluation rows or batch size")
    lengths = np.asarray(store.offsets[rows + 1] - store.offsets[rows], dtype=np.int64)
    if np.any(lengths < 1):
        raise ValueError("each evaluated impression requires at least one candidate")
    effective_batch_size = min(batch_size, 32) if kind == "ranker" else batch_size
    order = np.argsort(lengths, kind="stable") if kind == "ranker" else np.arange(len(rows))

    def slots(values):
        return sum(int(values[start:start + effective_batch_size].max())
                   * len(values[start:start + effective_batch_size])
                   for start in range(0, len(values), effective_batch_size))

    original_slots, scheduled_slots = slots(lengths), slots(lengths[order])
    real = int(lengths.sum())
    return order, {"strategy": "stable_candidate_count" if kind == "ranker" else "original_order",
                   "ranker_only": True, "impressions": len(rows), "batch_size": effective_batch_size,
                   "candidate_exposures": real, "original_candidate_slots": original_slots,
                   "scheduled_candidate_slots": scheduled_slots,
                   "original_padding_slots": original_slots - real,
                   "scheduled_padding_slots": scheduled_slots - real,
                   "candidate_slot_ratio": scheduled_slots / original_slots,
                   "padding_slot_ratio": ((scheduled_slots - real) / (original_slots - real)
                                          if original_slots > real else None),
                   "same_population": True, "same_candidates_within_each_impression": True,
                   "metric_rows_restored_to_original_order": True}


def evaluate_scheduled(store, model, kind, rows, *, output=None, batch_size=128):
    """Reuse evaluate/model/metrics exactly, then undo only the row permutation."""
    started = time.perf_counter()
    rows = np.asarray(rows)
    order, scheduling = scheduling_plan(store, rows, kind=kind, batch_size=batch_size)
    report, sorted_values = evaluate(store, model, kind, rows[order], batch_size=batch_size)
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, values)
    scheduling["wrapper_seconds"] = time.perf_counter() - started
    report["evaluation_scheduling"] = scheduling
    return report, values


def _profile(call):
    process = psutil.Process()
    samples = [process.memory_info().rss]
    stop = threading.Event()

    def observe():
        while not stop.wait(.02):
            samples.append(process.memory_info().rss)

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    started = time.perf_counter()
    try:
        value = call()
    finally:
        seconds = time.perf_counter() - started
        samples.append(process.memory_info().rss)
        stop.set()
        observer.join(timeout=1)
    return value, {"seconds": seconds, "initial_rss_bytes": samples[0],
                   "max_observed_rss_bytes": max(samples), "final_rss_bytes": samples[-1],
                   "rss_sampling_interval_ms": 20}


def main():
    from .store import Store
    from .train import build_model, training_source_digests, training_cache_signature

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/ranker_seed17.pt"))
    parser.add_argument("--rows", type=int, choices=[512, 12000], default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    threadpool_limits(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    store = Store(args.data)
    store.ensure_training_cache()
    manifest_hash = hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    if (checkpoint["variant"] != "ranker" or checkpoint["seed"] != 17
            or checkpoint["training_source_sha256"] != training_source_digests()
            or checkpoint["training_cache_signature"] != training_cache_signature(store)
            or checkpoint["data_manifest_sha256"] != manifest_hash
            or not checkpoint["trained_on_all_eligible_clicks"]
            or [record["epoch"] for record in checkpoint["history"]] != [1, 2]):
        raise ValueError("diagnostic requires the completed, provenance-matched ranker seed17 checkpoint")
    model, config, kind = build_model("ranker", store.features.shape[1])
    if checkpoint["config"] != config:
        raise ValueError("model configuration mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    fixed_rows = select_validation_rows(store, 12000)
    rows = fixed_rows[:args.rows]
    if len(rows) != args.rows or not np.all(store.split[rows] == 1):
        raise ValueError("diagnostic rows must belong exclusively to the fixed internal-validation sample")
    (original_report, original_values), original_cost = _profile(lambda: evaluate(store, model, kind, rows))
    (scheduled_report, scheduled_values), scheduled_cost = _profile(lambda: evaluate_scheduled(store, model, kind, rows))
    differences = {}
    for column, metric in enumerate(METRICS):
        first, second = original_values[:, column], scheduled_values[:, column]
        finite = np.isfinite(first) & np.isfinite(second)
        absolute = np.abs(first[finite] - second[finite])
        differences[metric] = {"finite_pairs": int(finite.sum()),
                               "same_nan_mask": bool(np.array_equal(np.isnan(first), np.isnan(second))),
                               "max_absolute_difference": float(absolute.max(initial=0)),
                               "pairs_differing_more_than_1e_12": int((absolute > 1e-12).sum()),
                               "aggregate_mean_difference": (scheduled_report["metrics"][metric]["mean"]
                                                             - original_report["metrics"][metric]["mean"])}
    passed = bool(np.allclose(original_values, scheduled_values, atol=1e-12, rtol=0, equal_nan=True))
    result = {"scope": "internal validation only; original evaluator/model/metrics unchanged",
              "run_id": checkpoint["run_id"], "selected_epoch": checkpoint["best_epoch"],
              "rows": len(rows), "fixed_validation_rows": len(fixed_rows),
              "row_selection": "first rows in the sorted fixed seed-2026 internal-validation sample",
              "selected_row_indices_sha256": hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "data_manifest_sha256": manifest_hash, "training_source_sha256": training_source_digests(),
              "scheduler_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "per_impression_metric_equivalence_atol": 1e-12, "metric_equivalence_passed": passed,
              "metric_differences": differences, "original": original_cost, "scheduled": scheduled_cost,
              "observed_wall_time_ratio_original_over_scheduled": original_cost["seconds"] / scheduled_cost["seconds"],
              "scheduling": scheduled_report["evaluation_scheduling"],
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "threads": 1, "device": "cpu", "system_available_memory_bytes": psutil.virtual_memory().available},
              "limitations": ["One original-order then scheduled-order pair on a shared desktop; background training and filesystem caching are uncontrolled.",
                              "Observed RSS is sampled at 20 ms and includes already-loaded data/models; it is not an allocation peak.",
                              "Float differences refer to per-impression metrics, not candidate logits.",
                              "No official-dev labels were evaluated, and no training or checkpoint selection was changed."]}
    output = args.output or Path(f"reports/evaluation_scheduler_validation_{args.rows}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit("Scheduling changed per-impression metrics; do not adopt this schedule")


if __name__ == "__main__":
    main()
