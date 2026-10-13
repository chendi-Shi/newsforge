"""Internal-only, paired equivalence audit of evaluation route scheduling.

This tool never tunes a recipe or evaluates relevance. It compares every
CandidateBatch field, and times crossed ABBA sweeps on sixteen fixed internal
requests. Those sweep timings are not independent-request service latency.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .experiment import MIN_VALID_TIME, MAX_VALID_TIME, recipes
from .retrieval import CandidateBatch, CandidateGenerator, sha256_file
from .temporal import TemporalStore


def compare_batches(before, after):
    differences = []
    for field in fields(CandidateBatch):
        left, right = getattr(before, field.name), getattr(after, field.name)
        if isinstance(left, np.ndarray):
            equal = left.dtype == right.dtype and np.array_equal(left, right)
        else:
            equal = left == right
        if not equal:
            differences.append(field.name)
    max_score_error = (float(np.max(np.abs(before.scores - after.scores), initial=0))
                       if before.scores.shape == after.scores.shape else None)
    return differences, max_score_error


def verify(data="data/processed", artifacts="artifacts", temporal="data/temporal_v2",
           output="reports/v2/query_cache_equivalence.json"):
    output, data, artifacts, temporal = map(Path, (output, data, artifacts, temporal))
    if output.exists():
        raise FileExistsError("Preserve existing evidence; use a fresh report path")
    split = np.load(data / "all_split.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(data / "all_timestamps.npy", mmap_mode="r", allow_pickle=False)
    history = np.load(data / "all_history.npy", mmap_mode="r", allow_pickle=False)
    eligible = np.flatnonzero((split == 1) & (timestamps >= MIN_VALID_TIME)
                             & (timestamps < MAX_VALID_TIME))
    if len(eligible) < 16:
        raise ValueError("sixteen internal validation requests are required")
    rows = np.random.default_rng(2026).choice(eligible, 16, replace=False)
    rows = rows[np.argsort(timestamps[rows], kind="stable")]
    configs = recipes(48)
    expected = {}
    observations, mismatches = [], []
    max_error = 0.
    started = time.perf_counter()
    temporal_store = TemporalStore(temporal, processed_dir=data)
    generator = None
    try:
        generator = CandidateGenerator.from_paths(data, artifacts, temporal_store)
        with threadpool_limits(1):
            # Warm both source mappings and code before crossed observations.
            # No per-query routes are reused in this unmeasured pass.
            for row in rows:
                for name, config in configs.items():
                    expected[(int(row), name)] = generator.retrieve(history[row], int(timestamps[row]), config=config)
            for observation, cached in enumerate((False, True, True, False), 1):
                generator._cache.clear()
                generator._query_routes.clear()
                generator._route_query_key = None
                tick = time.perf_counter()
                compared = 0
                with patch.object(generator, "_dot", wraps=generator._dot) as dot:
                    for row in rows:
                        for name, config in configs.items():
                            actual = generator.retrieve(history[row], int(timestamps[row]), config=config,
                                                        reuse_query_routes=cached)
                            different, error = compare_batches(expected[(int(row), name)], actual)
                            compared += 1
                            if different:
                                # Retain aggregate location only; do not publish raw request IDs/history.
                                mismatches.append({"observation": observation, "recipe": name, "fields": different})
                            if error is not None:
                                max_error = max(max_error, error)
                    calls = dot.call_count
                observations.append({"observation": observation, "reuse_query_routes": cached,
                                     "seconds": time.perf_counter() - tick,
                                     "matrix_scoring_calls": calls, "batches_compared": compared})
                print(json.dumps(observations[-1]), flush=True)
            pools = [{key: pool.get(key) for key in ("internal_api", "prefix", "num_threads")}
                     for pool in threadpool_info()]
        uncached_calls = [record["matrix_scoring_calls"] for record in observations if not record["reuse_query_routes"]]
        cached_calls = [record["matrix_scoring_calls"] for record in observations if record["reuse_query_routes"]]
        if any(value != 1 for value in (pool["num_threads"] for pool in pools)):
            raise ValueError("equivalence timing protocol requires one native thread")
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "passed": not mismatches,
            "split": "internal_validation only", "minimum_timestamp": MIN_VALID_TIME,
            "maximum_timestamp_exclusive": MAX_VALID_TIME,
            "official_dev_metrics_evaluated": False,
            "feedback_artifact_integrity": "TemporalStore verifies source hashes, including labels; no labels used for selection, relevance or metrics in this audit.",
            "queries": len(rows), "seed": 2026,
            "row_indices_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
            "recipes": {name: asdict(config) for name, config in configs.items()},
            "candidate_batches_compared": sum(value["batches_compared"] for value in observations),
            "field_names_compared": [field.name for field in fields(CandidateBatch)],
            "comparison": "exact field equality, including ndarray dtype and every score, ID, source list, mask, cutoff, shortfall and metadata value",
            "mismatch_count": len(mismatches), "mismatches": mismatches[:10],
            "max_absolute_score_error": max_error,
            "warmup": "one uncached pass of the same 16 queries x 9 recipes; excluded from observation durations",
            "observation_order": "uncached, cached, cached, uncached (ABBA)",
            "timing_includes": "retrieval sweep, exact equality checks and _dot call-count instrumentation",
            "catalogue_and_query_caches_cleared_before_each_observation": True,
            "observations": observations,
            "matrix_scoring_call_reduction": 1 - sum(cached_calls) / max(sum(uncached_calls), 1),
            "native_thread_pools": pools,
            "bindings": generator.bindings,
            "input_array_sha256": {name: sha256_file(data / name) for name in
                                   ("all_split.npy", "all_timestamps.npy", "all_history.npy")},
            "source_sha256": {name: sha256_file(Path(__file__).parent / name) for name in
                              ("retrieval.py", "verify_cache.py", "experiment.py", "temporal.py")},
            "seconds_including_loading_and_warmup": time.perf_counter() - started,
            "limitations": ["Cached observations share computation across ablation recipes; they are not independent strategy or HTTP request latency.",
                            "Shared desktop timings are individual observations, not a hardware-independent speed guarantee.",
                            "Production service retains reuse_query_routes=False; only current-query semantic top IDs are reusable."],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        if mismatches:
            raise AssertionError(f"Query route cache changed outputs; inspect {output}")
        return report
    finally:
        if generator is not None:
            generator.close()
        temporal_store.close()
        for array in (split, timestamps, history):
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--temporal", default="data/temporal_v2")
    parser.add_argument("--output", default="reports/v2/query_cache_equivalence.json")
    args = parser.parse_args()
    result = verify(args.data, args.artifacts, args.temporal, args.output)
    print(json.dumps({"passed": result["passed"], "report": args.output,
                      "matrix_scoring_call_reduction": result["matrix_scoring_call_reduction"]}), flush=True)


if __name__ == "__main__":
    main()
