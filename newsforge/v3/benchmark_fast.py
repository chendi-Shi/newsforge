"""Paired ABBA microbenchmark of v2 and exact v3 popular-only retrieval.

This measures retrieval, not HTTP capacity. It never selects a quality model
or consumes click-label values. All sampled requests are from mature internal
validation, and every CandidateBatch field is compared exactly outside the
timed region. Cold-derived-state and warm-hourly-state runs are separate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from newsforge.v2.retrieval import CandidateGenerator, sha256_file
from newsforge.v2.serve import V2ServiceConfig, verify_generator_selection
from newsforge.v2.temporal import TemporalStore
from .fast_retrieval import FastCandidateGenerator


ABBA = ("v2", "v3_fast", "v3_fast", "v2")


def batch_differences(expected, actual):
    """Return unequal dataclass fields; floats use equality, never tolerance."""
    differences = []
    for field in fields(expected):
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(left, np.ndarray):
            equal = (isinstance(right, np.ndarray) and left.dtype == right.dtype
                     and left.shape == right.shape and np.array_equal(left, right))
        else:
            equal = left == right
        if not equal:
            differences.append(field.name)
    return differences


def sample_queries(processed_dir, frozen_selection, count=64):
    """Hash-bind and sample only split, timestamp and history arrays."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1024:
        raise ValueError("request count must be an integer between 1 and 1024")
    root = Path(processed_dir)
    bounds = frozen_selection["selection_label_time_bounds"]
    low, high = int(bounds["minimum_inclusive"]), int(bounds["maximum_exclusive"])
    expected = frozen_selection["bindings"]["processed_files"]
    names = ("all_split.npy", "all_timestamps.npy", "all_history.npy")
    hashes = {name: sha256_file(root / name) for name in names}
    if any(hashes[name] != expected[name] for name in names):
        raise ValueError("benchmark request arrays differ from frozen v2 selection")
    arrays = {}
    try:
        for name in names:
            arrays[name] = np.load(root / name, mmap_mode="r", allow_pickle=False)
        split, timestamps, history = (arrays[name] for name in names)
        eligible = np.flatnonzero((split == 1) & (timestamps >= low) & (timestamps < high))
        if len(eligible) < count:
            raise ValueError("insufficient mature internal requests")
        rows = np.sort(np.random.default_rng(2026).choice(eligible, count, replace=False))
        queries = [(np.asarray(history[row][history[row] > 0], dtype=np.int64).tolist(),
                    int(timestamps[row])) for row in rows]
        query_digest = hashlib.sha256(json.dumps(queries, separators=(",", ":")).encode()).hexdigest()
        audit = {"count": count, "seed": 2026, "split": 1,
                 "minimum_timestamp_inclusive": low, "maximum_timestamp_exclusive": high,
                 "sampled_timestamp_min": min(query[1] for query in queries),
                 "sampled_timestamp_max": max(query[1] for query in queries),
                 "unique_snapshot_hours": len({query[1] // 3600 for query in queries}),
                 "row_indices_sha256": hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest(),
                 "queries_sha256": query_digest, "source_arrays_sha256": hashes,
                 "labels_deserialized": False, "labels_used": False}
        return queries, audit
    finally:
        for array in arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()


def clear_cache(generator):
    if isinstance(generator, FastCandidateGenerator):
        generator.clear_cache()
    else:
        # The benchmark runs a private, sequential instance of frozen v2.
        generator._cache.clear()
        generator._query_routes.clear()
        generator._route_query_key = None


def cache_bytes(generator):
    if isinstance(generator, FastCandidateGenerator):
        return generator.cache_info()["array_bytes"]
    return sum(array.nbytes for entry in generator._cache.values() for array in entry)


def measure_pass(generator, queries, references, *, mode, implementation, cycle, position):
    """Measure retrieval alone; exclude priming, parity checks and RSS reads."""
    if mode not in ("cold", "warm") or len(queries) != len(references) or not queries:
        raise ValueError("invalid benchmark pass")
    clear_cache(generator)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    available_before = psutil.virtual_memory().available
    started = time.perf_counter()
    pass_cpu_start = time.process_time_ns()
    wall_ns, cpu_ns, mismatch_fields = [], [], {}
    rss_peak_boundary, cache_peak = rss_before, 0
    for (history, as_of), reference in zip(queries, references):
        if mode == "cold":
            # This is a cold derived catalogue cache, not a cold OS file cache.
            clear_cache(generator)
        else:
            # A matching hourly entry is guaranteed warm even when the 64
            # sampled requests span more hours than the LRU's capacity.
            generator.retrieve(history, as_of)
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        result = generator.retrieve(history, as_of)
        wall_ns.append(time.perf_counter_ns() - wall_start)
        cpu_ns.append(time.process_time_ns() - cpu_start)
        for field in batch_differences(reference, result):
            mismatch_fields[field] = mismatch_fields.get(field, 0) + 1
        rss_peak_boundary = max(rss_peak_boundary, process.memory_info().rss)
        cache_peak = max(cache_peak, cache_bytes(generator))
    total_pass_seconds = time.perf_counter() - started
    elapsed = sum(wall_ns) / 1e9
    latencies = np.asarray(wall_ns, dtype=np.float64) / 1e6
    report = {"implementation": implementation, "mode": mode, "cycle": cycle,
              "abba_position": position, "measured_queries": len(queries),
              "measured_query_wall_seconds": elapsed,
              "measured_process_cpu_seconds": sum(cpu_ns) / 1e9,
              "measured_queries_per_second": len(queries) / elapsed,
              "latency_ms": {"p50": float(np.quantile(latencies, .5)),
                             "p95": float(np.quantile(latencies, .95)),
                             "p99": float(np.quantile(latencies, .99)),
                             "max": float(latencies.max())},
              "total_pass_wall_seconds_including_setup_and_checks": total_pass_seconds,
              "total_pass_process_cpu_seconds_including_setup_and_checks": (time.process_time_ns() - pass_cpu_start) / 1e9,
              "rss_before_bytes": rss_before, "rss_after_bytes": process.memory_info().rss,
              "max_query_boundary_rss_bytes": rss_peak_boundary,
              "derived_cache_array_bytes_peak": cache_peak,
              "available_memory_before_bytes": available_before,
              "available_memory_after_bytes": psutil.virtual_memory().available,
              "exact_field_comparisons": len(queries) * len(fields(references[0])),
              "mismatch_fields": mismatch_fields,
              "priming_queries_excluded_from_measurement": len(queries) if mode == "warm" else 0,
              "per_query_latency_ms": latencies.tolist(),
              "per_query_process_cpu_ms": (np.asarray(cpu_ns, dtype=np.float64) / 1e6).tolist()}
    if isinstance(generator, FastCandidateGenerator):
        report["fast_cache_cumulative_statistics"] = generator.cache_info()
    return report


def summarize_passes(passes):
    output = {}
    for mode in ("cold", "warm"):
        summary = {}
        for implementation in ("v2", "v3_fast"):
            values = [entry for entry in passes if entry["mode"] == mode
                      and entry["implementation"] == implementation]
            latencies = np.concatenate([entry["per_query_latency_ms"] for entry in values])
            elapsed = sum(entry["measured_query_wall_seconds"] for entry in values)
            summary[implementation] = {
                "measured_queries": sum(entry["measured_queries"] for entry in values),
                "measured_query_wall_seconds": elapsed,
                "measured_process_cpu_seconds": sum(entry["measured_process_cpu_seconds"] for entry in values),
                "measured_queries_per_second": len(latencies) / elapsed,
                "latency_ms": {"p50": float(np.quantile(latencies, .5)),
                               "p95": float(np.quantile(latencies, .95)),
                               "p99": float(np.quantile(latencies, .99))},
                "maximum_derived_cache_array_bytes": max(entry["derived_cache_array_bytes_peak"] for entry in values),
                "maximum_query_boundary_rss_bytes": max(entry["max_query_boundary_rss_bytes"] for entry in values),
            }
        summary["observed_wall_speedup_v2_over_v3"] = (
            summary["v2"]["measured_query_wall_seconds"] / summary["v3_fast"]["measured_query_wall_seconds"])
        measured_cpu = summary["v3_fast"]["measured_process_cpu_seconds"]
        summary["observed_instrumented_query_cpu_ratio_v2_over_v3"] = (
            summary["v2"]["measured_process_cpu_seconds"] / measured_cpu if measured_cpu > 0 else None)
        summary["paired_abba_cycles"] = []
        for cycle in sorted({entry["cycle"] for entry in passes}):
            subset = [entry for entry in passes if entry["mode"] == mode and entry["cycle"] == cycle]
            times = {name: sum(entry["measured_query_wall_seconds"] for entry in subset
                               if entry["implementation"] == name) for name in ("v2", "v3_fast")}
            summary["paired_abba_cycles"].append({"cycle": cycle,
                                                 "observed_wall_speedup_v2_over_v3": times["v2"] / times["v3_fast"]})
        output[mode] = summary
    return output


def run_benchmark(processed_dir="data/processed", artifact_dir="artifacts",
                  temporal_dir="data/temporal_v2", selection_path="artifacts/v2/selection.json",
                  *, count=64, cycles=3):
    if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= 20:
        raise ValueError("ABBA cycles must be an integer between 1 and 20")
    started = time.perf_counter()
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    config = V2ServiceConfig.from_selection(selection_path)
    if config.candidate_config.channels != ("popular",) or config.use_reranker:
        raise ValueError("the frozen selection must be popular_only without reranking")
    queries, sample_audit = sample_queries(processed_dir, selection, count)
    sources = {"benchmark_fast.py": Path(__file__), "fast_retrieval.py": Path(__file__).with_name("fast_retrieval.py"),
               **{f"v2/{name}.py": Path(__file__).parents[1] / "v2" / f"{name}.py"
                  for name in ("retrieval", "temporal", "serve")}}
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    generators, temporal, passes = {}, None, []
    with threadpool_limits(limits=1):
        try:
            temporal = TemporalStore(temporal_dir, processed_dir=processed_dir, verify=True)
            for name, generator_class in (("v2", CandidateGenerator), ("v3_fast", FastCandidateGenerator)):
                generator = generator_class.from_paths(processed_dir, artifact_dir, temporal,
                                                       encoder_path=config.encoder_path, config=config.candidate_config)
                generators[name] = generator
                verify_generator_selection(config, generator)
            if generators["v2"].bindings != generators["v3_fast"].bindings:
                raise ValueError("reference/optimized artifact bindings differ")
            references = [generators["v2"].retrieve(history, as_of) for history, as_of in queries]
            for mode in ("cold", "warm"):
                for cycle in range(1, cycles + 1):
                    for position, name in enumerate(ABBA, 1):
                        for generator in generators.values():
                            clear_cache(generator)
                        report = measure_pass(generators[name], queries, references,
                                              mode=mode, implementation=name, cycle=cycle, position=position)
                        passes.append(report)
                        print(f"{mode} ABBA {cycle}/{cycles} position {position} {name}: "
                              f"{report['measured_query_wall_seconds']:.6f}s, "
                              f"mismatch fields={report['mismatch_fields']}", flush=True)
            after = {name: sha256_file(path) for name, path in sources.items()}
            if after != source_hashes:
                raise ValueError("benchmark/runtime source changed during measurement")
            if sha256_file(selection_path) != config.selection_sha256:
                raise ValueError("frozen selection changed during measurement")
            mismatches = sum(sum(report["mismatch_fields"].values()) for report in passes)
            result = {
                "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": "frozen_v2_vs_exact_v3_popular_only_retrieval",
                "selection_sha256": config.selection_sha256,
                "candidate_config": asdict(config.candidate_config),
                "input_bindings": generators["v2"].bindings,
                "generator_versions": {name: generator.version for name, generator in generators.items()},
                "source_sha256": source_hashes, "sample": sample_audit,
                "protocol": {"sequence": list(ABBA), "cycles_per_mode": cycles,
                             "modes": ["cold", "warm"], "threads": 1,
                             "cold": "Clear only the derived catalogue cache before every measured query; OS/file caches are not flushed.",
                             "warm": "Prime the same request once immediately before each timed retrieval; only hourly state is reusable, query-route reuse is disabled.",
                             "timed_region": "generator.retrieve only; excludes prime, cache clearing, parity, RSS and JSON work",
                             "memory": "Same process with both input generators loaded; all caches cleared between passes. RSS is whole-process and cannot be attributed to a single generator.",
                             "rss_sampling": "At pass boundaries and after each query; temporary allocation peaks may be missed.",
                             "cpu_measurement": "Per-query process CPU readings can be quantized by OS accounting, especially on Windows, and their sum may undercount short queries. Zero denominators produce a null CPU ratio. Whole-pass CPU is also reported but includes prime/setup/checks.",
                             "labels": "No labels are deserialized or used. TemporalStore integrity verification hashes source files, including label-file bytes.",
                             "statistical_scope": "Observed ratios of paired local runs, without a population-level speedup confidence interval."},
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "platform": platform.platform(), "processor": platform.processor(),
                                "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                                "native_thread_pools": [{key: pool.get(key) for key in ("internal_api", "prefix", "num_threads")}
                                                        for pool in threadpool_info()]},
                "equivalence": {"exact": mismatches == 0, "mismatch_field_occurrences": mismatches,
                                "candidate_batches_compared": sum(report["measured_queries"] for report in passes),
                                "dataclass_fields_per_batch": len(fields(references[0])),
                                "fields": [field.name for field in fields(references[0])],
                                "numeric_tolerance": 0},
                "summary": summarize_passes(passes), "passes": passes,
                "end_to_end_wall_seconds_including_loading_verification": time.perf_counter() - started,
                "limitations": ["CPU desktop load is uncontrolled; these are local observations, not production capacity.",
                                "Retrieval microbenchmark omits HTTP, JSON, queueing and optional v3 reranking.",
                                "Cold means derived state cold, not a cold process or a cold OS page cache.",
                                "Warm timings exclude an explicit per-query priming request; priming cost must be paid in a real service."]}
            return result
        finally:
            for generator in generators.values():
                generator.close()
            if temporal is not None:
                temporal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--temporal-dir", default="data/temporal_v2")
    parser.add_argument("--selection", default="artifacts/v2/selection.json")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--output", default="reports/v3/fast_retrieval_benchmark.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("benchmark evidence already exists; choose a new output path")
    report = run_benchmark(args.processed_dir, args.artifact_dir, args.temporal_dir, args.selection,
                           count=args.count, cycles=args.cycles)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if not report["equivalence"]["exact"]:
        raise RuntimeError("exact CandidateBatch parity failed; failure evidence was retained")
    print(json.dumps({"output": str(output), "equivalence": report["equivalence"],
                      "summary": report["summary"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
