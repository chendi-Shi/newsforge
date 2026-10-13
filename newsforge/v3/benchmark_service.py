"""Isolated offline/HTTP validation of the deployment-gated v3 service.

The fixed 64-query workload is sampled from dev without click labels. This is
an end-to-end local service measurement; no speed ratio to historical v2 HTTP
measurements is calculated. The retrieval-only ABBA experiment is separate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from newsforge.v2 import benchmark as legacy
from newsforge.v2.retrieval import sha256_file


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def select_requests(processed_dir, v2_selection_path, count=64):
    if not isinstance(count, int) or isinstance(count, bool) or count < 8 or count % 8:
        raise ValueError("query count must be a positive multiple of eight")
    path = Path(processed_dir)
    selection = json.loads(Path(v2_selection_path).read_text(encoding="utf-8"))
    names = ("all_split.npy", "all_timestamps.npy", "all_history.npy", "news_ids.json")
    hashes = {name: sha256_file(path / name) for name in names}
    if any(hashes[name] != selection["bindings"]["processed_files"][name] for name in names):
        raise ValueError("HTTP workload inputs differ from frozen v2 data")
    arrays = {}
    try:
        for name in names[:-1]:
            arrays[name] = np.load(path / name, mmap_mode="r", allow_pickle=False)
        eligible = np.flatnonzero(arrays["all_split.npy"] == 2)
        if len(eligible) < count:
            raise ValueError("not enough dev requests for the declared workload")
        rows = np.sort(np.random.default_rng(2026).choice(eligible, count, replace=False))
        news_ids = json.loads((path / "news_ids.json").read_text(encoding="utf-8"))
        queries = [{"history": [news_ids[int(item)] for item in arrays["all_history.npy"][row] if item > 0],
                    "as_of": int(arrays["all_timestamps.npy"][row]), "top_k": 10} for row in rows]
        return rows, queries, hashes
    finally:
        for array in arrays.values():
            array._mmap.close()


def response_equal(reference, actual):
    """Require exact equality of every response field, including float scores."""
    return isinstance(reference, dict) and isinstance(actual, dict) and reference == actual


def measure_configuration(*args, **kwargs):
    # Reuse the frozen transport/workload implementation while injecting only
    # a stricter comparator. The scope spans all worker joins, then restores
    # the original function; no v2 file is edited.
    with patch.object(legacy, "response_equal", response_equal):
        return legacy.measure_configuration(*args, **kwargs)


def shared_arguments(args):
    return ["--processed-dir", str(args.processed_dir), "--artifact-dir", str(args.artifact_dir),
            "--temporal-dir", str(args.temporal_dir), "--selection", str(args.selection),
            "--confirm", str(args.confirm), "--v2-selection", str(args.v2_selection),
            "--fit-report", str(args.fit_report)]


def source_hashes():
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), package / "v2" / "benchmark.py"]
    paths += [package / "v3" / name for name in ("serve.py", "residual.py", "fast_retrieval.py")]
    paths += [package / "v2" / name for name in ("serve.py", "retrieval.py", "temporal.py", "ranker.py")]
    return {path.relative_to(package).as_posix(): sha256_file(path) for path in paths}


def offline_worker(args):
    from .serve import V2RecommendRequest, load_engine
    payload = json.loads(Path(args.raw_dir, "requests.json").read_text(encoding="utf-8"))
    engine = load_engine(args.processed_dir, args.artifact_dir, args.temporal_dir, args.selection,
                         args.confirm, args.v2_selection, args.fit_report)
    try:
        started = time.perf_counter()
        responses, checked_items = [], 0
        for query in payload["requests"]:
            response = engine.recommend(V2RecommendRequest(**query))
            ids = [item["item_id"] for item in response["items"]]
            if (len(set(ids)) != len(ids) or set(ids) & set(query["history"])
                    or any(engine.generator.first_seen[engine.id_to_index[item]] >= response["snapshot_cutoff"]
                           for item in ids)):
                raise RuntimeError("offline output violates uniqueness/seen/causal catalogue rules")
            checked_items += len(ids)
            responses.append(response)
        write_new(Path(args.raw_dir, "offline.json"), {
            "health": engine.health(), "responses": responses,
            "elapsed_seconds": time.perf_counter() - started,
            "causal_output_check": {"passed": True, "queries": len(responses), "items": checked_items,
                                    "checks": ["unique IDs", "all supplied history excluded", "first_seen strictly below cutoff"]}})
    finally:
        engine.close()


def run(args):
    raw_dir, report_path = Path(args.raw_dir), Path(args.report)
    if raw_dir.exists() or report_path.exists():
        raise FileExistsError("benchmark outputs already exist; select new raw/report paths")
    frozen_sources = source_hashes()
    evidence_paths = {"selection": args.selection, "confirm": args.confirm,
                      "fit_report": args.fit_report, "v2_selection": args.v2_selection}
    evidence_hashes = {name: sha256_file(path) for name, path in evidence_paths.items()}
    rows, queries, input_hashes = select_requests(args.processed_dir, args.v2_selection, args.queries)
    raw_dir.mkdir(parents=True, exist_ok=False)
    request_path = raw_dir / "requests.json"
    write_new(request_path, {"rows": rows.tolist(), "requests": queries})
    shared = shared_arguments(args)
    offline_cmd = [sys.executable, "-m", "newsforge.v3.benchmark_service", "--offline-worker",
                   "--raw-dir", str(raw_dir), *shared]
    print("Generating v3 offline references in one isolated worker", flush=True)
    with (raw_dir / "offline.log").open("x", encoding="utf-8") as log:
        worker = subprocess.Popen(offline_cmd, stdout=log, stderr=subprocess.STDOUT, **legacy.child_options())
        owned_worker = legacy.OwnedProcessTree(worker)
        try:
            if owned_worker.wait(timeout=600):
                raise RuntimeError("v3 offline worker failed; inspect its retained private log")
        finally:
            legacy.stop_owned_process(owned_worker)
    offline_path = raw_dir / "offline.json"
    offline = json.loads(offline_path.read_text(encoding="utf-8"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server_cmd = [sys.executable, "-m", "newsforge.v3.serve", *shared,
                  "--host", "127.0.0.1", "--port", str(port)]
    owned_server = None
    started = time.perf_counter()
    with (raw_dir / "server.log").open("x", encoding="utf-8") as log:
        try:
            server = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT, **legacy.child_options())
            owned_server = legacy.OwnedProcessTree(server)
            health = legacy.wait_ready(owned_server, base_url, timeout=240)
            actual_server = owned_server.verify_server(health)
            if health.get("api_version") != "v3" or health["model_version"] != offline["health"]["model_version"]:
                raise RuntimeError("offline/HTTP v3 model bindings differ")
            with legacy.ResourceSampler(actual_server) as resources:
                checks = legacy.api_contract_checks(base_url, health, queries[0])
                configs, raw_records = [], {}
                for batch_size, concurrency in ((1, 1), (8, 1), (1, 4), (8, 4)):
                    print(f"v3 HTTP benchmark: batch={batch_size}, concurrency={concurrency}", flush=True)
                    config, raw = measure_configuration(base_url, queries, offline["responses"],
                                                        batch_size=batch_size, concurrency=concurrency,
                                                        rounds=args.rounds, warmup=args.warmup)
                    configs.append(config)
                    raw_records[f"batch{batch_size}_concurrency{concurrency}"] = raw
                write_new(raw_dir / "http_records.json", raw_records)
            if source_hashes() != frozen_sources:
                raise RuntimeError("benchmark or runtime source changed during measurement")
            if {name: sha256_file(path) for name, path in evidence_paths.items()} != evidence_hashes:
                raise RuntimeError("selected release evidence changed during measurement")
            resource_report = resources.result()
            report = {
                "benchmark_version": 3, "created_at": datetime.now(timezone.utc).isoformat(),
                "sampling": {"split": "dev", "seed": 2026, "queries": len(rows),
                             "row_sha256": hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest(),
                             "timestamp_min": min(query["as_of"] for query in queries),
                             "timestamp_max": max(query["as_of"] for query in queries),
                             "source_arrays_sha256": input_hashes, "labels_deserialized_for_sampling": False},
                "protocol": {"transport": "localhost HTTP via urllib; fresh connection per request",
                             "request_order": "ascending sampled row IDs, same order in each round",
                             "cpu_threads": 1, "server_workers": 1,
                             "execution": "offline worker exits before HTTP server starts; cleanup is restricted to owned PID/create-time identities",
                             "latency_includes": "client JSON serialization, connection, server work, response JSON decoding",
                             "parity": "all response fields and numeric scores exact; no float tolerance",
                             "timing_limit": "Shared desktop CPU and fixed small workload; not production capacity, and not compared as a speed ratio to historical v2 HTTP timings."},
                "source_sha256": frozen_sources, "release_evidence_sha256": evidence_hashes,
                "processed_manifest_sha256": sha256_file(Path(args.processed_dir) / "manifest.json"),
                "private_evidence": {"requests_sha256": sha256_file(request_path),
                                     "offline_sha256": sha256_file(offline_path),
                                     "http_records_sha256": sha256_file(raw_dir / "http_records.json")},
                "health": {key: value for key, value in health.items() if key != "server_pid"},
                "api_contract_checks": checks, "offline_causal_output_check": offline["causal_output_check"],
                "configurations": configs, "resources": resource_report,
                "elapsed_seconds": time.perf_counter() - started}
            report["passed"] = (all(check["passed"] for check in checks)
                                and all(config["http_errors"] == config["offline_http_mismatches"] == 0 for config in configs)
                                and resource_report["sampling_error"] is None)
            write_new(report_path, report)
        finally:
            legacy.stop_owned_process(owned_server)
    if not report["passed"]:
        raise RuntimeError("v3 benchmark recorded API/parity/resource failures; evidence was retained")
    print(f"Wrote v3 HTTP evidence to {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--temporal-dir", default="data/temporal_v2")
    parser.add_argument("--selection", default="artifacts/v3/selection.json")
    parser.add_argument("--confirm", default="reports/v3/confirm.json")
    parser.add_argument("--v2-selection", default="artifacts/v2/selection.json")
    parser.add_argument("--fit-report", default="reports/v3/fit.json")
    parser.add_argument("--raw-dir", default="artifacts/v3/service_benchmark")
    parser.add_argument("--report", default="reports/v3/service_benchmark.json")
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--offline-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.queries < 8 or args.queries % 8 or args.rounds < 1 or args.warmup < 0:
        parser.error("queries must be a positive multiple of eight; rounds >= 1; warmup >= 0")
    if args.offline_worker:
        offline_worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
