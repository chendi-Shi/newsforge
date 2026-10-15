"""Owned localhost HTTP benchmark with saved-evaluation and offline parity."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError

import numpy as np

from newsforge.v2.benchmark import (OwnedProcessTree, ResourceSampler, child_options,
                                  http_json, stop_owned_process, wait_ready)
from .common import check_run, phase_receipt, sha256, write_new_json
from .serve import Engine, RecommendRequest


CONFIGURATIONS = ((1, 1), (8, 1), (1, 4), (8, 4))
QUERY_COUNT = 64
ROUNDS = 3
WARMUP = 8


def select_requests(run, count=QUERY_COUNT):
    """Evenly spaced rows of the already declared evaluation sample."""
    if type(count) is not int or count < 1:
        raise ValueError("request count must be a positive integer")
    queries = np.load(Path(run) / "prepare/evaluate/queries.npy", allow_pickle=False)
    if queries.ndim != 2 or queries.shape[1] != 4 or len(queries) < count:
        raise ValueError("insufficient or malformed saved evaluation queries")
    rows = np.linspace(0, len(queries) - 1, count, dtype=np.int64)
    requests = [{"user": int(queries[row, 2]), "timestamp": int(queries[row, 1]), "top_k": 10}
                for row in rows]
    return rows, requests


def compare_saved_evaluation(run, rows, responses):
    """Compare actual served identities and scores to the persisted evaluation."""
    run = Path(run)
    arrays = {name: np.load(run / folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)
              for folder, names in (("prepare/evaluate", ("queries", "pools", "sizes")),
                                    ("evaluate", ("orders", "scores"))) for name in names}
    try:
        if len(rows) != len(responses):
            raise ValueError("saved-evaluation response cardinality mismatch")
        items = 0
        for row, response in zip(rows, responses):
            size = int(arrays["sizes"][row])
            order = arrays["orders"][row, 7, :size]
            if not np.array_equal(np.sort(order), np.arange(size)):
                raise ValueError("saved deployed ranking is not a permutation")
            chosen = order[:10]
            expected = [{"item_id": int(arrays["pools"][row, position]),
                         "score": float(arrays["scores"][row, 7, position])} for position in chosen]
            if (response.get("items") != expected or response.get("top_k") != 10
                    or response.get("user") != int(arrays["queries"][row, 2])
                    or response.get("timestamp") != int(arrays["queries"][row, 1])):
                raise ValueError("served Top10 or score differs from the saved deployed evaluation")
            items += len(chosen)
        return {"passed": True, "queries": len(rows), "items": items,
                "method_index": 7, "comparison": "exact ordered Top10 item IDs and float64 scores",
                "reference": "persisted evaluation pools, orders and scores"}
    finally:
        for array in arrays.values():
            array._mmap.close()


def response_equal(expected, actual):
    return isinstance(expected, dict) and isinstance(actual, dict) and expected == actual


def measure_configuration(base_url, requests, references, *, batch_size, concurrency,
                          rounds=ROUNDS, warmup=WARMUP):
    if (batch_size not in (1, 8) or concurrency not in (1, 4) or rounds < 1 or warmup < 0
            or not requests or len(requests) != len(references) or len(requests) % batch_size):
        raise ValueError("invalid HTTP benchmark workload")
    batches = [list(range(index, index + batch_size)) for index in range(0, len(requests), batch_size)]
    endpoint = "/recommend" if batch_size == 1 else "/recommend_batch"

    def execute(indices):
        payload = requests[indices[0]] if batch_size == 1 else {"requests": [requests[index] for index in indices]}
        began = time.perf_counter()
        record = {"indices": indices, "request": payload, "query_count": len(indices),
                  "http_errors": 0, "failed_queries": 0, "mismatches": 0}
        try:
            status, body = http_json(base_url, endpoint, payload, timeout=60)
            record.update(seconds=time.perf_counter() - began, status=status, response=body)
            if status != 200:
                record.update(http_errors=1, failed_queries=len(indices))
                return record
            responses = [body] if batch_size == 1 else body.get("results", [])
            if len(responses) != len(indices) or (batch_size > 1 and body.get("batch_size") != len(indices)):
                raise ValueError("HTTP response cardinality mismatch")
            record["mismatches"] = sum(not response_equal(references[index], result)
                                       for index, result in zip(indices, responses))
            return record
        except (URLError, OSError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            record.update(seconds=time.perf_counter() - began, http_errors=1,
                          failed_queries=len(indices), error_type=type(exc).__name__)
            return record

    warmup_records = [execute(batches[index % len(batches)]) for index in range(warmup)]
    records, round_reports = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for index in range(rounds):
            began = time.perf_counter()
            current = list(executor.map(execute, batches))
            elapsed = time.perf_counter() - began
            records.extend(current)
            failures = sum(record["failed_queries"] for record in current)
            round_reports.append({"round": index + 1, "wall_seconds": elapsed,
                                  "successful_queries_per_second": (len(requests) - failures) / elapsed})
    elapsed = sum(record["wall_seconds"] for record in round_reports)
    failed_queries = sum(record["failed_queries"] for record in records)
    milliseconds = np.asarray([record["seconds"] * 1000 for record in records])
    summary = {"batch_size": batch_size, "concurrency": concurrency, "rounds": rounds,
               "measured_queries": len(requests) * rounds, "measured_http_requests": len(records),
               "warmup_http_requests": warmup, "warmup_queries": warmup * batch_size,
               "warmup_http_errors": sum(record["http_errors"] for record in warmup_records),
               "warmup_mismatches": sum(record["mismatches"] for record in warmup_records),
               "wall_seconds": elapsed, "successful_queries_per_second": (len(requests) * rounds - failed_queries) / elapsed,
               "http_errors": sum(record["http_errors"] for record in records),
               "failed_queries": failed_queries,
               "offline_http_mismatches": sum(record["mismatches"] for record in records),
               "latency_unit": "full HTTP request including every query in a batch",
               "latency_ms": {"p50": float(np.quantile(milliseconds, .50)),
                              "p95": float(np.quantile(milliseconds, .95)),
                              "p99": float(np.quantile(milliseconds, .99)),
                              "max": float(milliseconds.max())}, "round_results": round_reports}
    return summary, {"warmup": warmup_records, "measured": records}


def api_contract_checks(base_url, health, known):
    minimum, maximum = health["minimum_timestamp"], health["maximum_timestamp_exclusive"]
    cases = [
        ("user_zero", "/recommend", {**known, "user": 0}, 422),
        ("unknown_user", "/recommend", {**known, "user": health["maximum_user"] + 1}, 422),
        ("boolean_user", "/recommend", {**known, "user": True}, 422),
        ("before_coverage", "/recommend", {**known, "timestamp": minimum - 1}, 422),
        ("exclusive_upper_bound", "/recommend", {**known, "timestamp": maximum}, 422),
        ("boolean_timestamp", "/recommend", {**known, "timestamp": True}, 422),
        ("float_timestamp", "/recommend", {**known, "timestamp": float(minimum)}, 422),
        ("top_k_zero", "/recommend", {**known, "top_k": 0}, 422),
        ("top_k_101", "/recommend", {**known, "top_k": 101}, 422),
        ("empty_batch", "/recommend_batch", {"requests": []}, 422),
        ("oversized_batch", "/recommend_batch", {"requests": [known] * 33}, 422),
        ("missing_user", "/recommend", {"timestamp": minimum}, 422),
        ("unknown_field", "/recommend", {**known, "unrecognized": 1}, 422),
        ("minimum_timestamp", "/recommend", {**known, "timestamp": minimum}, 200),
        ("last_timestamp", "/recommend", {**known, "timestamp": maximum - 1}, 200),
        ("maximum_top_k", "/recommend", {**known, "top_k": 100}, 200),
        ("batch_single", "/recommend_batch", {"requests": [known]}, 200),
    ]
    boundary = (minimum // 3600 + 1) * 3600
    cases += [("hour_before", "/recommend", {**known, "timestamp": boundary - 1}, 200),
              ("hour_exact", "/recommend", {**known, "timestamp": boundary}, 200)]
    summaries, records = [], []
    for name, endpoint, payload, expected in cases:
        try:
            status, body = http_json(base_url, endpoint, payload, timeout=60)
            passed = status == expected
            if status == 200:
                responses = [body] if endpoint == "/recommend" else body.get("results", [])
                passed &= bool(responses) and all(value.get("version") == health["version"] for value in responses)
                if endpoint == "/recommend":
                    passed &= body.get("snapshot_cutoff") == payload["timestamp"] // 3600 * 3600 - 3600
                else:
                    passed &= body.get("batch_size") == 1 and len(responses) == 1
            summaries.append({"name": name, "expected_status": expected, "observed_status": status, "passed": bool(passed)})
            records.append({"name": name, "endpoint": endpoint, "request": payload, "response": body, "status": status})
        except (URLError, OSError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            summaries.append({"name": name, "expected_status": expected, "observed_status": None,
                              "passed": False, "error_type": type(exc).__name__})
            records.append({"name": name, "request": payload, "error_type": type(exc).__name__})
    return summaries, records


def benchmark(repo, base, run, private_dir, report):
    repo, base, run, private_dir, report = map(lambda value: Path(value).resolve(), (repo, base, run, private_dir, report))
    if private_dir.exists() or report.exists():
        raise FileExistsError("benchmark outputs must be new paths")
    manifest = check_run(repo, base, run, verify_base=True)
    phase_receipt(run, "evaluate")
    bindings = {"manifest_sha256": sha256(run / "manifest.json"),
                "decision_sha256": sha256(run / "confirm/decision.json"),
                "evaluation_receipt_sha256": sha256(run / "evaluate/receipt.json"),
                "sources": manifest["sources"],
                "transport_utilities_sha256": sha256(repo / "newsforge/v2/benchmark.py")}
    private_dir.mkdir(parents=True, exist_ok=False)
    rows, requests = select_requests(run)
    write_new_json(private_dir / "requests.json", {"rows": rows.tolist(), "requests": requests})
    engine = Engine(repo, base, run)
    try:
        responses = [engine.recommend(RecommendRequest(**value)) for value in requests]
        offline_health = engine.health()
        saved_check = compare_saved_evaluation(run, rows, responses)
    finally:
        engine.close()
    write_new_json(private_dir / "offline.json", {"health": offline_health, "responses": responses,
                                                  "saved_evaluation_check": saved_check})
    # Reserve an OS-selected unused loopback port until immediately before
    # launching. The health PID must then belong to this launch's process tree.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "newsforge.adaptation.serve", "--repo", str(repo),
               "--base", str(base), "--run", str(run), "--port", str(port)]
    owned = None
    began = time.perf_counter()
    with (private_dir / "server.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, **child_options())
        try:
            try:
                owned = OwnedProcessTree(process)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                raise
            health = wait_ready(owned, base_url, timeout=60)
            server = owned.verify_server(health)
            if ({key: value for key, value in health.items() if key != "server_pid"}
                    != {key: value for key, value in offline_health.items() if key != "server_pid"}):
                raise ValueError("offline and HTTP engine identity or bounds differ")
            with ResourceSampler(server) as resources:
                contracts, contract_records = api_contract_checks(base_url, health, requests[0])
                write_new_json(private_dir / "api_contracts.json", contract_records)
                configs, raw = [], {}
                for batch_size, concurrency in CONFIGURATIONS:
                    print(f"HTTP replay benchmark: batch={batch_size}, concurrency={concurrency}", flush=True)
                    summary, evidence = measure_configuration(base_url, requests, responses,
                        batch_size=batch_size, concurrency=concurrency)
                    configs.append(summary)
                    raw[f"batch{batch_size}_concurrency{concurrency}"] = evidence
                write_new_json(private_dir / "http_records.json", raw)
            # All four configurations include the same 64 requests. Compare
            # actual HTTP responses, not just direct-engine responses, against
            # the independently persisted deployed evaluation one more time.
            http_responses = [record.get("response", {})
                              for record in raw["batch1_concurrency1"]["measured"][:QUERY_COUNT]]
            try:
                http_saved_check = compare_saved_evaluation(run, rows, http_responses)
            except ValueError:
                http_saved_check = {"passed": False, "queries": len(rows),
                    "error": "HTTP Top10 identities or scores differ from saved deployed evaluation"}
            check_run(repo, base, run, verify_base=True)
            for key, path in (("manifest_sha256", run / "manifest.json"),
                              ("decision_sha256", run / "confirm/decision.json"),
                              ("evaluation_receipt_sha256", run / "evaluate/receipt.json")):
                if sha256(path) != bindings[key]:
                    raise ValueError("benchmark-bound evidence changed during HTTP measurement")
            resource_report = resources.result()
            passed = (http_saved_check["passed"] and all(check["passed"] for check in contracts)
                      and all(not any(config[key] for key in ("http_errors", "failed_queries", "offline_http_mismatches",
                                                             "warmup_http_errors", "warmup_mismatches")) for config in configs)
                      and resource_report["sampling_error"] is None and resource_report["resource_samples"] > 0)
            result = {"schema_version": 1, "passed": bool(passed),
                      "created_at": datetime.now(timezone.utc).isoformat(), "bindings": bindings,
                      "scope": "localhost retrospective replay of previously observed evaluation requests; not production capacity",
                      "sampling": {"queries": QUERY_COUNT, "selection": "64 evenly spaced saved evaluation rows",
                                   "row_array_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()},
                      "protocol": {"rounds": ROUNDS, "batch_sizes": [1, 8], "concurrency": [1, 4],
                          "numeric_threads": 1, "server_workers": 1, "startup_deadline_seconds": 60,
                          "transport": "loopback HTTP, fresh urllib connection per request",
                          "latency_includes": "request JSON encoding, connection, server replay/ranking, response decode",
                          "latency_excludes": "queue wait before the benchmark worker begins its request",
                          "execution": "offline engine is closed before the owned HTTP server launches"},
                      "health": {key: value for key, value in health.items() if key != "server_pid"},
                      "api_contract_checks": contracts, "api_contract_count": len(contracts),
                      "configurations": configs,
                      "measured_queries": sum(config["measured_queries"] for config in configs),
                      "measured_http_requests": sum(config["measured_http_requests"] for config in configs),
                      "offline_saved_evaluation_check": saved_check,
                      "http_saved_evaluation_check": http_saved_check,
                      "resources": {key: value for key, value in resource_report.items()
                                    if key not in ("server_pid", "server_create_time")},
                      "private_evidence": {name: sha256(private_dir / name)
                          for name in ("requests.json", "offline.json", "api_contracts.json", "http_records.json")},
                      "elapsed_seconds": time.perf_counter() - began}
            write_new_json(report, result)
        finally:
            stop_owned_process(owned)
    if not result["passed"]:
        raise RuntimeError("HTTP benchmark did not pass; retained aggregate report and private evidence")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "base", "run", "private-dir", "report"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = benchmark(args.repo, args.base, args.run, args.private_dir, args.report)
    print(f"HTTP benchmark passed: {result['measured_queries']} measured queries; {result['api_contract_count']} API checks", flush=True)


if __name__ == "__main__":
    main()
