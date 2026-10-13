"""Reproducible localhost HTTP benchmark of the frozen v2 serving recipe.

Offline references and the HTTP engine run in separate child processes, never
at the same time. Only aggregate evidence is published; requests and item IDs
remain in the ignored artifacts directory. No production capacity is claimed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import psutil


MIN_VALID_TIME = 1573722000
COMPARISON_FIELDS = ("model_version", "candidate_config_version", "snapshot_version",
                     "snapshot_index", "snapshot_cutoff", "as_of", "candidate_count",
                     "shortfall", "retrieval_shortfall")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def select_requests(processed_dir, count=64):
    """Sample internal validation requests, without loading any click labels."""
    if count < 1:
        raise ValueError("request count must be positive")
    path = Path(processed_dir)
    arrays = {name: np.load(path / f"all_{name}.npy", mmap_mode="r")
              for name in ("split", "timestamps", "history")}
    try:
        eligible = np.flatnonzero((arrays["split"] == 1) & (arrays["timestamps"] >= MIN_VALID_TIME))
        if len(eligible) < count:
            raise ValueError("not enough internal validation requests for the declared workload")
        rows = np.sort(np.random.default_rng(2026).choice(eligible, count, replace=False))
        news_ids = json.loads((path / "news_ids.json").read_text(encoding="utf-8"))
        queries = [{"history": [news_ids[int(item)] for item in arrays["history"][row] if item > 0],
                    "as_of": int(arrays["timestamps"][row]), "top_k": 10} for row in rows]
        return rows, queries
    finally:
        for value in arrays.values():
            value._mmap.close()


def response_equal(reference, actual):
    """Compare ranking identities and provenance; scores are checked separately."""
    try:
        if any(reference[field] != actual[field] for field in COMPARISON_FIELDS):
            return False
        left, right = reference["items"], actual["items"]
        if len(left) != len(right):
            return False
        for expected, observed in zip(left, right):
            if expected["item_id"] != observed["item_id"] or expected["sources"] != observed["sources"]:
                return False
            for key in ("retrieval_score", "rerank_score"):
                a, b = expected[key], observed[key]
                if (a is None) != (b is None):
                    return False
                if a is not None and (not np.isfinite(b) or not np.isclose(a, b, rtol=1e-6, atol=1e-7)):
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def http_json(base_url, endpoint, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(base_url + endpoint, data=data,
                      headers={"Content-Type": "application/json", "Connection": "close"},
                      method="GET" if payload is None else "POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        try:
            body = json.load(error)
        except (ValueError, UnicodeDecodeError):
            body = {"error": "non-JSON HTTP error"}
        finally:
            error.close()
        return error.code, body


def measure_configuration(base_url, queries, references, *, batch_size, concurrency,
                          rounds=3, warmup=8):
    if batch_size < 1 or concurrency < 1 or rounds < 1 or warmup < 0 or len(queries) != len(references):
        raise ValueError("invalid benchmark configuration")
    if not queries or len(queries) % batch_size:
        raise ValueError("request count must be divisible by batch size")
    batches = [list(range(i, i + batch_size)) for i in range(0, len(queries), batch_size)]
    endpoint = "/recommend" if batch_size == 1 else "/recommend_batch"

    def execute(indices):
        payload = queries[indices[0]] if batch_size == 1 else {"requests": [queries[i] for i in indices]}
        started = time.perf_counter()
        try:
            status, body = http_json(base_url, endpoint, payload)
            seconds = time.perf_counter() - started
            if status != 200:
                return {"seconds": seconds, "http_error": 1, "query_errors": len(indices),
                        "mismatches": 0, "shortfalls": 0, "retrieval_shortfalls": 0,
                        "status": status, "body": body}
            results = [body] if batch_size == 1 else body.get("results", [])
            if len(results) != len(indices) or (batch_size != 1 and body.get("batch_size") != len(indices)):
                raise ValueError("response cardinality mismatch")
            return {"seconds": seconds, "http_error": 0, "query_errors": 0,
                    "mismatches": sum(not response_equal(references[i], value) for i, value in zip(indices, results)),
                    "shortfalls": sum(int(value["shortfall"]) > 0 for value in results),
                    "retrieval_shortfalls": sum(int(value["retrieval_shortfall"]) > 0 for value in results),
                    "status": status}
        except (URLError, OSError, TimeoutError, ValueError, KeyError, TypeError) as error:
            return {"seconds": time.perf_counter() - started, "http_error": 1,
                    "query_errors": len(indices), "mismatches": 0, "shortfalls": 0,
                    "retrieval_shortfalls": 0, "error": type(error).__name__}

    warmup_results = [execute(batches[i % len(batches)]) for i in range(warmup)]
    if any(value["http_error"] or value["mismatches"] for value in warmup_results):
        raise RuntimeError("HTTP warmup failed or disagreed with offline reference")
    # Reuse worker threads across rounds. Every round includes every sampled query.
    records, round_reports = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for index in range(rounds):
            started = time.perf_counter()
            current = list(executor.map(execute, batches))
            elapsed = time.perf_counter() - started
            records.extend(current)
            successful = len(queries) - sum(value["query_errors"] for value in current)
            round_reports.append({"round": index + 1, "wall_seconds": elapsed,
                                  "successful_queries_per_second": successful / elapsed})
    total_seconds = sum(value["wall_seconds"] for value in round_reports)
    failed_queries = sum(value["query_errors"] for value in records)
    latency_ms = np.asarray([value["seconds"] * 1000 for value in records])
    aggregate = {"batch_size": batch_size, "concurrency": concurrency, "rounds": rounds,
                 "warmup_http_requests": warmup, "measured_http_requests": len(records),
                 "measured_queries": len(queries) * rounds, "wall_seconds": total_seconds,
                 "successful_queries_per_second": (len(queries) * rounds - failed_queries) / total_seconds,
                 "latency_unit": "one HTTP request, including all queries in a batch",
                 "latency_ms": {"p50": float(np.quantile(latency_ms, .5)),
                                "p95": float(np.quantile(latency_ms, .95)),
                                "p99": float(np.quantile(latency_ms, .99)),
                                "max": float(latency_ms.max())},
                 "http_errors": sum(value["http_error"] for value in records),
                 "failed_queries": failed_queries,
                 "offline_http_mismatches": sum(value["mismatches"] for value in records),
                 "queries_with_output_shortfall": sum(value["shortfalls"] for value in records),
                 "queries_with_retrieval_shortfall": sum(value["retrieval_shortfalls"] for value in records),
                 "round_results": round_reports}
    return aggregate, {"warmup": warmup_results, "measured": records}


def api_contract_checks(base_url, health, known_query):
    minimum, maximum = int(health["minimum_as_of"]), int(health["maximum_as_of"])
    checks = []
    cases = [("oov_history", "/recommend", {"history": ["__NEWSFORGE_OOV__"]}, 422),
             ("pad_history", "/recommend", {"history": ["[PAD]"]}, 422),
             ("negative_timestamp", "/recommend", {"as_of": -1}, 422),
             ("before_coverage", "/recommend", {"as_of": minimum - 1}, 422),
             ("after_coverage", "/recommend", {"as_of": maximum + 1}, 422),
             ("invalid_top_k", "/recommend", {"top_k": 0}, 422),
             ("empty_batch", "/recommend_batch", {"requests": []}, 422),
             ("oversized_batch", "/recommend_batch", {"requests": [{}] * 33}, 422),
             ("empty_history", "/recommend", {"history": [], "as_of": known_query["as_of"]}, 200),
             ("coverage_minimum", "/recommend", {"as_of": minimum}, 200),
             ("coverage_maximum", "/recommend", {"as_of": maximum}, 200)]
    for name, endpoint, payload, expected_status in cases:
        status, body = http_json(base_url, endpoint, payload)
        ok = status == expected_status
        if status == 200:
            ok &= body.get("model_version") == health["model_version"]
        checks.append({"name": name, "expected_status": expected_status, "observed_status": status,
                       "passed": bool(ok)})
    boundary = ((int(known_query["as_of"]) // 3600) + 1) * 3600
    if not minimum < boundary <= maximum:
        raise ValueError("sampled request cannot exercise an in-coverage hour boundary")
    for timestamp in (boundary - 1, boundary):
        status, body = http_json(base_url, "/recommend", {"as_of": timestamp})
        expected_cutoff = (timestamp // 3600) * 3600 - 3600
        checks.append({"name": "hour_boundary_minus_one" if timestamp < boundary else "hour_boundary_exact",
                       "expected_status": 200, "observed_status": status,
                       "passed": status == 200 and body.get("snapshot_cutoff") == expected_cutoff,
                       "expected_cutoff": expected_cutoff, "observed_cutoff": body.get("snapshot_cutoff")})
    return checks


class ResourceSampler:
    def __init__(self, process):
        # Retain the already verified Process identity. Reopening a bare PID
        # could accidentally sample an unrelated process after PID reuse.
        self.process = process
        self.identity = (process.pid, process.create_time())
        self.stop_event = threading.Event()
        self.rss_peak = 0
        self.available_min = psutil.virtual_memory().available
        self.samples = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                if (not self.process.is_running()
                        or self.process.create_time() != self.identity[1]):
                    self.error = "server identity is no longer alive"
                    return
                self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
                self.available_min = min(self.available_min, psutil.virtual_memory().available)
                self.samples += 1
            except psutil.Error as error:
                self.error = type(error).__name__
                return
            self.stop_event.wait(.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def result(self):
        return {"server_peak_rss_bytes": self.rss_peak, "minimum_system_available_bytes": self.available_min,
                "resource_samples": self.samples, "sample_interval_seconds": .1,
                "sampling_error": self.error,
                "measurement_window": "healthy server: API checks and all four HTTP cases; startup excluded",
                "server_pid": self.identity[0], "server_create_time": self.identity[1],
                "rss_scope": "one owned HTTP server process; excludes benchmark client"}


def child_options():
    options = {"stdin": subprocess.DEVNULL,
               "env": {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    return options


class OwnedProcessTree:
    """Remember only this Popen launch and its observed descendants.

    Adapted from v1 benchmark_service._OwnedProcessTree without importing its
    Torch/Faiss dependencies. Windows venv launchers may own the actual Python
    server as a child. Retained (PID, creation time) identities survive wrapper
    exit and psutil's is_running/signaling guards reject recycled PIDs.
    """
    def __init__(self, process):
        self.process = process
        root = psutil.Process(process.pid)
        self.owned = {(root.pid, root.create_time()): root}

    @staticmethod
    def _alive(identity, process):
        try:
            return process.is_running() and process.create_time() == identity[1]
        except psutil.NoSuchProcess:
            return False

    def refresh(self):
        for identity, parent in list(self.owned.items()):
            if not self._alive(identity, parent):
                continue
            try:
                for child in parent.children(recursive=True):
                    self.owned[(child.pid, child.create_time())] = child
            except psutil.NoSuchProcess:
                continue

    def alive_processes(self):
        return [process for identity, process in self.owned.items() if self._alive(identity, process)]

    def verify_server(self, health):
        pid = health.get("server_pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
            raise RuntimeError("server health did not provide a valid server_pid")
        self.refresh()
        for identity, process in self.owned.items():
            if identity[0] == pid and self._alive(identity, process):
                return process
        raise RuntimeError("health server_pid is not owned by this launched process tree")

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            self.refresh()
            code = self.process.poll()
            # A successful wrapper alone does not prove that its worker exited.
            if code is not None and (code != 0 or not self.alive_processes()):
                return code
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("owned process tree", timeout)
            time.sleep(.1)

    def close(self, terminate_timeout=5, kill_timeout=3):
        self.refresh()
        targets = list(reversed(self.alive_processes()))
        for process in targets:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets, timeout=terminate_timeout)
        for process in alive:
            # psutil's Process.kill also checks its retained identity.
            try:
                identity = (process.pid, process.create_time())
                if identity in self.owned and self._alive(identity, process):
                    process.kill()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(alive, timeout=kill_timeout)
        if remaining:
            raise RuntimeError("Owned benchmark processes did not exit")
        self.process.wait(timeout=kill_timeout)


def stop_owned_process(owned):
    if owned is not None:
        if not isinstance(owned, OwnedProcessTree):
            raise TypeError("cleanup requires the tracked tree from this launch")
        owned.close()


def wait_ready(owned, base_url, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        owned.refresh()
        if owned.process.poll() is not None and not owned.alive_processes():
            raise RuntimeError(f"owned HTTP server exited during startup with code {owned.process.returncode}")
        try:
            status, health = http_json(base_url, "/health", timeout=1)
            if status == 200 and health.get("status") == "ok":
                owned.verify_server(health)
                return health
        except (URLError, OSError, TimeoutError, ValueError):
            pass
        time.sleep(.2)
    raise TimeoutError("owned HTTP server did not become healthy within startup timeout")


def offline_worker(args):
    from .serve import V2RecommendRequest, load_engine
    payload = json.loads(Path(args.raw_dir, "requests.json").read_text(encoding="utf-8"))
    engine = load_engine(args.processed_dir, args.artifact_dir, args.temporal_dir, args.selection)
    try:
        started = time.perf_counter()
        responses = [engine.recommend(V2RecommendRequest(**query)) for query in payload["requests"]]
        write_json(Path(args.raw_dir, "offline.json"), {"health": engine.health(), "responses": responses,
                                                       "elapsed_seconds": time.perf_counter() - started})
    finally:
        engine.close()


def run(args):
    raw_dir, report_path = Path(args.raw_dir), Path(args.report)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows, queries = select_requests(args.processed_dir, args.queries)
    request_path = raw_dir / "requests.json"
    write_json(request_path, {"rows": rows.tolist(), "requests": queries})
    shared = ["--processed-dir", str(args.processed_dir), "--artifact-dir", str(args.artifact_dir),
              "--temporal-dir", str(args.temporal_dir), "--selection", str(args.selection)]
    offline_cmd = [sys.executable, "-m", "newsforge.v2.benchmark", "--offline-worker",
                   "--raw-dir", str(raw_dir), *shared]
    print("Generating offline references in one isolated worker", flush=True)
    with (raw_dir / "offline.log").open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(offline_cmd, stdout=log, stderr=subprocess.STDOUT, **child_options())
        owned_worker = OwnedProcessTree(worker)
        try:
            if owned_worker.wait(timeout=600):
                raise RuntimeError("offline reference worker failed; inspect its private log")
        finally:
            stop_owned_process(owned_worker)
    offline_path = raw_dir / "offline.json"
    offline = json.loads(offline_path.read_text(encoding="utf-8"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server_cmd = [sys.executable, "-m", "newsforge.v2.serve", *shared, "--host", "127.0.0.1", "--port", str(port)]
    owned_server = None
    started = time.perf_counter()
    with (raw_dir / "server.log").open("w", encoding="utf-8") as log:
        try:
            server = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT, **child_options())
            owned_server = OwnedProcessTree(server)
            health = wait_ready(owned_server, base_url)
            actual_server = owned_server.verify_server(health)
            with ResourceSampler(actual_server) as resources:
                if health["model_version"] != offline["health"]["model_version"]:
                    raise RuntimeError("offline and HTTP model bindings differ")
                checks = api_contract_checks(base_url, health, queries[0])
                configs, raw_records = [], {}
                for batch_size, concurrency in ((1, 1), (8, 1), (1, 4), (8, 4)):
                    print(f"HTTP benchmark: batch={batch_size}, concurrency={concurrency}", flush=True)
                    config, raw = measure_configuration(base_url, queries, offline["responses"],
                                                        batch_size=batch_size, concurrency=concurrency,
                                                        rounds=args.rounds, warmup=args.warmup)
                    configs.append(config)
                    raw_records[f"batch{batch_size}_concurrency{concurrency}"] = raw
                write_json(raw_dir / "http_records.json", raw_records)
            report = {"benchmark_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
                      "sampling": {"split": "internal_validation", "seed": 2026, "queries": len(rows),
                                   "minimum_timestamp": MIN_VALID_TIME,
                                   "row_sha256": hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest()},
                      "protocol": {"transport": "localhost HTTP via urllib; fresh connection per request",
                                   "request_order": "ascending sampled row IDs, same order in each round",
                                   "cpu_threads": 1, "server_workers": 1,
                                   "execution": "offline worker exits before HTTP server starts",
                                   "latency_includes": "client JSON serialization, connection, server work, response JSON decoding",
                                   "timing_limit": "Shared desktop CPU; fixed small workload, not production capacity or a confidence interval."},
                      "source_sha256": sha256(__file__), "selection_sha256": sha256(args.selection),
                      "processed_manifest_sha256": sha256(Path(args.processed_dir) / "manifest.json"),
                      "private_evidence": {"requests_sha256": sha256(request_path),
                                           "offline_sha256": sha256(offline_path),
                                           "http_records_sha256": sha256(raw_dir / "http_records.json")},
                      "health": {key: value for key, value in health.items() if key != "server_pid"},
                      "api_contract_checks": checks, "configurations": configs,
                      "resources": resources.result(), "elapsed_seconds": time.perf_counter() - started}
            report["passed"] = (all(check["passed"] for check in checks)
                                and all(config["http_errors"] == config["offline_http_mismatches"] == 0
                                        for config in configs))
            write_json(report_path, report)
        finally:
            stop_owned_process(owned_server)
    if not report["passed"]:
        raise RuntimeError("benchmark completed with API/parity failures; see aggregate report")
    print(f"Wrote aggregate benchmark evidence to {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--temporal-dir", default="data/temporal_v2")
    parser.add_argument("--selection", default="artifacts/v2/selection.json")
    parser.add_argument("--raw-dir", default="artifacts/v2/benchmark")
    parser.add_argument("--report", default="reports/v2/service_benchmark.json")
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--offline-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.queries < 8 or args.queries % 8 or args.rounds < 1 or args.warmup < 0:
        parser.error("queries must be a positive multiple of 8; rounds >= 1 and warmup >= 0")
    if args.offline_worker:
        offline_worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
