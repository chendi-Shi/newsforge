"""Real-data ANN and localhost HTTP benchmarks; no synthetic timing results.

Samples only internal-validation requests (all_split == 1), using seed 2026.
ANN recall compares eligible HNSW candidates with eligible exact inner-product
neighbors; it is not recommendation relevance recall. HTTP timing includes
JSON, actual loopback transport, user encoding, retrieval and reranking.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import faiss
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from newsforge.retrieval import CatalogIndex, sha256_file
from newsforge.serve import load_engine


def timing_summary(values_ms):
    values = np.asarray(values_ms, dtype=np.float64)
    if not len(values):
        return {"count": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None, "max_ms": None}
    return {"count": len(values), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(values.mean()),
            "max_ms": float(values.max())}


def sample_queries(processed_dir, count=128, seed=2026, *, top_k=10, candidate_pool_size=100):
    processed_dir = Path(processed_dir)
    histories = np.load(processed_dir / "all_history.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(processed_dir / "all_timestamps.npy", mmap_mode="r", allow_pickle=False)
    splits = np.load(processed_dir / "all_split.npy", mmap_mode="r", allow_pickle=False)
    ids = json.loads((processed_dir / "news_ids.json").read_text(encoding="utf-8"))
    # A stable random subset without replacement; no held-out label reads.
    eligible = np.flatnonzero(splits == 1)
    if len(eligible) == 0:
        raise ValueError("internal-validation split contains no requests")
    rows = np.random.default_rng(seed).choice(eligible, size=min(count, len(eligible)), replace=False)
    requests = [{"history": [ids[int(item)] for item in histories[row] if int(item) > 0],
                 "as_of": int(timestamps[row]), "top_k": top_k,
                 "candidate_pool_size": candidate_pool_size} for row in rows]
    return requests, {"split": "internal_validation", "split_value": 1, "seed": seed,
                       "available_requests": len(eligible), "sampled_requests": len(requests),
                       "sampling": "uniform without replacement, no labels read",
                       "empty_history_requests": sum(not request["history"] for request in requests),
                       "sample_array_rows": rows.astype(int).tolist()}


class RSSSampler:
    """Observed RSS from one process, sampled every 20 ms; not a memory profiler."""
    def __init__(self, pid):
        self.process = psutil.Process(pid) if pid is not None else None
        self.values = []
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        if self.process is not None:
            def sample():
                while not self.stop.is_set():
                    try:
                        if not self.process.is_running():
                            break
                        self.values.append(self.process.memory_info().rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                    self.stop.wait(.02)
            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def summary(self):
        if not self.values:
            return {"available": False, "reason": "server PID not supplied or process inaccessible"}
        return {"available": True, "pid": self.process.pid, "samples": len(self.values),
                "sampling_interval_ms": 20, "first_rss_mb": self.values[0] / 2**20,
                "max_observed_rss_mb": max(self.values) / 2**20,
                "last_rss_mb": self.values[-1] / 2**20}


def ann_benchmark(processed_dir, artifact_dir, index_dir, requests, *, threads=1,
                  k=10, warmup=8, max_overfetch=8192):
    process = psutil.Process()
    with RSSSampler(process.pid) as memory:
        engine = load_engine(processed_dir, artifact_dir, index_dir, torch_threads=threads,
                             faiss_threads=threads, max_overfetch=max_overfetch)
        approx = engine.catalogue
        embeddings = np.load(Path(artifact_dir) / "article_embeddings.npy", mmap_mode="r", allow_pickle=False)
        expected_embeddings = approx.manifest.get("source_hashes", {}).get("article_embeddings")
        if expected_embeddings and sha256_file(Path(artifact_dir) / "article_embeddings.npy") != expected_embeddings:
            raise ValueError("ANN index/article_embeddings source hash mismatch")
        exact = CatalogIndex.build(embeddings, approx.news_ids, approx.first_seen, approx.train_clicks,
                                   default_as_of=approx.default_as_of, kind="exact", threads=threads,
                                   model_version=approx.manifest["model_version"])
        encoded_queries = []
        # Encode outside the retrieval timer, in fixed small batches.
        nonempty = [request for request in requests if request["history"]]
        for start in range(0, len(nonempty), 32):
            batch = nonempty[start:start + 32]
            history_ids = np.zeros((len(batch), engine.max_history), dtype=np.int64)
            for row, request in enumerate(batch):
                numbers = [approx.id_to_index[item] for item in request["history"]][-engine.max_history:]
                history_ids[row, :len(numbers)] = numbers
            with torch.inference_mode():
                vectors = engine.retriever.encode_user(
                    torch.from_numpy(np.ascontiguousarray(engine.news_features[history_ids], dtype=np.float32)),
                    torch.from_numpy(history_ids != 0)).numpy()
            for request, vector in zip(batch, vectors):
                if float(np.linalg.norm(vector)) > 1e-12:
                    encoded_queries.append((request, vector))
        if not encoded_queries:
            raise ValueError("no nonempty, nonzero real queries available for ANN benchmark")
        for request, vector in encoded_queries[:warmup]:
            seen = {approx.id_to_index[item] for item in request["history"]}
            exact.search(vector, k, as_of=request["as_of"], seen=seen)
            approx.search(vector, k, as_of=request["as_of"], seen=seen, max_overfetch=max_overfetch)
        records, exact_ms, approx_ms, recalls = [], [], [], []
        for position, (request, vector) in enumerate(encoded_queries):
            seen = {approx.id_to_index[item] for item in request["history"]}
            results = {}
            order = [("exact", exact), ("ann", approx)]
            if position % 2:
                order.reverse()
            for name, index in order:
                started = time.perf_counter_ns()
                result = index.search(vector, k, as_of=request["as_of"], seen=seen,
                                      max_overfetch=max_overfetch if name == "ann" else None)
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                (exact_ms if name == "exact" else approx_ms).append(elapsed_ms)
                results[name] = result
            expected, actual = set(results["exact"].indices), set(results["ann"].indices)
            recall = len(expected & actual) / len(expected) if expected else None
            if recall is not None:
                recalls.append(recall)
            records.append({"query_number": position, "eligible_exact_count": len(expected),
                            "ann_count": len(actual), "recall_at_k": recall,
                            "ann_examined": results["ann"].examined,
                            "ann_shortfall": results["ann"].shortfall})
        # Verify exact FAISS against independent NumPy dot products. Ties at the
        # cutoff are accepted by score, since FAISS and NumPy tie IDs may differ.
        independent = []
        indexed_vectors = np.asarray(embeddings[exact.indexed_ids], dtype=np.float32)
        indexed_vectors /= np.linalg.norm(indexed_vectors, axis=1, keepdims=True)
        for request, vector in encoded_queries[:min(8, len(encoded_queries))]:
            seen = {approx.id_to_index[item] for item in request["history"]}
            result = exact.search(vector, k, as_of=request["as_of"], seen=seen)
            dot = indexed_vectors @ (vector / np.linalg.norm(vector))
            mask = exact.eligible(exact.indexed_ids, as_of=request["as_of"], seen=seen)
            candidate_scores = dot[mask]
            expected_count = min(k, len(candidate_scores))
            threshold = float(np.partition(candidate_scores, -expected_count)[-expected_count]) if expected_count else None
            positions = np.searchsorted(exact.indexed_ids, np.array(result.indices, dtype=np.int64))
            max_error = float(np.max(np.abs(dot[positions] - result.scores))) if result.indices else 0.0
            failures = sum(score < threshold - 1e-5 for score in result.scores) if threshold is not None else 0
            independent.append({"max_score_abs_error": max_error, "below_top_k_threshold": failures,
                                "correct_count": len(result.indices) == expected_count,
                                "passed": max_error <= 1e-5 and failures == 0 and len(result.indices) == expected_count})
        summary = {
            "queries": len(encoded_queries), "k": k, "warmup_queries": min(warmup, len(encoded_queries)),
            "catalogue_rows": len(approx.news_ids), "indexed_items": approx.index.ntotal,
            "embedding_dimension": approx.index.d, "ann_kind": approx.manifest["kind"],
            "ann_index_version": approx.version, "model_version": engine.model_version,
            "exact_latency": timing_summary(exact_ms), "ann_latency": timing_summary(approx_ms),
            "mean_ann_recall_at_k": float(np.mean(recalls)) if recalls else None,
            "queries_with_ann_shortfall": sum(record["ann_shortfall"] > 0 for record in records),
            "exact_vs_numpy": {"all_passed": all(item["passed"] for item in independent), "checks": independent},
            "faiss_threads": threads, "ef_search": approx.manifest.get("ef_search"),
            "max_overfetch": max_overfetch,
            "timing_scope": "FAISS search, adaptive overfetch, temporal and seen filters; user encoding excluded",
            "recall_definition": "intersection(eligible ANN top-k, eligible exact top-k) / size(eligible exact top-k)",
            "limitations": ["ANN recall measures approximation agreement, not click relevance.",
                            "Equal-score ties can choose different item IDs.",
                            "Measured on real internal-validation queries, not official dev."],
            "per_query": records}
        engine.close()
    summary["memory"] = memory.summary()
    return summary


def http_json(url, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"},
                      method="GET" if payload is None else "POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def validate_response(value, query, health, id_to_index, first_seen):
    if value["model_version"] != health["model_version"] or value["index_version"] != health["index_version"]:
        raise ValueError("server version changed during measurement")
    if value["as_of"] != query["as_of"] or len(value["items"]) > query["top_k"]:
        raise ValueError("response cutoff/count contract violated")
    returned = [item["item_id"] for item in value["items"]]
    if len(set(returned)) != len(returned):
        raise ValueError("duplicate recommendation IDs")
    for item in value["items"]:
        numeric = id_to_index.get(item["item_id"], 0)
        if numeric <= 0 or item["item_id"] in query["history"] or first_seen[numeric] > query["as_of"]:
            raise ValueError("returned PAD/OOV, future, or previously seen item")
        if not np.isfinite(item["retrieval_score"]) or (item["rerank_score"] is not None and not np.isfinite(item["rerank_score"])):
            raise ValueError("non-finite score")


def http_benchmark(base_url, processed_dir, requests, *, server_pid=None, warmup=8,
                   batch_sizes=(1, 8), concurrency=(1, 4), repetitions=3):
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("HTTP benchmark must target a real localhost HTTP server")
    health = http_json(base_url.rstrip("/") + "/health")
    actual_pid = health.get("server_pid")
    if not isinstance(actual_pid, int) or isinstance(actual_pid, bool) or actual_pid < 1:
        raise ValueError("server health must report its actual server_pid")
    if server_pid is not None and actual_pid != server_pid:
        raise ValueError("provided server PID does not match /health server_pid")
    server_pid = actual_pid
    processed_dir = Path(processed_dir)
    ids = json.loads((processed_dir / "news_ids.json").read_text(encoding="utf-8"))
    id_to_index = {value: i for i, value in enumerate(ids)}
    first_seen = np.load(processed_dir / "news_first_seen.npy", allow_pickle=False)
    cases = []
    for batch_size in batch_sizes:
        if not 1 <= batch_size <= 32:
            raise ValueError("HTTP batch size must be 1..32")
        batches = [requests[start:start + batch_size] for start in range(0, len(requests), batch_size)]
        endpoint = "/recommend" if batch_size == 1 else "/recommend_batch"

        def call(batch):
            payload = batch[0] if batch_size == 1 else {"requests": batch}
            started = time.perf_counter_ns()
            result = http_json(base_url.rstrip("/") + endpoint, payload)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            values = [result] if batch_size == 1 else result["results"]
            if len(values) != len(batch):
                raise ValueError("server returned wrong batch length")
            for value, query in zip(values, batch):
                validate_response(value, query, health, id_to_index, first_seen)
            return elapsed, len(batch), sum(value["shortfall"] > 0 for value in values)

        for worker_count in concurrency:
            if worker_count < 1:
                raise ValueError("concurrency must be positive")
            for batch in batches[:warmup]:
                call(batch)
            latencies, errors = [], []
            successful_queries = shortfalls = 0
            workload = batches * repetitions
            with RSSSampler(server_pid) as memory:
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(call, batch) for batch in workload]
                    for future in futures:
                        try:
                            latency, size, lacking = future.result()
                            latencies.append(latency)
                            successful_queries += size
                            shortfalls += lacking
                        except Exception as error:
                            errors.append(f"{type(error).__name__}: {error}")
                wall_seconds = time.perf_counter() - started
            cases.append({"http_batch_size": batch_size, "client_concurrency": worker_count,
                          "measured_http_requests": len(workload), "successful_http_requests": len(latencies),
                          "successful_recommendation_queries": successful_queries,
                          "error_count": len(errors), "error_examples": errors[:3],
                          "queries_with_top_k_shortfall": shortfalls,
                          "warmup_http_requests_excluded": min(warmup, len(batches)),
                          "request_latency": timing_summary(latencies), "wall_seconds": wall_seconds,
                          "http_requests_per_second": len(latencies) / wall_seconds,
                          "recommendation_queries_per_second": successful_queries / wall_seconds,
                          "server_memory": memory.summary()})
    return {"transport": "real localhost HTTP via urllib, a new connection per call",
            "health": health, "base_url": base_url, "server_pid": server_pid,
            "sampled_queries": len(requests), "repetitions": repetitions, "cases": cases,
            "all_requests_passed": all(case["error_count"] == 0 for case in cases),
            "latency_definition": "HTTP request round-trip; batch request latency is not per-query latency",
            "throughput_definition": "successful recommendation queries / workload wall time",
            "limitations": ["Loopback desktop measurement, not remote network or production SLO.",
                            "Client queue wait before a worker starts is excluded from request latency but included in throughput wall time.",
                            "Warmup excluded; one server worker; model lock serializes concurrent inference.",
                            "RSS samples include loaded models/index/features, and can miss sub-20ms peaks."]}


class _OwnedProcessTree:
    """Track only this launch's process identities, including venv wrappers.

    psutil Process objects retain creation times and check PID reuse before
    signaling. Remember descendants while their launcher is alive so cleanup
    can still reach them if the wrapper exits first. No name-based process
    discovery or termination is used.
    """
    def __init__(self, process):
        self.process = process
        self.root = psutil.Process(process.pid)
        self.owned = {(self.root.pid, self.root.create_time()): self.root}

    def refresh(self):
        for parent in list(self.owned.values()):
            try:
                if not parent.is_running():
                    continue
                for child in parent.children(recursive=True):
                    self.owned[(child.pid, child.create_time())] = child
            except psutil.NoSuchProcess:
                continue

    def verify_server_pid(self, health):
        pid = health.get("server_pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
            raise RuntimeError("server health did not provide a valid server_pid")
        self.refresh()
        if not any(child.pid == pid and child.is_running() for child in self.owned.values()):
            raise RuntimeError("health server_pid is not owned by this launched process tree")
        return pid

    def close(self, terminate_timeout=5, kill_timeout=3):
        self.refresh()
        # Capture all descendants before terminating the wrapper. Its exit can
        # detach the actual server from later parent/child enumeration.
        targets = [child for child in reversed(list(self.owned.values())) if child.is_running()]
        for child in targets:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets, timeout=terminate_timeout)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(alive, timeout=kill_timeout)
        if remaining:
            raise RuntimeError(f"owned server processes did not exit: {[child.pid for child in remaining]}")
        # Reap the Popen handle as well, even when a child was the actual server.
        self.process.wait(timeout=kill_timeout)


@contextmanager
def launched_server(processed_dir, artifact_dir, index_dir, *, threads, max_overfetch, log_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "newsforge.serve", "--processed-dir", str(processed_dir),
               "--artifact-dir", str(artifact_dir), "--index-dir", str(index_dir),
               "--port", str(port), "--torch-threads", str(threads), "--faiss-threads", str(threads),
               "--max-overfetch", str(max_overfetch)]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        owned = _OwnedProcessTree(process)
        try:
            deadline = time.monotonic() + 120
            while True:
                owned.refresh()
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with code {process.returncode}; see {log_path}")
                try:
                    health = http_json(url + "/health", timeout=2)
                except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"server did not become healthy; see {log_path}")
                    time.sleep(.25)
                    continue
                actual_pid = owned.verify_server_pid(health)
                if health.get("status") != "ok":
                    raise RuntimeError("launched server did not report healthy status")
                log.write(json.dumps({"launcher_pid": process.pid, "server_pid": actual_pid,
                                      "server_ownership": "verified descendant or launcher"}) + "\n")
                log.flush()
                break
            yield url, actual_pid
        finally:
            owned.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--index-dir", default="artifacts/index_hnsw")
    parser.add_argument("--output", default="reports/service_benchmark.json")
    parser.add_argument("--mode", choices=["ann", "http", "all"], default="all")
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ann-k", type=int, default=100,
                        help="ANN/exact candidate-pool recall cutoff, separate from served top-k")
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-overfetch", type=int, default=8192)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--launch-server", action="store_true")
    args = parser.parse_args()
    if args.query_count < 1 or args.repetitions < 1 or args.threads < 1 or args.warmup < 0 or args.ann_k < 1:
        parser.error("counts/threads/repetitions must be positive, warmup nonnegative")
    requests, sample = sample_queries(args.processed_dir, args.query_count, args.seed,
                                      top_k=args.top_k, candidate_pool_size=args.candidate_pool_size)
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    threadpool_limits(args.threads)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "pytorch": torch.__version__,
                              "faiss": faiss.__version__, "platform": platform.platform(),
                              "processor": platform.processor(), "physical_cpu_count": psutil.cpu_count(logical=False),
                              "logical_cpu_count": psutil.cpu_count(), "configured_threads": args.threads,
                              "native_thread_pools": [{key: pool.get(key) for key in
                                                       ("internal_api", "prefix", "num_threads")}
                                                      for pool in threadpool_info()],
                              "device": "cpu", "background_activity_controlled": False,
                              "system_cpu_percent_before": psutil.cpu_percent(interval=.2),
                              "system_available_ram_mb_before": psutil.virtual_memory().available / 2**20,
                              "background_activity_note": "Shared desktop; unrelated processes are not stopped by this benchmark."},
              "query_sample": sample,
              "dataset_manifest_sha256": sha256_file(Path(args.processed_dir) / "manifest.json")}
    if args.mode in {"ann", "all"}:
        report["ann"] = ann_benchmark(args.processed_dir, args.artifact_dir, args.index_dir,
                                       requests, threads=args.threads, k=args.ann_k,
                                       warmup=args.warmup, max_overfetch=args.max_overfetch)
        print(json.dumps({"ann_recall": report["ann"]["mean_ann_recall_at_k"],
                          "exact_p95_ms": report["ann"]["exact_latency"]["p95_ms"],
                          "ann_p95_ms": report["ann"]["ann_latency"]["p95_ms"]}), flush=True)
        gc.collect()
    if args.mode in {"http", "all"}:
        settings = {"warmup": args.warmup, "batch_sizes": args.batch_sizes,
                    "concurrency": args.concurrency, "repetitions": args.repetitions}
        if args.launch_server:
            with launched_server(args.processed_dir, args.artifact_dir, args.index_dir,
                                 threads=args.threads, max_overfetch=args.max_overfetch,
                                 log_path=Path(args.output).with_suffix(".server.log")) as (url, pid):
                report["http"] = http_benchmark(url, args.processed_dir, requests, server_pid=pid, **settings)
        else:
            report["http"] = http_benchmark(args.base_url, args.processed_dir, requests,
                                            server_pid=args.server_pid, **settings)
        print(json.dumps({"http_cases": report["http"]["cases"]}), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    failed = ("ann" in report and not report["ann"]["exact_vs_numpy"]["all_passed"] or
              "http" in report and not report["http"]["all_requests_passed"])
    if failed:
        raise SystemExit("Benchmark integrity checks failed; inspect the saved report")


if __name__ == "__main__":
    main()
