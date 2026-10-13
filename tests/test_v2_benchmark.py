"""HTTP load accounting, deterministic sampling, and child ownership contracts."""
from copy import deepcopy
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from newsforge.v2.benchmark import (MIN_VALID_TIME, OwnedProcessTree, ResourceSampler,
                                    child_options, http_json, measure_configuration,
                                    response_equal, select_requests, stop_owned_process,
                                    wait_ready)
from tests.support import workspace_temporary_directory


def reference(index):
    return {"items": [{"item_id": f"N{index}", "sources": ["popular"],
                       "retrieval_score": .02, "rerank_score": .4}],
            "model_version": "model", "candidate_config_version": "config",
            "snapshot_version": "snapshot", "snapshot_index": 4, "snapshot_cutoff": 100,
            "as_of": 7300, "candidate_count": 100, "shortfall": 0, "retrieval_shortfall": 0}


@contextmanager
def http_fixture(*, fail_index=None, mismatch_index=None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            queries = data["requests"] if self.path == "/recommend_batch" else [data]
            indices = [int(query["history"][0]) for query in queries]
            status = 500 if fail_index in indices else 200
            if status == 500:
                result = {"detail": "intentional fixture failure"}
            else:
                results = [reference(index) for index in indices]
                for index, response in zip(indices, results):
                    if index == mismatch_index:
                        response["model_version"] = "changed-model"
                result = {"results": results, "batch_size": len(results)} if self.path == "/recommend_batch" else results[0]
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class V2BenchmarkTest(unittest.TestCase):
    def test_response_comparison_detects_identity_provenance_and_score_drift(self):
        good = reference(0)
        self.assertTrue(response_equal(good, deepcopy(good)))
        for key, value in (("model_version", "other"), ("snapshot_cutoff", 99), ("candidate_count", 99)):
            changed = deepcopy(good)
            changed[key] = value
            self.assertFalse(response_equal(good, changed))
        for key, value in (("item_id", "other"), ("rerank_score", float("nan")),
                           ("rerank_score", None), ("retrieval_score", .03), ("sources", ["tower"])):
            changed = deepcopy(good)
            changed["items"][0][key] = value
            self.assertFalse(response_equal(good, changed))
        self.assertFalse(response_equal(good, {"items": []}))

    def test_real_loopback_requests_count_queries_and_batches_separately(self):
        queries = [{"history": [str(i)]} for i in range(8)]
        references = [reference(i) for i in range(8)]
        with http_fixture() as base_url:
            for batch_size, concurrency in ((1, 1), (8, 1), (1, 4), (8, 4)):
                with self.subTest(batch_size=batch_size, concurrency=concurrency):
                    result, raw = measure_configuration(base_url, queries, references,
                                                        batch_size=batch_size, concurrency=concurrency,
                                                        rounds=2, warmup=1)
                    self.assertEqual(result["measured_queries"], 16)
                    self.assertEqual(result["measured_http_requests"], 16 // batch_size)
                    self.assertEqual(len(raw["measured"]), 16 // batch_size)
                    self.assertEqual(result["http_errors"], 0)
                    self.assertEqual(result["offline_http_mismatches"], 0)
                    self.assertGreater(result["successful_queries_per_second"], 0)
                    self.assertGreater(result["latency_ms"]["p95"], 0)

    def test_http_errors_do_not_count_as_successful_queries(self):
        queries = [{"history": [str(i)]} for i in range(8)]
        with http_fixture(fail_index=7) as base_url:
            result, _ = measure_configuration(base_url, queries, [reference(i) for i in range(8)],
                                             batch_size=8, concurrency=1, rounds=2, warmup=0)
        self.assertEqual(result["http_errors"], 2)
        self.assertEqual(result["failed_queries"], 16)
        self.assertEqual(result["successful_queries_per_second"], 0)

    def test_parity_failure_is_counted_and_invalid_warmup_fails_fast(self):
        queries = [{"history": [str(i)]} for i in range(8)]
        refs = [reference(i) for i in range(8)]
        with http_fixture(mismatch_index=0) as base_url:
            result, _ = measure_configuration(base_url, queries, refs, batch_size=1,
                                             concurrency=2, rounds=2, warmup=0)
            self.assertEqual(result["offline_http_mismatches"], 2)
            self.assertEqual(result["http_errors"], 0)
            with self.assertRaisesRegex(RuntimeError, "warmup"):
                measure_configuration(base_url, queries, refs, batch_size=1,
                                      concurrency=1, rounds=1, warmup=1)

    def test_sampler_uses_only_internal_post_cutoff_rows_without_label_files(self):
        with workspace_temporary_directory(prefix="v2-benchmark-test-") as directory:
            path = Path(directory)
            split = np.array([0] * 10 + [1] * 100 + [2] * 10, dtype=np.int8)
            timestamps = np.full(120, MIN_VALID_TIME + 1, dtype=np.int64)
            timestamps[10:20] = MIN_VALID_TIME - 1
            history = np.tile(np.array([1, 2, 0], dtype=np.int32), (120, 1))
            for name, array in (("split", split), ("timestamps", timestamps), ("history", history)):
                np.save(path / f"all_{name}.npy", array)
            (path / "news_ids.json").write_text(json.dumps(["PAD", "N1", "N2"]))
            rows, queries = select_requests(path, count=64)
            again, repeated = select_requests(path, count=64)
            np.testing.assert_array_equal(rows, again)
            self.assertEqual(queries, repeated)
            self.assertTrue(np.all(split[rows] == 1))
            self.assertTrue(np.all(timestamps[rows] >= MIN_VALID_TIME))
            self.assertEqual(queries[0]["history"], ["N1", "N2"])
            self.assertEqual(len(set(rows)), 64)
            with self.assertRaisesRegex(ValueError, "not enough"):
                select_requests(path, count=100)

    def test_server_pid_mismatch_rejected_without_stopping_foreign_process(self):
        process = MagicMock(pid=123)
        process.poll.return_value = None
        root = fake_process(123)
        with patch("newsforge.v2.benchmark.psutil.Process", return_value=root), \
                patch("newsforge.v2.benchmark.http_json", return_value=(200, {"status": "ok", "server_pid": 456})):
            owned = OwnedProcessTree(process)
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                wait_ready(owned, "http://127.0.0.1:1", timeout=.1)
        process.terminate.assert_not_called()
        root.terminate.assert_not_called()

    def test_cleanup_requires_tracked_tree_and_accepts_no_launch(self):
        with self.assertRaisesRegex(TypeError, "tracked tree"):
            stop_owned_process(MagicMock())
        stop_owned_process(None)


def fake_process(pid):
    value = MagicMock(pid=pid)
    value.create_time.return_value = float(pid)
    value.is_running.return_value = True
    value.children.return_value = []
    return value


def fake_wait(processes, timeout):
    return ([process for process in processes if not process.is_running()],
            [process for process in processes if process.is_running()])


class OwnedProcessTreeTest(unittest.TestCase):
    def test_wrapper_child_health_is_accepted_and_sampler_uses_child_identity(self):
        root, child = fake_process(100), fake_process(101)
        root.children.return_value = [child]
        launch = MagicMock(pid=100)
        launch.poll.return_value = None
        with patch("newsforge.v2.benchmark.psutil.Process", return_value=root) as constructor, \
                patch("newsforge.v2.benchmark.http_json", return_value=(200, {"status": "ok", "server_pid": 101})):
            owned = OwnedProcessTree(launch)
            health = wait_ready(owned, "http://127.0.0.1:1", timeout=1)
            actual = owned.verify_server(health)
            self.assertIs(actual, child)
            sampler = ResourceSampler(actual)
            self.assertIs(sampler.process, child)
            self.assertEqual(sampler.result()["server_pid"], 101)
            self.assertEqual(sampler.result()["server_create_time"], 101.)
            constructor.assert_called_once_with(100)

    def test_descendant_cleanup_survives_wrapper_exit_and_kills_only_owned_tree(self):
        root, child, unrelated = fake_process(100), fake_process(101), fake_process(999)
        root.children.return_value = [child]
        launch = MagicMock(pid=100)
        child.kill.side_effect = lambda: setattr(child.is_running, "return_value", False)
        with patch("newsforge.v2.benchmark.psutil.Process", return_value=root), \
                patch("newsforge.v2.benchmark.psutil.wait_procs", side_effect=fake_wait):
            owned = OwnedProcessTree(launch)
            self.assertIs(owned.verify_server({"server_pid": 101}), child)
            root.is_running.return_value = False
            root.children.return_value = []
            self.assertIs(owned.verify_server({"server_pid": 101}), child)
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                owned.verify_server({"server_pid": 999})
            owned.close(terminate_timeout=.01, kill_timeout=.01)
        root.terminate.assert_not_called()
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        unrelated.terminate.assert_not_called()
        unrelated.kill.assert_not_called()
        launch.wait.assert_called_once_with(timeout=.01)

    def test_reused_pid_creation_time_is_rejected_and_never_signalled(self):
        root, child = fake_process(100), fake_process(101)
        root.children.return_value = [child]
        root.terminate.side_effect = lambda: setattr(root.is_running, "return_value", False)
        with patch("newsforge.v2.benchmark.psutil.Process", return_value=root), \
                patch("newsforge.v2.benchmark.psutil.wait_procs", side_effect=fake_wait):
            owned = OwnedProcessTree(MagicMock(pid=100))
            owned.refresh()
            # Simulate a recycled PID, even if a fake is_running says True.
            root.children.return_value = []
            child.create_time.return_value = 9999.
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                owned.verify_server({"server_pid": 101})
            owned.close(terminate_timeout=.01, kill_timeout=.01)
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        root.terminate.assert_called_once()

    def test_offline_wait_does_not_finish_with_a_live_worker_after_wrapper_exit(self):
        root, child = fake_process(100), fake_process(101)
        launch = MagicMock(pid=100)
        launch.poll.return_value = 0
        with patch("newsforge.v2.benchmark.psutil.Process", return_value=root):
            owned = OwnedProcessTree(launch)
        with patch.object(owned, "refresh"), patch.object(owned, "alive_processes", side_effect=[[child], []]), \
                patch("newsforge.v2.benchmark.time.sleep") as pause:
            self.assertEqual(owned.wait(timeout=1), 0)
            pause.assert_called_once_with(.1)

    def test_sampler_stops_before_reading_reused_pid_memory(self):
        child = fake_process(101)
        sampler = ResourceSampler(child)
        child.create_time.return_value = 9999.
        sampler._run()
        child.memory_info.assert_not_called()
        self.assertEqual(sampler.samples, 0)
        self.assertIn("identity", sampler.result()["sampling_error"])

    def test_real_stdlib_http_worker_is_owned_sampled_and_cleaned_after_exception(self):
        # Lifecycle smoke only: no model imports, data, or formal timing run.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        source = "\n".join([
            "import json, os",
            "from http.server import BaseHTTPRequestHandler, HTTPServer",
            "class Handler(BaseHTTPRequestHandler):",
            " def do_GET(self):",
            "  body=json.dumps({'status':'ok','server_pid':os.getpid()}).encode()",
            "  self.send_response(200)",
            "  self.send_header('Content-Length', str(len(body)))",
            "  self.end_headers()",
            "  self.wfile.write(body)",
            " def log_message(self, *args): pass",
            f"HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()",
        ])
        launch = subprocess.Popen([sys.executable, "-S", "-c", source], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, **child_options())
        owned = OwnedProcessTree(launch)
        observed = []
        with self.assertRaisesRegex(RuntimeError, "intentional lifecycle failure"):
            try:
                health = wait_ready(owned, f"http://127.0.0.1:{port}", timeout=30)
                actual = owned.verify_server(health)
                observed = list(owned.owned.values())
                with ResourceSampler(actual) as sampler:
                    status, response = http_json(f"http://127.0.0.1:{port}", "/health")
                    self.assertEqual(status, 200)
                    self.assertEqual(response["server_pid"], actual.pid)
                    sampler.stop_event.wait(.15)
                self.assertEqual(sampler.result()["server_pid"], actual.pid)
                self.assertGreater(sampler.result()["server_peak_rss_bytes"], 0)
                raise RuntimeError("intentional lifecycle failure")
            finally:
                stop_owned_process(owned)
        self.assertTrue(observed)
        self.assertTrue(all(not process.is_running() for process in observed))


if __name__ == "__main__":
    unittest.main()
