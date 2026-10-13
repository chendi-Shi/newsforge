"""HTTP contract, privacy, cold-start, cutoff, and batching tests."""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import numpy as np
import psutil
import torch
from torch import nn

from newsforge.retrieval import CatalogIndex, build_from_files, sha256_file
from newsforge.serve import RecommendationEngine, RecommendRequest, create_app, load_engine
from tests.support import workspace_temporary_directory


class TestRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def encode_user(self, history, mask):
        self.batch_sizes.append(len(history))
        result = (history * mask.unsqueeze(-1)).sum(dim=1)
        return torch.nn.functional.normalize(result, dim=-1)


class TestRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_shapes = []

    def score_candidates(self, history, mask, candidates):
        self.batch_shapes.append(tuple(candidates.shape))
        return candidates[:, :, 1] + .01 * (history * mask.unsqueeze(-1)).sum(dim=(1, 2))[:, None]


def engine():
    features = np.array([[0, 0], [1, .1], [.8, .3], [.6, .5], [.4, .9], [.5, .7], [.9, .2]], dtype=np.float32)
    index = CatalogIndex.build(features, ["PAD", "N1", "N2", "N3", "N4", "N5", "FUTURE"],
                               np.array([0, 10, 10, 10, 10, 10, 30]),
                               np.array([999, 10, 20, 5, 3, 2, 999]),
                               default_as_of=20, kind="exact")
    index.manifest["index_version"] = "test-index-version"
    return RecommendationEngine(index, features, TestRetriever(), TestRanker(),
                                 model_version="test-model-version", candidate_batch_size=2,
                                 max_history=2, torch_threads=1)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine()
        self.client = TestClient(create_app(self.engine, warmup_iterations=0))

    def tearDown(self):
        self.client.close()

    def test_health_exposes_versions_not_user_data(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        value = response.json()
        self.assertEqual(value["model_version"], "test-model-version")
        self.assertEqual(value["index_version"], "test-index-version")
        self.assertIsNone(value["model_train_cutoff"])
        self.assertEqual(value["server_pid"], os.getpid())
        self.assertNotIn("history", value)
        self.assertNotIn("user_id", value)

    def test_recommend_reranks_and_filters_seen_future_and_pad(self):
        response = self.client.post("/recommend", json={"history": ["N1"], "top_k": 3,
                                                        "candidate_pool_size": 6})
        self.assertEqual(response.status_code, 200, response.text)
        value = response.json()
        self.assertEqual([item["item_id"] for item in value["items"]], ["N4", "N5", "N3"])
        self.assertEqual(value["candidate_count"], 4)
        self.assertEqual(value["shortfall"], 0)
        self.assertEqual(value["retrieval_shortfall"], 2)
        self.assertTrue(all(item["rerank_score"] is not None for item in value["items"]))
        self.assertEqual(len(self.engine.ranker.batch_shapes), 2)
        self.assertNotIn("history", value)

    def test_cold_start_uses_only_training_popularity_and_no_ranker(self):
        response = self.client.post("/recommend", json={"history": [], "top_k": 3})
        self.assertEqual(response.status_code, 200)
        value = response.json()
        self.assertEqual([item["item_id"] for item in value["items"]], ["N2", "N1", "N3"])
        self.assertEqual(value["strategy"], "train_popularity_cold_start")
        self.assertTrue(all(item["rerank_score"] is None for item in value["items"]))
        self.assertEqual(self.engine.retriever.batch_sizes, [])
        self.assertEqual(self.engine.ranker.batch_shapes, [])

    def test_empty_catalogue_at_cutoff_returns_shortfall(self):
        response = self.client.post("/recommend", json={"history": [], "as_of": 1, "top_k": 5})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [])
        self.assertEqual(response.json()["shortfall"], 5)

    def test_training_cutoff_blocks_warm_cold_and_batch_before_inference(self):
        bounded = RecommendationEngine(self.engine.catalogue, self.engine.news_features,
                                       TestRetriever(), TestRanker(), model_version="bounded",
                                       model_train_cutoff=20)
        with TestClient(create_app(bounded, warmup_iterations=0)) as client:
            self.assertEqual(client.get("/health").json()["model_train_cutoff"], 20)
            for history in ([], ["N1"]):
                response = client.post("/recommend", json={"history": history, "as_of": 19})
                self.assertEqual(response.status_code, 422)
                self.assertIn("model_train_cutoff", response.json()["detail"])
            response = client.post("/recommend_batch", json={"requests": [
                {"history": ["N1"], "as_of": 20}, {"history": [], "as_of": 19}]})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(bounded.retriever.batch_sizes, [])
            self.assertEqual(bounded.ranker.batch_shapes, [])
            for payload in ({"as_of": 20}, {"history": ["N1"], "as_of": 21}, {}):
                self.assertEqual(client.post("/recommend", json=payload).status_code, 200)
        # The generic catalogue can still answer historical availability queries.
        self.assertEqual(self.engine.catalogue.popular(3, as_of=10).indices, [2, 1, 3])

    def test_invalid_model_cutoff_or_earlier_index_default_rejected(self):
        for cutoff in (-1, True, 1.5, int(np.iinfo(np.int64).max), 21):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                RecommendationEngine(self.engine.catalogue, self.engine.news_features,
                                     TestRetriever(), TestRanker(), model_version="invalid",
                                     model_train_cutoff=cutoff)

    def test_pad_oov_unknown_and_future_history_rejected(self):
        for article in ("PAD", "OOV", "UNRECOGNIZED", "FUTURE"):
            with self.subTest(article=article):
                response = self.client.post("/recommend", json={"history": [article]})
                self.assertEqual(response.status_code, 422)

    def test_invalid_request_bounds_and_candidate_pool_rejected(self):
        for payload in ({"top_k": 0}, {"top_k": 51}, {"as_of": -1},
                        {"top_k": 10, "candidate_pool_size": 5}, {"candidate_pool_size": 1001}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/recommend", json=payload).status_code, 422)

    def test_batch_is_equivalent_to_single_and_batches_model_calls(self):
        payloads = [{"history": ["N1"], "top_k": 2, "candidate_pool_size": 5},
                    {"history": ["N2"], "top_k": 2, "candidate_pool_size": 5},
                    {"history": [], "top_k": 2, "candidate_pool_size": 5}]
        response = self.client.post("/recommend_batch", json={"requests": payloads})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.engine.retriever.batch_sizes, [2])
        self.assertTrue(all(shape[0] == 2 for shape in self.engine.ranker.batch_shapes))
        results = response.json()["results"]
        for payload, actual in zip(payloads, results):
            expected = self.client.post("/recommend", json=payload).json()
            self.assertEqual(actual, expected)

    def test_truncated_history_still_excludes_every_seen_article(self):
        self.engine.max_history = 1
        response = self.client.post("/recommend", json={"history": ["N1", "N2", "N3"],
                                                        "top_k": 5, "candidate_pool_size": 6})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["history_length_used"], 1)
        self.assertEqual({item["item_id"] for item in response.json()["items"]}, {"N4", "N5"})
        self.assertEqual(response.json()["shortfall"], 3)

    def test_warmup_exercises_index_and_models(self):
        self.engine.warmup(iterations=1)
        self.assertGreater(len(self.engine.retriever.batch_sizes), 0)
        self.assertGreater(len(self.engine.ranker.batch_shapes), 0)

    def test_known_history_without_features_uses_cold_fallback_and_stays_excluded(self):
        self.engine.news_features[1] = 0
        response = self.client.post("/recommend", json={"history": ["N1"], "top_k": 5})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["strategy"], "train_popularity_cold_start")
        self.assertNotIn("N1", [item["item_id"] for item in response.json()["items"]])

    def test_real_model_checkpoint_load_and_stale_index_rejection(self):
        from newsforge.models import TwoTower, AttentionRanker
        with workspace_temporary_directory() as folder:
            folder = Path(folder)
            features = self.engine.news_features
            np.save(folder / "news_features.npy", features)
            np.save(folder / "news_first_seen.npy", self.engine.catalogue.first_seen)
            np.save(folder / "news_train_clicks.npy", self.engine.catalogue.train_clicks)
            (folder / "news_ids.json").write_text(json.dumps(self.engine.catalogue.news_ids), encoding="utf-8")
            torch.manual_seed(31)
            config = {"feature_dim": 2, "embedding_dim": 4, "hidden": 4, "user_pooling": "mean"}
            ranker_config = {"feature_dim": 2, "embedding_dim": 4, "hidden": 4}
            retriever, ranker = TwoTower(**config), AttentionRanker(**ranker_config)
            torch.save({"config": config, "state_dict": retriever.state_dict(), "run_id": "test-run",
                        "train_cutoff_timestamp": 18},
                       folder / "retriever.pt")
            torch.save({"config": ranker_config, "state_dict": ranker.state_dict(),
                        "train_cutoff_timestamp": 20}, folder / "reranker.pt")
            with torch.inference_mode():
                np.save(folder / "article_embeddings.npy", retriever.encode_news(torch.from_numpy(features)).numpy())
            build_from_files(folder, folder / "article_embeddings.npy", folder / "index",
                              default_as_of=20, kind="exact", model_version=sha256_file(folder / "retriever.pt"))
            restored = load_engine(folder, folder, folder / "index")
            result = restored.recommend(RecommendRequest(history=["N1"], top_k=2))
            self.assertEqual(len(result.items), 2)
            self.assertEqual(len(result.model_version), 64)
            self.assertEqual(restored.health()["model_train_cutoff"], 20)
            with TestClient(create_app(restored, warmup_iterations=0)) as client:
                for history in ([], ["N1"]):
                    response = client.post("/recommend", json={"history": history, "as_of": 19})
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(client.post("/recommend", json={"history": history,
                                                                    "as_of": 20}).status_code, 200)
            with torch.no_grad():
                retriever.user_encoder[0].weight.zero_()
            torch.save({"config": config, "state_dict": retriever.state_dict(), "run_id": "test-run",
                        "train_cutoff_timestamp": 18},
                       folder / "retriever.pt")
            with self.assertRaisesRegex(ValueError, "model-version mismatch"):
                load_engine(folder, folder, folder / "index")
            restored.close()
            with self.assertRaisesRegex(RuntimeError, "engine is closed"):
                restored.recommend(RecommendRequest(history=["N1"], top_k=2))
            for unbound_version in ("unknown", "test-run"):
                build_from_files(folder, folder / "article_embeddings.npy", folder / "index",
                                  default_as_of=20, kind="exact", model_version=unbound_version)
                with self.assertRaisesRegex(ValueError, "model-version mismatch"):
                    load_engine(folder, folder, folder / "index")
            for cutoff in (None, -1, True, 1.5, int(np.iinfo(np.int64).max)):
                with self.subTest(checkpoint_cutoff=cutoff):
                    checkpoint = {"config": ranker_config, "state_dict": ranker.state_dict()}
                    if cutoff is not None:
                        checkpoint["train_cutoff_timestamp"] = cutoff
                    torch.save(checkpoint, folder / "reranker.pt")
                    with self.assertRaisesRegex(ValueError, "reranker checkpoint.*train_cutoff_timestamp"):
                        load_engine(folder, folder, folder / "index")


class ProcessLifecycleTests(unittest.TestCase):
    def test_owned_tree_rejects_unrelated_or_reused_pid_and_cleans_wrapper_children(self):
        from newsforge.benchmark_service import _OwnedProcessTree

        def fake_process(pid):
            value = Mock(pid=pid)
            value.create_time.return_value = float(pid)
            value.is_running.return_value = True
            value.children.return_value = []
            return value

        root, child, unrelated = (fake_process(pid) for pid in (100, 101, 999))
        root.children.return_value = [child]
        popen = Mock(pid=root.pid)
        root.terminate.side_effect = lambda: setattr(root.is_running, "return_value", False)
        # Simulate a child that needs kill after the bounded terminate wait.
        child.kill.side_effect = lambda: setattr(child.is_running, "return_value", False)

        def wait(procs, timeout):
            return ([proc for proc in procs if not proc.is_running()],
                    [proc for proc in procs if proc.is_running()])

        with patch("newsforge.benchmark_service.psutil.Process", return_value=root) as constructor, \
                patch("newsforge.benchmark_service.psutil.wait_procs", side_effect=wait):
            tree = _OwnedProcessTree(popen)
            self.assertEqual(tree.verify_server_pid({"server_pid": child.pid}), child.pid)
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                tree.verify_server_pid({"server_pid": unrelated.pid})
            # psutil is_running rejects a stored identity after its PID is reused.
            child.is_running.return_value = False
            root.children.return_value = []
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                tree.verify_server_pid({"server_pid": child.pid})
            child.is_running.return_value = True
            tree.close(terminate_timeout=.01, kill_timeout=.01)
            constructor.assert_called_once_with(root.pid)
        root.terminate.assert_called_once()
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        unrelated.terminate.assert_not_called()
        unrelated.kill.assert_not_called()
        popen.wait.assert_called_once_with(timeout=.01)

    def test_http_sampler_rejects_launcher_pid_when_health_reports_child(self):
        from newsforge.benchmark_service import http_benchmark
        with patch("newsforge.benchmark_service.http_json", return_value={"server_pid": 101}):
            with self.assertRaisesRegex(ValueError, "does not match"):
                http_benchmark("http://127.0.0.1:8000", "unused", [], server_pid=100)

    def test_real_localhost_service_pid_rss_and_exception_cleanup(self):
        """Lifecycle smoke only: one synthetic request, no benchmark timings."""
        from newsforge.benchmark_service import RSSSampler, http_json, launched_server
        from newsforge.models import TwoTower, AttentionRanker

        fixture = engine()
        with workspace_temporary_directory() as temporary:
            folder = Path(temporary)
            np.save(folder / "news_features.npy", fixture.news_features)
            np.save(folder / "news_first_seen.npy", fixture.catalogue.first_seen)
            np.save(folder / "news_train_clicks.npy", fixture.catalogue.train_clicks)
            (folder / "news_ids.json").write_text(json.dumps(fixture.catalogue.news_ids), encoding="utf-8")
            config = {"feature_dim": 2, "embedding_dim": 4, "hidden": 4, "user_pooling": "mean"}
            ranker_config = {"feature_dim": 2, "embedding_dim": 4, "hidden": 4}
            retriever, ranker = TwoTower(**config), AttentionRanker(**ranker_config)
            torch.save({"config": config, "state_dict": retriever.state_dict(),
                        "train_cutoff_timestamp": 20}, folder / "retriever.pt")
            torch.save({"config": ranker_config, "state_dict": ranker.state_dict(),
                        "train_cutoff_timestamp": 20}, folder / "reranker.pt")
            with torch.inference_mode():
                np.save(folder / "article_embeddings.npy",
                        retriever.encode_news(torch.from_numpy(fixture.news_features)).numpy())
            build_from_files(folder, folder / "article_embeddings.npy", folder / "index",
                             default_as_of=20, kind="exact", model_version=sha256_file(folder / "retriever.pt"))
            observed = []
            log_path = folder / "lifecycle.log"
            with self.assertRaisesRegex(RuntimeError, "deliberate lifecycle exception"):
                with launched_server(folder, folder, folder / "index", threads=1,
                                     max_overfetch=32, log_path=log_path) as (url, server_pid):
                    health = http_json(url + "/health")
                    self.assertEqual(server_pid, health["server_pid"])
                    audit = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
                             if line.startswith('{"launcher_pid":')][-1]
                    self.assertEqual(audit["server_pid"], server_pid)
                    observed = [psutil.Process(pid) for pid in {audit["launcher_pid"], server_pid}]
                    for process in observed:
                        process.create_time()  # retain identity for PID-reuse-safe assertions
                    with RSSSampler(server_pid) as sampler:
                        response = http_json(url + "/recommend", {"history": ["N1"], "top_k": 2})
                    self.assertEqual(len(response["items"]), 2)
                    memory = sampler.summary()
                    self.assertTrue(memory["available"])
                    self.assertEqual(memory["pid"], server_pid)
                    self.assertGreater(memory["max_observed_rss_mb"], 1)
                    print(f"lifecycle smoke launcher_pid={audit['launcher_pid']} "
                          f"actual_server_pid={server_pid} rss_mb={memory['max_observed_rss_mb']:.1f}", flush=True)
                    raise RuntimeError("deliberate lifecycle exception")
            self.assertTrue(observed)
            self.assertTrue(all(not process.is_running() for process in observed))


if __name__ == "__main__":
    unittest.main()
