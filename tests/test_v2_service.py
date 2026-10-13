"""HTTP/engine parity, strict coverage and selected-recipe contracts."""

import json
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np

from newsforge.v2.retrieval import CandidateConfig, rerank_order
from newsforge.v2.serve import (V2Engine, V2RecommendRequest, V2ServiceConfig, create_app,
                               verify_generator_selection, verify_ranker_selection)
from tests.test_v2_retrieval import fixture
from tests.support import workspace_temporary_directory


class TiedRanker:
    model_version = "synthetic-ranker"

    def score(self, history_ids, candidate_ids, as_of):
        return np.asarray(candidate_ids) % 3


def engine_fixture(use_reranker=True, training_cutoff=None):
    generator = fixture()
    generator.temporal.manifest = {"min_as_of": 720000, "max_as_of": 727199}
    generator.temporal.version = "synthetic-snapshot"
    generator.temporal.policy_hash = "synthetic-policy"
    if training_cutoff is not None:
        generator.encoder.metadata["train_cutoff_timestamp"] = training_cutoff
    config = V2ServiceConfig(CandidateConfig(), use_reranker=use_reranker)
    return V2Engine(generator, ["PAD"] + [f"N{item}" for item in range(1, 260)], config,
                    ranker=TiedRanker() if use_reranker else None)


class V2ServiceTest(unittest.TestCase):
    def test_engine_and_offline_share_candidate_pool_and_tie_order(self):
        engine = engine_fixture()
        self.addCleanup(engine.close)
        history, as_of = [1, 2, 3], 720100
        candidates = engine.generator.retrieve(history, as_of)
        scores = engine.ranker.score(history, candidates.ids, as_of)
        expected = candidates.ids[rerank_order(candidates.ids, scores)[:10]]
        response = engine.recommend(V2RecommendRequest(history=["N1", "N2", "N3"], as_of=as_of))
        self.assertEqual([item["item_id"] for item in response["items"]], [f"N{item}" for item in expected])
        self.assertEqual(response["candidate_count"], 100)
        self.assertEqual(response["model_version"], engine.health()["model_version"])

    def test_no_rerank_preserves_fused_order_and_reports_null_scores(self):
        engine = engine_fixture(use_reranker=False)
        self.addCleanup(engine.close)
        candidates = engine.generator.retrieve([1], 720100)
        response = engine.recommend(V2RecommendRequest(history=["N1"], as_of=720100))
        self.assertEqual([item["item_id"] for item in response["items"]],
                         [f"N{item}" for item in candidates.ids[:10]])
        self.assertTrue(all(item["rerank_score"] is None for item in response["items"]))

    def test_model_maturity_is_strict_at_exact_hour(self):
        for training_cutoff in (716400, 716401):
            engine = engine_fixture(training_cutoff=training_cutoff)
            with TestClient(create_app(engine)) as client:
                self.assertEqual(client.get("/health").json()["minimum_as_of"], 723600)
                self.assertEqual(client.post("/recommend", json={"as_of": 723599}).status_code, 422)
                self.assertEqual(client.post("/recommend", json={"as_of": 723600}).status_code, 200)

    def test_long_history_excludes_every_id_but_ranker_uses_last_twenty(self):
        engine = engine_fixture()
        self.addCleanup(engine.close)
        history = list(range(1, 41))
        with patch.object(engine.ranker, "score", wraps=engine.ranker.score) as score:
            response = engine.recommend(V2RecommendRequest(history=[f"N{i}" for i in history], as_of=720100))
        self.assertEqual(score.call_args.args[0], history[-20:])
        self.assertFalse(set(f"N{i}" for i in history) & {item["item_id"] for item in response["items"]})
        self.assertEqual(response["history_length_used"], 20)

    def test_http_filters_invalid_times_ids_and_exposes_binding(self):
        engine = engine_fixture()
        with TestClient(create_app(engine)) as client:
            health = client.get("/health").json()
            self.assertEqual(health["threads"], 1)
            self.assertEqual(health["bindings"]["temporal_policy_sha256"], "synthetic-policy")
            self.assertEqual(health["maximum_as_of"], 727199)
            for payload in ({"history": ["PAD"]}, {"history": ["OOV"]}, {"as_of": 727200},
                            {"as_of": 719999}, {"history": ["N259"], "as_of": 720100}):
                with self.subTest(payload=payload):
                    self.assertEqual(client.post("/recommend", json=payload).status_code, 422)
            response = client.post("/recommend", json={"history": ["N1"], "as_of": 720100})
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(result["snapshot_cutoff"], 716400)
            self.assertFalse({"N1", "N258", "N259", "PAD"} & {item["item_id"] for item in result["items"]})
        self.assertTrue(engine._closed)

    def test_default_latest_snapshot_and_batch_match_single(self):
        engine = engine_fixture()
        with TestClient(create_app(engine)) as client:
            query = {"history": ["N1", "N2"], "as_of": 720100}
            single = client.post("/recommend", json=query).json()
            batch = client.post("/recommend_batch", json={"requests": [query, query]}).json()
            self.assertEqual(batch["results"], [single, single])
            latest = client.post("/recommend", json={}).json()
            self.assertEqual(latest["as_of"], 727199)

    def test_zero_query_service_returns_popularity_without_422(self):
        engine = engine_fixture(use_reranker=False)
        engine.generator.encoder.arrays["w2"][:] = 0
        with TestClient(create_app(engine)) as client:
            result = client.post("/recommend", json={"history": ["N1"], "as_of": 720100})
            self.assertEqual(result.status_code, 200)
            self.assertIn("tower", result.json()["zero_query_channels"])
            self.assertEqual(len(result.json()["items"]), 10)

    def test_selection_requires_explicit_recipe_and_selected_ranker(self):
        with workspace_temporary_directory(prefix="v2-selection-") as directory:
            path = Path(directory) / "selection.json"
            value = {"serving": {"candidate_config": {"recent_hours": 24}, "use_reranker": False}}
            path.write_text(json.dumps(value))
            config = V2ServiceConfig.from_selection(path)
            self.assertEqual(config.candidate_config.recent_hours, 24)
            self.assertEqual(len(config.selection_sha256), 64)
            value["serving"]["use_reranker"] = True
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                V2ServiceConfig.from_selection(path)

    def test_runtime_rejects_self_consistent_artifact_replacements(self):
        generator_inputs = {key: f"original-{key}" for key in (
            "features", "first_seen", "train_clicks", "train_exposures", "news_ids", "embeddings",
            "retriever", "encoder", "encoder_manifest", "data_manifest", "temporal_manifest")}
        frozen = {
            "processed_files": {filename: generator_inputs[key] for filename, key in (
                ("news_features.npy", "features"), ("news_first_seen.npy", "first_seen"),
                ("news_train_clicks.npy", "train_clicks"), ("news_train_exposures.npy", "train_exposures"),
                ("news_ids.json", "news_ids"))},
            "retrieval_artifacts": {key: generator_inputs[actual] for key, actual in (
                ("article_embeddings", "embeddings"), ("retriever", "retriever"),
                ("encoder", "encoder"), ("encoder_manifest", "encoder_manifest"))},
            "data_manifest": generator_inputs["data_manifest"],
            "temporal_manifest": generator_inputs["temporal_manifest"],
            "source": {name: "source-hash" for name in ("retrieval.py", "temporal.py", "ranker.py")},
            "ranker_artifacts": {"full-seed17": {"model_path": "artifacts/selected.joblib",
                "model_sha256": "selected-ranker", "manifest_sha256": "selected-sidecar"}},
        }
        identity = {"model_path": "artifacts/selected.joblib", "model_sha256": "selected-ranker",
                    "variant": "full", "seed": 17}
        config = V2ServiceConfig(CandidateConfig(), ranker_path="artifacts/selected.joblib",
                                 ranker_identity=identity, frozen_bindings=frozen)
        generator = SimpleNamespace(bindings=generator_inputs)
        with patch("newsforge.v2.serve.sha256_file", return_value="source-hash"):
            verify_generator_selection(config, generator)
            for key in generator_inputs:
                with self.subTest(key=key):
                    changed = {**generator_inputs, key: "replacement"}
                    with self.assertRaisesRegex(ValueError, "candidate inputs"):
                        verify_generator_selection(config, SimpleNamespace(bindings=changed))
            changed = copy.deepcopy(frozen)
            changed["source"]["retrieval.py"] = "changed-code"
            with self.assertRaisesRegex(ValueError, "runtime source"):
                verify_generator_selection(replace(config, frozen_bindings=changed), generator)
        with self.assertRaisesRegex(ValueError, "required"):
            verify_generator_selection(replace(config, frozen_bindings=None), generator)
        ranker = SimpleNamespace(model_version="selected-ranker", variant="full", metadata={"seed": 17})
        verify_ranker_selection(config, ranker, "selected-sidecar")
        for version, sidecar, seed in (("replacement", "selected-sidecar", 17),
                                       ("selected-ranker", "replacement", 17), ("selected-ranker", "selected-sidecar", 29)):
            with self.subTest(version=version, sidecar=sidecar, seed=seed):
                replaced = SimpleNamespace(model_version=version, variant="full", metadata={"seed": seed})
                with self.assertRaisesRegex(ValueError, "ranker differs"):
                    verify_ranker_selection(config, replaced, sidecar)


if __name__ == "__main__":
    unittest.main()
