"""Synthetic HTTP parity, deployment gate and v3 ensemble scoring contracts."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np

from newsforge.v2.ranker import content_features
from newsforge.v2.serve import V2Engine, V2RecommendRequest, V2ServiceConfig
from newsforge.v3.residual import PairwiseResidual
from newsforge.v3.serve import V3Engine, create_app, rank_ensemble, load_release
from newsforge.v2.retrieval import sha256_file
from newsforge.v3.benchmark_service import response_equal, measure_configuration
from newsforge.v2 import benchmark as legacy_benchmark
from tests.support import workspace_temporary_directory
from tests.test_v3_fast_retrieval import fast_fixture, POPULAR


def models_fixture():
    models = []
    for multiplier in (3., -2., 7.):
        model = PairwiseResidual()
        model.mean, model.scale = np.zeros(13), np.ones(13)
        model.weights = np.arange(13, dtype=np.float64) * multiplier
        models.append(model)
    return models


def engine_fixture(*, gate=True, delta=3., availability=720000, features_zero=False):
    generator = fast_fixture()
    generator.temporal.manifest = {"min_as_of": 720000, "max_as_of": 727199}
    generator.temporal.version, generator.temporal.policy_hash = "synthetic-snapshot", "synthetic-policy"
    if features_zero:
        generator.features[1] = 0
    config = V2ServiceConfig(POPULAR, use_reranker=False)
    return V3Engine(generator, ["PAD"] + [f"N{i}" for i in range(1, 260)], config,
                    models=models_fixture(), delta=delta, deployment_gate_passed=gate,
                    text_dimensions=2, deployment_available_as_of=availability,
                    evidence_bindings={"selection_sha256": "synthetic-selection"})


def release_fixture(directory):
    root = Path(directory)
    paths = {name: root / f"{name}.json" for name in ("fit", "selection", "confirm", "v2", "tune")}
    paths["v2"].write_text("{}", encoding="utf-8")
    sources = {name: sha256_file(name) for name in
               ("newsforge/v2/ranker.py", "newsforge/v3/residual.py", "newsforge/v3/fast_retrieval.py")}
    model_paths = {variant: {} for variant in ("full13", "no_category", "no_recent")}
    files, runs = {}, {}
    for seed in (17, 29, 43):
        records = {}
        for variant in model_paths:
            path = root / f"{variant}-seed{seed}.npz"
            sidecar = Path(str(path) + ".json")
            path.write_bytes(f"synthetic model {variant} {seed}".encode())
            sidecar.write_text("{}", encoding="utf-8")
            model_paths[variant][str(seed)] = path.as_posix()
            records[variant] = {"path": path.as_posix(), "npz_sha256": sha256_file(path),
                                "sidecar_sha256": sha256_file(sidecar), "fit_metadata": {}}
            files[path.as_posix()], files[sidecar.as_posix()] = sha256_file(path), sha256_file(sidecar)
        runs[str(seed)] = {"models": records, "pairs_sha256": str(seed), "pair_features_sha256": str(seed)}
    fit = {"sources": sources, "input_files": {}, "environment": {}, "fit_rows": {},
           "runs": runs, "models": model_paths}
    paths["fit"].write_text(json.dumps(fit), encoding="utf-8")
    tune_binding = {"fit_report_sha256": sha256_file(paths["fit"]), "sources": sources}
    (root / "tune").mkdir()
    (root / "confirm").mkdir()
    tune_cache = root / "tune" / "cache.json"
    confirm_cache = root / "confirm" / "cache.json"
    tune_cache.write_text(json.dumps({"bindings": tune_binding}), encoding="utf-8")
    tune = {"phase": "tune", "selected_delta": 5, "bindings": tune_binding,
            "cache_manifest_sha256": sha256_file(tune_cache)}
    paths["tune"].write_text(json.dumps(tune), encoding="utf-8")
    selection = {"delta": 5, "primary_recipe": "full13+gate", "sources": sources, "models": model_paths,
                 "model_files": files, "fit_report_sha256": sha256_file(paths["fit"]),
                 "tune_cache_manifest_sha256": sha256_file(tune_cache),
                 "v2_selection_sha256": sha256_file(paths["v2"])}
    paths["selection"].write_text(json.dumps(selection), encoding="utf-8")
    confirm = {"phase": "confirm", "delta": 5, "selection_sha256": sha256_file(paths["selection"]),
               "deployment_gate_passed": True,
               "bindings": {"fit_report_sha256": selection["fit_report_sha256"], "sources": sources,
                            "selection_sha256": sha256_file(paths["selection"])},
               "metrics": {"full13": {"versus_baseline": {"delta": .03, "ci95": [.01, .05]}}}}
    confirm_cache.write_text(json.dumps({"bindings": confirm["bindings"]}), encoding="utf-8")
    confirm["cache_manifest_sha256"] = sha256_file(confirm_cache)
    confirm["tune_report_sha256"] = sha256_file(paths["tune"])
    paths["confirm"].write_text(json.dumps(confirm), encoding="utf-8")
    return paths, selection, confirm


class V3ServiceTest(unittest.TestCase):
    def test_ensemble_clips_each_seed_before_mean_then_gates(self):
        models = models_fixture()
        models[0].weights[:] = 0
        models[0].weights[0] = 100
        models[1].weights[:] = 0
        models[1].weights[0] = -2
        models[2].weights[:] = 0
        features = np.zeros((4, 13))
        features[:, 0] = [0., 1., -.5, 3.]
        raw = np.stack([model.predict_raw(features) for model in models])
        expected = -np.arange(4) + 5 * .5 * np.clip(raw, -1, 1).mean(axis=0)
        order, scores = rank_ensemble(models, features, history_length=5, delta=5)
        np.testing.assert_array_equal(scores, expected)
        np.testing.assert_array_equal(order, np.lexsort((np.arange(4), -expected)))
        self.assertFalse(np.array_equal(np.clip(raw.mean(axis=0), -1, 1), np.clip(raw, -1, 1).mean(axis=0)))

    def test_exact_residual_score_ties_keep_original_rank(self):
        models = models_fixture()
        for model in models:
            model.weights[:] = 0
            model.weights[0] = 1
        features = np.zeros((2, 13))
        features[:, 0] = [-1., 1.]
        order, scores = rank_ensemble(models, features, history_length=5, delta=1)
        np.testing.assert_array_equal(scores, [-.5, -.5])
        np.testing.assert_array_equal(order, [0, 1])

    def test_empty_history_and_zero_delta_never_evaluate_model(self):
        for history_length, delta in ((0, 10), (3, 0)):
            models = models_fixture()
            with patch.object(models[0], "score", side_effect=AssertionError("must skip model")):
                order, scores = rank_ensemble(models, np.zeros((4, 13)), history_length=history_length, delta=delta)
            np.testing.assert_array_equal(order, np.arange(4))
            np.testing.assert_array_equal(scores, -np.arange(4))

    def test_service_matches_offline_ensemble_full_history_filter_and_last20_features(self):
        engine = engine_fixture()
        self.addCleanup(engine.close)
        history, as_of = list(range(1, 41)), 720100
        candidates = engine.generator.retrieve(history, as_of)
        features = content_features(engine.generator.features[history[-20:]][None],
                                    engine.generator.features[candidates.ids][None], text_dimensions=2)[0, :, :13]
        order, scores = rank_ensemble(engine.models, features, history_length=20, delta=engine.delta)
        actual = engine.recommend(V2RecommendRequest(history=[f"N{i}" for i in history], as_of=as_of, top_k=50))
        self.assertEqual([item["item_id"] for item in actual["items"]],
                         [f"N{i}" for i in candidates.ids[order[:50]]])
        self.assertEqual([item["rerank_score"] for item in actual["items"]], scores[order[:50]].tolist())
        self.assertFalse(set(f"N{i}" for i in history) & {item["item_id"] for item in actual["items"]})
        self.assertEqual(actual["history_length_used"], 20)
        self.assertEqual(actual["candidate_count"], len(candidates.ids))

    def test_failed_gate_exactly_preserves_v2_items_sources_and_null_rank_scores(self):
        engine = engine_fixture(gate=False)
        self.addCleanup(engine.close)
        generator = fast_fixture()
        generator.temporal.manifest = dict(engine.temporal.manifest)
        generator.temporal.version, generator.temporal.policy_hash = "synthetic-snapshot", "synthetic-policy"
        original = V2Engine(generator, engine.news_ids, engine.config)
        self.addCleanup(original.close)
        request = V2RecommendRequest(history=["N1", "N2"], as_of=720100)
        with patch.object(engine.models[0], "score", side_effect=AssertionError("failed gate must skip model")):
            actual = engine.recommend(request)
        expected = original.recommend(request)
        for key in expected:
            if key != "model_version":
                self.assertEqual(actual[key], expected[key], key)
        self.assertFalse(actual["residual_applied"])
        self.assertFalse(engine.health()["reranker_enabled"])

    def test_zero_usable_history_and_zero_delta_preserve_baseline_order(self):
        for kwargs, history in (({"features_zero": True}, ["N1"]), ({}, []), ({"delta": 0}, ["N1"])):
            engine = engine_fixture(**kwargs)
            self.addCleanup(engine.close)
            ids = [engine.id_to_index[item] for item in history]
            candidates = engine.generator.retrieve(ids, 720100)
            actual = engine.recommend(V2RecommendRequest(history=history, as_of=720100))
            self.assertEqual([item["item_id"] for item in actual["items"]], [f"N{i}" for i in candidates.ids[:10]])
            self.assertTrue(all(item["rerank_score"] is None for item in actual["items"]))
            self.assertFalse(actual["residual_applied"])

    def test_confirmed_model_rejects_requests_before_deployment_maturity(self):
        engine = engine_fixture(availability=723600)
        with TestClient(create_app(engine)) as client:
            self.assertEqual(client.get("/health").json()["minimum_as_of"], 723600)
            self.assertEqual(client.post("/recommend", json={"as_of": 723599}).status_code, 422)
            self.assertEqual(client.post("/recommend", json={"as_of": 723600}).status_code, 200)

    def test_http_contract_batch_parity_provenance_and_lifespan(self):
        engine = engine_fixture()
        with TestClient(create_app(engine)) as client:
            health = client.get("/health").json()
            self.assertEqual(health["api_version"], "v3")
            self.assertIn("v3/serve.py", health["bindings"]["v3_runtime_sources"])
            self.assertEqual(len(health["bindings"]["residual_model_content_sha256"]), 3)
            self.assertEqual(health["bindings"]["v3_evidence"]["selection_sha256"], "synthetic-selection")
            query = {"history": ["N1", "N2"], "as_of": 720100}
            single = client.post("/recommend", json=query).json()
            batch = client.post("/recommend_batch", json={"requests": [query, query]}).json()
            self.assertEqual(batch["results"], [single, single])
            self.assertEqual(single["model_version"], health["model_version"])
            for payload in ({"history": ["PAD"]}, {"history": ["OOV"]}, {"as_of": 719999},
                            {"history": ["N259"], "as_of": 720100}, {"top_k": 0}, {"top_k": 51}):
                self.assertEqual(client.post("/recommend", json=payload).status_code, 422)
            for queries in ([], [{}] * 33):
                self.assertEqual(client.post("/recommend_batch", json={"requests": queries}).status_code, 422)
        self.assertTrue(engine._closed)
        self.assertTrue(engine.generator.cache_info()["closed"])

    def test_mutating_selected_delta_or_evidence_changes_model_version(self):
        first, second = engine_fixture(delta=3), engine_fixture(delta=5)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        self.assertNotEqual(first.model_version, second.model_version)
        self.assertFalse(first.models[0].weights.flags.writeable)

    def test_invalid_ensemble_configuration_rejected(self):
        for value in (True, -1, float("inf")):
            with self.assertRaises(ValueError):
                rank_ensemble(models_fixture(), np.zeros((2, 13)), history_length=1, delta=value)
        with self.assertRaisesRegex(ValueError, "three seeds"):
            rank_ensemble(models_fixture()[:2], np.zeros((2, 13)), history_length=1, delta=1)
        with self.assertRaisesRegex(ValueError, "history_length"):
            rank_ensemble(models_fixture(), np.zeros((2, 13)), history_length=True, delta=1)

    def test_release_loader_checks_artifact_chain_and_model_training_bindings(self):
        with workspace_temporary_directory(prefix="v3-release-") as directory:
            paths, selection, _ = release_fixture(directory)
            with patch("newsforge.v3.serve.PairwiseResidual.load", side_effect=models_fixture()) as load:
                selected, gate, models, evidence = load_release(paths["selection"], paths["confirm"], paths["fit"], paths["v2"])
            self.assertTrue(gate)
            self.assertEqual(selected, selection)
            self.assertEqual(len(models), 3)
            self.assertEqual(evidence["model_files_sha256"], selection["model_files"])
            self.assertEqual([call.kwargs["expected_bindings"]["seed"] for call in load.call_args_list], [17, 29, 43])
            replaced = Path(selection["models"]["full13"]["17"])
            replaced.write_bytes(b"replacement model")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_release(paths["selection"], paths["confirm"], paths["fit"], paths["v2"])

    def test_release_loader_rejects_confirm_mismatch_and_inconsistent_gate(self):
        with workspace_temporary_directory(prefix="v3-gate-") as directory:
            paths, _, confirm = release_fixture(directory)
            for changed in ({**confirm, "selection_sha256": "wrong"},
                            {**confirm, "deployment_gate_passed": 1},
                            {**confirm, "deployment_gate_passed": False},
                            {**confirm, "delta": 10}):
                paths["confirm"].write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_release(paths["selection"], paths["confirm"], paths["fit"], paths["v2"])

    def test_release_loader_rejects_incomplete_or_replaced_tune_evidence(self):
        with workspace_temporary_directory(prefix="v3-tune-chain-") as directory:
            paths, _, _ = release_fixture(directory)
            tune = json.loads(paths["tune"].read_text(encoding="utf-8"))
            tune["selected_delta"] = 10
            paths["tune"].write_text(json.dumps(tune), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tune/confirmation"):
                load_release(paths["selection"], paths["confirm"], paths["fit"], paths["v2"])

    def test_http_benchmark_compares_every_response_field_exactly_and_restores_legacy(self):
        reference = {"items": [{"item_id": "N1", "rerank_score": .25}], "residual_applied": True,
                     "shortfall": 0, "retrieval_shortfall": 0}
        changed_score = {**reference, "items": [{"item_id": "N1", "rerank_score": .25 + 1e-12}]}
        self.assertFalse(response_equal(reference, changed_score))
        self.assertFalse(response_equal(reference, {**reference, "residual_applied": False}))
        original = legacy_benchmark.response_equal
        with patch.object(legacy_benchmark, "http_json", return_value=(200, reference)):
            result, _ = measure_configuration("http://unused", [{"history": []}], [reference],
                                              batch_size=1, concurrency=1, rounds=1, warmup=1)
        self.assertEqual(result["offline_http_mismatches"], 0)
        self.assertIs(legacy_benchmark.response_equal, original)
        with patch.object(legacy_benchmark, "http_json", return_value=(200, changed_score)):
            result, _ = measure_configuration("http://unused", [{"history": []}], [reference],
                                              batch_size=1, concurrency=1, rounds=1, warmup=0)
        self.assertEqual(result["offline_http_mismatches"], 1)
        self.assertIs(legacy_benchmark.response_equal, original)


if __name__ == "__main__":
    unittest.main()
