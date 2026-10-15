"""Synthetic service contracts, gate enforcement, and HTTP evidence checks."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np

from newsforge.adaptation import benchmark, serve
from newsforge.adaptation.common import sha256, write_new_json
from newsforge.adaptation.scoring import rank_methods
from newsforge.external.common import END, HOUR
from tests.support import workspace_temporary_directory


def stub_engine(gate=False):
    engine = object.__new__(serve.Engine)
    engine.closed = False
    engine.user_max = 2
    engine.base = Path("synthetic-unused")
    engine.version = "a" * 64
    engine.model_ids = {"local-seed17": "b" * 64}
    engine.first_seen = np.array([-1, 1, 2, 3, 4], dtype=np.int64)
    engine.local = np.zeros((5, 115), dtype=np.float32)
    engine.snapshots = np.zeros((169, 5), dtype=np.float64)
    engine.models = {"local": []}
    engine.decision = {"delta": 2, "gate_passed": gate,
                       "deployed_recipe": "adapted_local" if gate else "baseline"}
    engine._arrays, engine._limiter = [], None
    return engine


class ContractEngine(serve.Engine):
    def __init__(self):
        self.__dict__.update(stub_engine().__dict__)
        self.calls = []

    def recommend(self, request):
        request = self.validate_request(request)
        self.calls.append(request)
        return {"items": [{"item_id": 2, "score": 0.}], "user": request.user,
                "timestamp": request.timestamp, "top_k": request.top_k, "version": self.version,
                "snapshot_cutoff": request.timestamp // HOUR * HOUR - HOUR,
                "recipe": "baseline", "delta": 2, "gate_passed": False,
                "shortfall": max(0, request.top_k - 1), "candidate_count": 1,
                "history_length_used": 0, "api_version": "adressa-adaptation-v1"}


def release_fixture(directory):
    run = Path(directory)
    write_new_json(run / "prepare/receipt.json", {"status": "passed"})
    fit = {"prepare_receipt_sha256": sha256(run / "prepare/receipt.json")}
    write_new_json(run / "fit/receipt.json", fit)
    selection = {"delta": 2, "primary_recipe": "adapted_local",
                 "grid": {str(delta): {"hits10": 2 if delta == 0 else 3} for delta in (0, 2, 5, 10)},
                 "prepare_receipt_sha256": sha256(run / "prepare/receipt.json"),
                 "fit_receipt_sha256": sha256(run / "fit/receipt.json")}
    write_new_json(run / "tune/selection.json", selection)
    decision = {"gate_passed": True, "delta": 2,
                "primary_comparison": {"difference": .2, "ci95": [.05, .4]},
                "requests": 10, "baseline_hits10": 2, "adapted_local_hits10": 4,
                "deployed_recipe": "adapted_local", "selection_sha256": sha256(run / "tune/selection.json"),
                "earliest_service_timestamp": serve.EARLIEST_TIMESTAMP}
    write_new_json(run / "confirm/decision.json", decision)
    confirmation = {"decision": decision, "selection_sha256": decision["selection_sha256"]}
    write_new_json(run / "confirm/receipt.json", confirmation)
    receipts = {"fit": fit, "tune": {"selection": selection}, "confirm": confirmation,
                "evaluate": {"confirm_receipt_sha256": sha256(run / "confirm/receipt.json")}}
    return receipts


class ServiceContractTests(unittest.TestCase):
    def test_full_api_contract_probe_and_exact_reported_count(self):
        engine = ContractEngine()
        known = {"user": 1, "timestamp": serve.EARLIEST_TIMESTAMP + 12, "top_k": 10}
        with TestClient(serve.create_app(engine)) as client:
            health = client.get("/health").json()
            def transport(base_url, endpoint, payload=None, timeout=60):
                result = client.get(endpoint) if payload is None else client.post(endpoint, json=payload)
                return result.status_code, result.json()
            with patch.object(benchmark, "http_json", side_effect=transport):
                checks, records = benchmark.api_contract_checks("http://synthetic", health, known)
            self.assertEqual(len(checks), 19)
            self.assertEqual(len(records), len(checks))
            self.assertTrue(all(check["passed"] for check in checks), checks)
            self.assertEqual(health["minimum_timestamp"], serve.EARLIEST_TIMESTAMP)
            self.assertEqual(health["maximum_timestamp_exclusive"], END)
            self.assertEqual(health["maximum_top_k"], 100)
            self.assertEqual(health["maximum_batch_size"], 32)
            self.assertEqual(health["active_model_ids"], {})
        self.assertTrue(engine.closed)

    def test_mixed_invalid_batch_is_rejected_before_any_scoring(self):
        engine = ContractEngine()
        with TestClient(serve.create_app(engine)) as client:
            request = {"user": 1, "timestamp": serve.EARLIEST_TIMESTAMP}
            response = client.post("/recommend_batch", json={"requests": [request, {**request, "user": 3}]})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(engine.calls, [])
            good = client.post("/recommend_batch", json={"requests": [request] * 32})
            self.assertEqual(good.status_code, 200)
            self.assertEqual(good.json()["batch_size"], 32)

    def test_booleans_and_strings_are_not_coerced_into_integer_inputs(self):
        with TestClient(serve.create_app(ContractEngine())) as client:
            base = {"user": 1, "timestamp": serve.EARLIEST_TIMESTAMP, "top_k": 10}
            for key in base:
                for value in (True, str(base[key]), float(base[key])):
                    with self.subTest(key=key, value=value):
                        self.assertEqual(client.post("/recommend", json={**base, key: value}).status_code, 422)


class DeploymentTests(unittest.TestCase):
    def test_failed_gate_and_empty_history_skip_all_content_scoring(self):
        for gate, history in ((False, np.array([1])), (True, np.array([], dtype=np.int64))):
            with self.subTest(gate=gate, history=len(history)):
                engine = stub_engine(gate)
                connection = MagicMock()
                with patch.object(serve, "connect_readonly", return_value=connection), \
                     patch.object(serve, "replay_query", return_value=(history, np.array([4, 3, 2]))), \
                     patch.object(serve, "frozen_features", side_effect=AssertionError("content must not run")), \
                     patch.object(serve, "ensemble", side_effect=AssertionError("models must not run")):
                    result = engine.recommend({"user": 1, "timestamp": serve.EARLIEST_TIMESTAMP})
                self.assertEqual(result["items"], [{"item_id": 4, "score": 0.},
                                                 {"item_id": 3, "score": -1.}, {"item_id": 2, "score": -2.}])
                connection.close.assert_called_once()

    def test_enabled_service_exactly_matches_saved_method_equations(self):
        features = np.arange(39, dtype=np.float32).reshape(3, 13) / 10
        def model(offset):
            return SimpleNamespace(predict_raw=lambda matrix: matrix[:, 0] * .3 + offset)
        models = {name: [model(offset) for offset in (-.8, .1, .7)]
                  for name in ("english", "local", "local_no_recent")}
        engine = stub_engine(True)
        engine.models = models
        history, pool = np.array([1, 1, 1, 1]), np.array([4, 3, 2])
        orders, scores = rank_methods(features, features, len(history), models, models["english"], 2, True)
        with patch.object(serve, "connect_readonly", return_value=MagicMock()), \
             patch.object(serve, "replay_query", return_value=(history, pool)), \
             patch.object(serve, "frozen_features", return_value=features) as feature_call:
            result = engine.recommend({"user": 1, "timestamp": serve.EARLIEST_TIMESTAMP})
        self.assertEqual(result["items"], [{"item_id": int(pool[index]), "score": float(scores[7, index])}
                                         for index in orders[7]])
        feature_call.assert_called_once()
        self.assertEqual(result["recipe"], "adapted_local")

    def test_valid_release_is_accepted_and_gate_tampering_is_rejected(self):
        with workspace_temporary_directory() as directory:
            receipts = release_fixture(directory)
            self.assertTrue(serve.validate_release(directory, receipts)["gate_passed"])
            changed = copy.deepcopy(receipts["confirm"]["decision"])
            changed["primary_comparison"]["ci95"][0] = -.01
            # Deliberately make both public copies agree: validation must still
            # recompute the gate from the prescribed confirmation statistic.
            receipts["confirm"]["decision"] = changed
            with patch.object(serve, "read_json", side_effect=lambda path: changed if path.name == "decision.json"
                              else receipts["tune"]["selection"]):
                with self.assertRaisesRegex(ValueError, "gate disagrees"):
                    serve.validate_release(directory, receipts)

    def test_earlier_service_boundary_and_bad_hit_difference_are_rejected(self):
        with workspace_temporary_directory() as directory:
            receipts = release_fixture(directory)
            for key, value, error in (("earliest_service_timestamp", serve.EARLIEST_TIMESTAMP - 1, "boundary"),
                                      ("baseline_hits10", 1, "hit counts")):
                changed = {**receipts["confirm"]["decision"], key: value}
                altered = copy.deepcopy(receipts)
                altered["confirm"]["decision"] = changed
                with patch.object(serve, "read_json", side_effect=lambda path: changed if path.name == "decision.json"
                                  else altered["tune"]["selection"]):
                    with self.assertRaisesRegex(ValueError, error):
                        serve.validate_release(directory, altered)

    def test_initialization_checks_source_and_base_before_loading(self):
        with patch.object(serve, "check_run", side_effect=ValueError("source changed")) as check, \
             patch.object(serve, "load_adapted_models") as load:
            with self.assertRaisesRegex(ValueError, "source changed"):
                serve.Engine("repo", "base", "run")
            check.assert_called_once_with(Path("repo"), Path("base"), Path("run"), verify_base=True)
            load.assert_not_called()


class BenchmarkTests(unittest.TestCase):
    def test_actual_http_responses_are_compared_in_full_and_preserved_privately(self):
        requests = [{"user": 1, "timestamp": serve.EARLIEST_TIMESTAMP + index, "top_k": 10} for index in range(8)]
        expected = [{"items": [], "version": "frozen", "timestamp": value["timestamp"]} for value in requests]
        def transport(base_url, endpoint, payload, timeout=60):
            values = payload["requests"] if endpoint == "/recommend_batch" else [payload]
            responses = [expected[value["timestamp"] - serve.EARLIEST_TIMESTAMP] for value in values]
            return 200, {"results": responses, "batch_size": len(responses)} if endpoint == "/recommend_batch" else responses[0]
        with patch.object(benchmark, "http_json", side_effect=transport):
            summary, evidence = benchmark.measure_configuration("http://synthetic", requests, expected,
                batch_size=8, concurrency=4, rounds=3, warmup=1)
        self.assertEqual(summary["measured_queries"], 24)
        self.assertEqual(summary["measured_http_requests"], 3)
        self.assertEqual(summary["offline_http_mismatches"], 0)
        self.assertEqual(summary["http_errors"], 0)
        self.assertEqual(evidence["measured"][0]["response"]["results"], expected)
        self.assertFalse(benchmark.response_equal(expected[0], {**expected[0], "version": "changed"}))
        self.assertNotIn("response", summary)

    def test_http_failures_are_counted_as_all_failed_queries_in_the_batch(self):
        requests = [{"user": 1, "timestamp": serve.EARLIEST_TIMESTAMP}] * 8
        with patch.object(benchmark, "http_json", return_value=(500, {"detail": "synthetic"})):
            summary, _ = benchmark.measure_configuration("http://synthetic", requests, [{}] * 8,
                batch_size=8, concurrency=1, rounds=3, warmup=0)
        self.assertEqual(summary["http_errors"], 3)
        self.assertEqual(summary["failed_queries"], 24)
        self.assertEqual(summary["successful_queries_per_second"], 0)

    def test_linspace_sample_and_persisted_top10_check_catch_a_wrong_rank(self):
        with workspace_temporary_directory() as directory:
            run = Path(directory)
            (run / "prepare/evaluate").mkdir(parents=True)
            (run / "evaluate").mkdir()
            queries = np.array([[index, serve.EARLIEST_TIMESTAMP + index, 1, 3] for index in range(65)])
            pools = np.tile(np.array([2, 3, 4], dtype=np.int32), (65, 1))
            sizes = np.full(65, 3, dtype=np.int16)
            orders = np.tile(np.array([2, 0, 1], dtype=np.int16), (65, 8, 1))
            scores = np.tile(np.array([-.5, -1., 0.]), (65, 8, 1))
            for name, array in (("queries", queries), ("pools", pools), ("sizes", sizes)):
                np.save(run / "prepare/evaluate" / (name + ".npy"), array, allow_pickle=False)
            for name, array in (("orders", orders), ("scores", scores)):
                np.save(run / "evaluate" / (name + ".npy"), array, allow_pickle=False)
            rows, requests = benchmark.select_requests(run)
            self.assertEqual(len(rows), 64)
            self.assertEqual((rows[0], rows[-1]), (0, 64))
            responses = [{**request, "items": [{"item_id": 4, "score": 0.},
                         {"item_id": 2, "score": -.5}, {"item_id": 3, "score": -1.}]} for request in requests]
            self.assertTrue(benchmark.compare_saved_evaluation(run, rows, responses)["passed"])
            responses[0]["items"] = list(reversed(responses[0]["items"]))
            with self.assertRaisesRegex(ValueError, "saved deployed evaluation"):
                benchmark.compare_saved_evaluation(run, rows, responses)


if __name__ == "__main__":
    unittest.main()
