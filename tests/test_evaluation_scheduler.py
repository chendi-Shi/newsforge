"""Scheduling preserves original impression alignment and mask semantics."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import torch

from newsforge.evaluate import evaluate
from newsforge.evaluation_scheduler import evaluate_scheduled, scheduling_plan
from newsforge.final_evaluation import validation_cache_matches, verify_checkpoints, sha256
from newsforge.metrics import METRICS
from newsforge.models import AttentionRanker
from newsforge.store import Store
from newsforge.train import build_model, state_digest, training_source_digests, training_cache_signature, write_json
from tests.support import workspace_temporary_directory


def synthetic_store():
    """Real Store accessors, with empty/partial/full/zero-feature histories."""
    store = Store.__new__(Store)
    rng = np.random.default_rng(41)
    store.features = rng.normal(size=(48, 7)).astype(np.float32)
    store.features[[0, 47]] = 0
    lengths = np.array(([2, 9, 1, 17, 3, 7, 4] * 6)[:37], dtype=np.int64)
    store.offsets = np.r_[0, lengths.cumsum()]
    store.candidates = np.concatenate([rng.choice(np.arange(1, 47), int(size), replace=False)
                                       for size in lengths]).astype(np.int32)
    store.labels = np.zeros(len(store.candidates), dtype=np.uint8)
    for row, size in enumerate(lengths):
        start, end = store.offsets[row:row + 2]
        if row % 7 == 0:  # No positive: all metrics except AUC are also undefined.
            continue
        if row % 7 == 1:  # All positive: undefined AUC only.
            store.labels[start:end] = 1
        else:
            store.labels[start + int(size) // 2] = 1
    histories = np.array([[0] * 8, [1, 2, 0, 0, 0, 0, 0, 0],
                          [1, 2, 3, 4, 5, 6, 7, 8], [47, 47, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    store.history = histories[np.arange(len(lengths)) % 4]
    store.clicks = np.arange(48) % 3
    store.exposures = store.clicks + 5
    store.first_seen = np.where(np.arange(48) % 3 == 0, 11, 9)
    store.manifest = {"train_cutoff_timestamp": 10}
    features = store.features[store.history]
    store.mean_history = features.sum(1) / np.maximum(np.any(features != 0, axis=-1).sum(1, keepdims=True), 1)
    return store


class EvaluationSchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_stable_sort_and_known_padding_slots(self):
        store = SimpleNamespace(offsets=np.array([0, 5, 7, 12, 13, 17]))
        rows = np.array([2, 0, 4, 1, 3])
        order, metadata = scheduling_plan(store, rows, kind="ranker", batch_size=2)
        np.testing.assert_array_equal(order, [4, 3, 2, 0, 1])
        self.assertEqual(metadata["candidate_exposures"], 17)
        self.assertEqual(metadata["original_candidate_slots"], 19)
        self.assertEqual(metadata["scheduled_candidate_slots"], 19)
        self.assertTrue(metadata["metric_rows_restored_to_original_order"])

    def test_batch_32_boundary_and_nonranker_identity(self):
        sizes = np.array(([1, 20] * 17)[:33])
        store = SimpleNamespace(offsets=np.r_[0, sizes.cumsum()])
        rows = np.arange(33)
        _, metadata = scheduling_plan(store, rows, kind="ranker")
        self.assertEqual(metadata["batch_size"], 32)
        self.assertEqual(metadata["original_candidate_slots"], 641)
        self.assertEqual(metadata["scheduled_candidate_slots"], 660)
        # Sorting can increase padded slots on a tiny uneven tail: report the
        # actual ratio instead of promising a reduction for every population.
        self.assertGreater(metadata["candidate_slot_ratio"], 1)
        order, metadata = scheduling_plan(store, rows[::-1], kind="content")
        np.testing.assert_array_equal(order, np.arange(33))
        self.assertEqual(metadata["strategy"], "original_order")
        self.assertEqual(metadata["candidate_slot_ratio"], 1)

    def test_real_evaluator_restores_every_row_and_all_history_masks(self):
        store = synthetic_store()
        rows = np.random.default_rng(12).permutation(len(store.history))
        torch.manual_seed(17)
        model = AttentionRanker(feature_dim=7, hidden=12, embedding_dim=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            original, expected = evaluate(store, model, "ranker", rows)
            with workspace_temporary_directory(prefix="scheduler-test-") as temporary:
                path = Path(temporary) / "metrics.npy"
                actual, values = evaluate_scheduled(store, model, "ranker", rows, output=path)
                np.testing.assert_allclose(values, expected, atol=1e-12, rtol=0, equal_nan=True)
                np.testing.assert_array_equal(np.load(path), values)
        self.assertTrue(np.isnan(values[:, 0]).any())
        self.assertTrue(np.isnan(values[:, 1]).any())
        self.assertEqual(actual["candidate_exposures"], original["candidate_exposures"])
        for group, details in original["groups"].items():
            self.assertEqual(actual["groups"][group]["impressions"], details["impressions"])
            for metric in METRICS:
                wanted = details["metrics"][metric]
                observed = actual["groups"][group]["metrics"][metric]
                self.assertEqual(observed["eligible_impressions"], wanted["eligible_impressions"])
                np.testing.assert_allclose(observed["mean"], wanted["mean"], atol=1e-12, rtol=0, equal_nan=True)

    def test_empty_history_dimension_and_tied_scores(self):
        store = synthetic_store()
        store.history = np.zeros((len(store.history), 0), dtype=np.int32)
        model = AttentionRanker(feature_dim=7, hidden=12, embedding_dim=8)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        rows = np.arange(len(store.history))[::-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, expected = evaluate(store, model, "ranker", rows, batch_size=7)
            _, actual = evaluate_scheduled(store, model, "ranker", rows, batch_size=7)
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0, equal_nan=True)

    def test_invalid_rows_and_single_impression(self):
        store = SimpleNamespace(offsets=np.array([0, 2, 2, 4]))
        for rows in (np.array([], dtype=int), np.array([0.]), np.array([[0]]),
                     np.array([-1]), np.array([3]), np.array([1])):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    scheduling_plan(store, rows, kind="ranker")
        with self.assertRaises(ValueError):
            scheduling_plan(store, np.array([0]), kind="ranker", batch_size=0)
        order, metadata = scheduling_plan(store, np.array([2]), kind="ranker")
        np.testing.assert_array_equal(order, [0])
        self.assertEqual(metadata["candidate_slot_ratio"], 1)
        self.assertIsNone(metadata["padding_slot_ratio"])

    def test_validation_proof_rejects_changed_identity_or_failed_metrics(self):
        identity = {"run_id": "ranker-seed17", "selected_epoch": 1,
                    "selected_row_indices_sha256": "rows", "checkpoint_sha256": "checkpoint",
                    "data_manifest_sha256": "data", "training_source_sha256": {"train.py": "core"},
                    "scheduler_source_sha256": "scheduler"}
        record = {**identity, "metric_equivalence_passed": True,
                  "per_impression_metric_equivalence_atol": 1e-12, "rows": 12000, "fixed_validation_rows": 12000,
                  "environment": {"threads": 1, "torch": torch.__version__},
                  "scheduling": {"strategy": "stable_candidate_count", "batch_size": 32,
                                 "metric_rows_restored_to_original_order": True},
                  "metric_differences": {metric: {"same_nan_mask": True,
                                                  "pairs_differing_more_than_1e_12": 0,
                                                  "max_absolute_difference": 0.}
                                         for metric in METRICS}}
        self.assertTrue(validation_cache_matches(record, identity))
        for key in identity:
            changed = copy.deepcopy(record)
            changed[key] = "different"
            self.assertFalse(validation_cache_matches(changed, identity), key)
        for key, value in (("same_nan_mask", False), ("max_absolute_difference", 1e-10),
                           ("max_absolute_difference", float("nan")), ("pairs_differing_more_than_1e_12", 1)):
            changed = copy.deepcopy(record)
            changed["metric_differences"]["auc"][key] = value
            self.assertFalse(validation_cache_matches(changed, identity), key)
        for field, value in (("rows", 512), ("metric_equivalence_passed", False)):
            changed = copy.deepcopy(record)
            changed[field] = value
            self.assertFalse(validation_cache_matches(changed, identity), field)
        self.assertFalse(validation_cache_matches({}, identity))

    def test_nine_checkpoint_provenance_and_original_warmup_schema(self):
        # Synthetic metadata/weights only: no training, dev labels or formal
        # artifacts are touched. This catches schema drift from train.py.
        with workspace_temporary_directory(prefix="scheduler-provenance-") as temporary:
            root = Path(temporary)
            artifacts, reports, data = root / "artifacts", root / "reports", root / "data"
            for directory in (artifacts, reports, data):
                directory.mkdir()
            manifest = {"train_cutoff_timestamp": 10}
            write_json(data / "manifest.json", manifest)
            write_json(data / "training_cache.json", {"cache_version": 3, "sampler": "grouped_random_keys_v1",
                       "seed": 20261010, "pool_size": 16, "max_chunk_exposures": 400000,
                       "data_manifest_sha256": sha256(data / "manifest.json"), "features_sha256": "synthetic"})
            store = SimpleNamespace(path=data, features=np.zeros((6, 7), dtype=np.float32),
                                    news_ids=["PAD", "N1", "N2", "N3", "N4", "N5"], manifest=manifest,
                                    positive_rows=np.array([0, 1]), pool=np.zeros((2, 16)),
                                    offsets=np.arange(12001, dtype=np.int64))
            checkpoints = []
            warmup = {}
            for variant in ("mean_random", "mean_mixed", "ranker"):
                for seed in (17, 29, 43):
                    torch.manual_seed(seed)
                    model, config, kind = build_model(variant, 7)
                    digest = state_digest(model)
                    scores = [.7 + seed / 1000, .6]
                    history = [{"epoch": epoch, "training_examples": 2, "state_sha256": digest,
                                "negative_policy": "four_random" if epoch == 1 or variant == "mean_random"
                                else "two_hard_two_random",
                                "validation": {"impressions": 12000, "candidate_exposures": 12000,
                                               "metrics": {"ndcg10": {"mean": score}}}}
                               for epoch, score in enumerate(scores, 1)]
                    checkpoint = {"state_dict": model.state_dict(), "run_id": f"{variant}-seed{seed}",
                                  "variant": variant, "seed": seed, "kind": kind, "config": config,
                                  "training_configuration": {"epochs": 2, "batch_size": 1024, "threads": 1,
                                      "learning_rate": .001, "validation_impressions": 12000, "negative_pool_size": 16},
                                  "training_source_sha256": training_source_digests(),
                                  "training_cache_signature": training_cache_signature(store),
                                  "data_manifest_sha256": sha256(data / "manifest.json"),
                                  "train_cutoff_timestamp": 10, "trained_on_all_eligible_clicks": True,
                                  "eligible_positive_examples_per_epoch": 2, "history": history,
                                  "best_epoch": 1, "validation_ndcg10": scores[0]}
                    torch.save(checkpoint, artifacts / f"{variant}_seed{seed}.pt")
                    write_json(reports / f"{variant}_seed{seed}_training.json",
                               {key: value for key, value in checkpoint.items() if key != "state_dict"})
                    checkpoints.append(checkpoint)
                    if variant == "mean_random":
                        warmup[str(seed)] = {"identical": True, "sha256": [digest, digest]}
            write_json(reports / "matched_warmup_checks.json", warmup)
            for kind, filename in (("tower", "retriever.pt"), ("ranker", "reranker.pt")):
                selected = max((item for item in checkpoints if item["kind"] == kind),
                               key=lambda item: item["validation_ndcg10"])
                torch.save(selected, artifacts / filename)
            np.save(artifacts / "article_embeddings.npy", np.zeros((6, 64), dtype=np.float32))
            verified, observed, _, _ = verify_checkpoints(store, artifacts, reports, np.arange(12000))
            self.assertEqual(len(verified), 9)
            self.assertEqual(observed, warmup)
            warmup["17"]["sha256"] = warmup["17"]["sha256"][0]
            write_json(reports / "matched_warmup_checks.json", warmup)
            with self.assertRaisesRegex(ValueError, "matched-warmup"):
                verify_checkpoints(store, artifacts, reports, np.arange(12000))


if __name__ == "__main__":
    unittest.main()
