"""Label-free features, strict split isolation, and real estimator round-trip."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from newsforge.gbdt import (FEATURE_NAMES, evaluate_gbdt, feature_names, memory_preflight,
                           pair_features, preserve_similarity, sample_positive_indices,
                           select_feature_ablation, train_gbdt, training_matrix)


class TinyStore:
    """The real Store contract with observable label reads for split-isolation checks."""

    def __init__(self, path):
        self.path = Path(path)
        self.features = np.array([[0, 0, 0, 0], [1, 0, 1, 0],
                                  [-1, 0, 0, 1], [0, 1, 1, 0], [0, -1, 0, 1]], dtype=np.float32)
        self.split = np.array([0] * 40 + [1] * 8 + [2] * 4)
        self.history = np.tile([[1, 3, 0], [2, 4, 0]], (26, 1))
        self.history[40] = 0
        self.candidates = np.tile([1, 2, 3, 4], len(self.split)).astype(np.int32)
        self.labels = np.tile([[1, 0, 0, 0], [0, 1, 0, 0]], (26, 1)).ravel().astype(np.uint8)
        self.offsets = np.arange(len(self.split) + 1) * 4
        self.pool = np.tile([[2, 3, 4, 2], [1, 3, 4, 1]], (26, 1))
        self.positive_rows = np.arange(40)
        self.positive_items = np.tile([1, 2], 20)
        self.first_seen = np.zeros(5)
        self.clicks = np.array([0, 20, 20, 0, 0])
        self.manifest = {"features": {"text_dimensions": 2}, "train_cutoff_timestamp": 10}
        self.label_reads = []

    def ensure_training_cache(self):
        pass

    def history_batch(self, rows):
        features = self.features[self.history[rows]]
        return features, np.any(features != 0, axis=-1)

    def candidates_for(self, row):
        self.label_reads.append(int(row))
        return self.candidates[self.offsets[row]:self.offsets[row + 1]], self.labels[self.offsets[row]:self.offsets[row + 1]]

    def padded_candidates(self, rows):
        ids = self.candidates[self.offsets[rows, None] + np.arange(4)[None, :]]
        return ids, np.ones(ids.shape, dtype=bool)


class GBDTTests(unittest.TestCase):
    def setUp(self):
        # Python's Windows tempfile mode=0700 creates owner-only ACLs which the
        # sandbox token cannot reopen. Normal mkdir inherits the workspace ACL.
        self.temp = Path(__file__).resolve().parent / f"gbdt-test-{uuid.uuid4().hex}"
        self.temp.mkdir()
        self.addCleanup(shutil.rmtree, self.temp)
        self.store = TinyStore(self.temp)

    def test_padding_mask_and_candidate_batch_invariance(self):
        history, mask = self.store.history_batch(np.array([0, 1, 40]))
        candidates = np.broadcast_to(self.store.features[1:], (3, 4, 4)).copy()
        values = pair_features(history, candidates, mask, text_dimensions=2)
        padded = np.pad(history, ((0, 0), (0, 5), (0, 0)))
        padded_mask = np.pad(mask, ((0, 0), (0, 5)))
        np.testing.assert_allclose(values, pair_features(padded, candidates, padded_mask, text_dimensions=2), atol=1e-6)
        history[~mask] = 123
        np.testing.assert_allclose(values, pair_features(history, candidates, mask, text_dimensions=2), atol=1e-6)
        separate = np.concatenate([pair_features(history, candidates[:, c:c + 1], mask, text_dimensions=2)
                                   for c in range(4)], axis=1)
        np.testing.assert_allclose(values, separate, atol=1e-6)
        self.assertEqual(values.shape, (3, 4, len(FEATURE_NAMES)))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())

    def test_known_cosines_negative_max_and_category_features(self):
        history = np.array([[[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]], dtype=np.float32)
        candidates = np.array([[[1, 0, 1, 0], [-1, -1, 0, 1]]], dtype=np.float32)
        result = pair_features(history, candidates, text_dimensions=2)[0]
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self.assertAlmostEqual(result[0, index["text_cosine_mean"]], .5)
        self.assertAlmostEqual(result[0, index["text_cosine_max"]], 1)
        self.assertAlmostEqual(result[0, index["text_cosine_top3_mean"]], .5)
        self.assertAlmostEqual(result[0, index["category_overlap_mean"]], 1)
        self.assertAlmostEqual(result[1, index["category_overlap_max"]], 0)
        self.assertAlmostEqual(result[1, index["text_cosine_max"]], -1 / np.sqrt(2), places=6)
        self.assertAlmostEqual(result[1, index["text_cosine_top3_mean"]], -1 / np.sqrt(2), places=6)
        self.assertAlmostEqual(result[0, index["last_valid_history_text_cosine"]], 0)

    def test_empty_history_and_invalid_shapes(self):
        candidates = self.store.features[1:3][None]
        empty = pair_features(np.zeros((1, 0, 4)), candidates, text_dimensions=2)
        padded = pair_features(np.zeros((1, 20, 4)), candidates, text_dimensions=2)
        np.testing.assert_array_equal(empty, padded)
        self.assertTrue(np.isfinite(empty).all())
        np.testing.assert_array_equal(empty[..., -1], 1)
        np.testing.assert_array_equal(empty[..., -2], 0)
        with self.assertRaises(ValueError):
            pair_features(np.zeros((1, 2, 4)), candidates, text_dimensions=4)
        with self.assertRaises(ValueError):
            pair_features(np.zeros((2, 2, 4)), candidates, text_dimensions=2)

    def test_sampling_and_training_matrix_are_reproducible_and_train_only(self):
        first = sample_positive_indices(100, 10)
        np.testing.assert_array_equal(first, sample_positive_indices(100, 10))
        self.assertEqual(len(np.unique(first)), 10)
        np.testing.assert_array_equal(sample_positive_indices(10, 100), np.arange(10))
        matrix, labels, metadata = training_matrix(self.store, max_positives=12, batch_size=5)
        repeated, repeated_labels, _ = training_matrix(self.store, max_positives=12, batch_size=7)
        np.testing.assert_allclose(matrix, repeated, atol=1e-6)
        np.testing.assert_array_equal(labels, repeated_labels)
        self.assertEqual(matrix.shape, (60, 15))
        self.assertEqual(int(labels.sum()), 12)
        self.assertTrue(metadata["positive_pairs_subsampled"])
        self.assertEqual(self.store.label_reads, [])
        # Held-out labels cannot alter training features, samples, or objective.
        self.store.labels[160:] = 1 - self.store.labels[160:]
        changed, changed_labels, _ = training_matrix(self.store, max_positives=12, batch_size=5)
        np.testing.assert_array_equal(matrix, changed)
        np.testing.assert_array_equal(labels, changed_labels)

    def test_cache_rejects_heldout_rows_and_invalid_negatives(self):
        self.store.positive_rows[0] = 40
        with self.assertRaisesRegex(ValueError, "held-out"):
            training_matrix(self.store)
        self.store.positive_rows[0] = 0
        self.store.pool[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "non-PAD"):
            training_matrix(self.store)
        self.store.pool[0] = 1
        with self.assertRaisesRegex(ValueError, "overlaps"):
            training_matrix(self.store)

    def test_rich_features_include_raw_content_without_changing_similarity(self):
        history, mask = self.store.history_batch(np.array([0, 1, 40]))
        candidates = np.broadcast_to(self.store.features[1:], (3, 4, 4)).copy()
        rich = pair_features(history, candidates, mask, text_dimensions=2, feature_mode="rich")
        base = pair_features(history, candidates, mask, text_dimensions=2)
        np.testing.assert_array_equal(rich[..., :15], base)
        np.testing.assert_array_equal(rich[..., 15:17], candidates[..., :2])
        expected_mean = history.sum(axis=1) / np.maximum(mask.sum(axis=1)[:, None], 1)
        np.testing.assert_array_equal(rich[:, 0, 17:19], expected_mean[:, :2])
        np.testing.assert_array_equal(rich[..., 19:21], candidates[..., 2:])
        np.testing.assert_array_equal(rich[:, 0, 21:23], expected_mean[:, 2:])
        padded = np.pad(history, ((0, 0), (0, 4), (0, 0)))
        np.testing.assert_allclose(rich, pair_features(padded, candidates, text_dimensions=2,
                                                     feature_mode="rich"), atol=1e-6)
        separate = np.concatenate([pair_features(history, candidates[:, c:c + 1], mask,
            text_dimensions=2, feature_mode="rich") for c in range(4)], axis=1)
        np.testing.assert_allclose(rich, separate, atol=1e-6)
        self.assertTrue(np.isfinite(rich).all())
        self.assertEqual(len(feature_names(115, 96, "rich")), 85)

    def test_float64_memmap_keeps_same_samples_without_estimator_input_copy(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        first, labels, meta = training_matrix(self.store, max_positives=12, batch_size=5)
        mapped, rich_labels, rich_meta = training_matrix(self.store, max_positives=12, batch_size=7,
            feature_mode="rich", matrix_path=self.temp / "features.npy")
        try:
            self.assertEqual(mapped.dtype, np.float64)
            self.assertTrue(mapped.flags.f_contiguous)
            np.testing.assert_array_equal(mapped[:, :15], first)
            np.testing.assert_array_equal(rich_labels, labels)
            self.assertEqual(meta["selected_positive_indices_sha256"], rich_meta["selected_positive_indices_sha256"])
            model = HistGradientBoostingClassifier(categorical_features=None)
            validated, _ = model._preprocess_X(mapped, reset=True)
            self.assertTrue(np.shares_memory(mapped, validated))
            del validated
        finally:
            mapped._mmap.close()

    def test_memory_preflight_rejects_before_allocation(self):
        with patch("newsforge.gbdt.psutil.virtual_memory") as memory:
            memory.return_value.available = 0
            with self.assertRaisesRegex(MemoryError, "preflight refused"):
                memory_preflight(2500000, 85, memmap=True)
            memory.return_value.available = 8 * 2**30
            result = memory_preflight(2500000, 85, memmap=True)
            self.assertEqual(result["matrix_bytes"], 1700000000)
            self.assertEqual(result["estimated_float64_conversion_bytes"], 0)

    def test_two_feature_selection_preserves_original_and_keeps_test_locked(self):
        artifacts, reports = self.temp / "artifacts", self.temp / "reports"
        # This 30-positive fixture tests model/selection behavior, not current
        # desktop headroom. The production guard has its own rejection test.
        with patch("newsforge.gbdt.psutil.virtual_memory") as memory, \
                contextlib.redirect_stdout(io.StringIO()):
            memory.return_value.available = 2 * 2**30
            train_gbdt(self.store, artifacts=artifacts, reports=reports, max_positives=30,
                threads=1, batch_size=8, validation_impressions=5)
            original = (reports / "gbdt_training.json").read_bytes()
            model_bytes = (artifacts / "gbdt.joblib").read_bytes()
            preserve_similarity(artifacts, reports)
            train_gbdt(self.store, artifacts=artifacts, reports=reports, max_positives=30,
                threads=1, batch_size=8, validation_impressions=5, feature_mode="rich", run_name="gbdt_rich")
            result = select_feature_ablation(artifacts, reports)
        self.assertEqual((reports / "gbdt_similarity_training.json").read_bytes(), original)
        self.assertEqual((artifacts / "gbdt_similarity.joblib").read_bytes(), model_bytes)
        scores = {mode: record["internal_validation"]["metrics"]["ndcg10"]["mean"]
                  for mode, record in result["configurations"].items()}
        expected = max(("similarity", "rich"), key=lambda mode: scores[mode])
        self.assertEqual(result["selected_feature_mode"], expected)
        self.assertEqual((artifacts / "gbdt.joblib").read_bytes(), (artifacts / f"gbdt_{expected}.joblib").read_bytes())
        self.assertTrue(all(self.store.split[row] == 1 for row in self.store.label_reads))
        self.assertFalse((artifacts / ".gbdt_rich_training_features.npy").exists())
        (reports / "gbdt_evaluation.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "remain locked"):
            select_feature_ablation(artifacts, reports)

    def test_estimator_serialization_full_candidates_and_test_lock(self):
        root = self.temp
        # Keep the tiny estimator round-trip independent of ambient desktop RAM.
        with patch("newsforge.gbdt.psutil.virtual_memory") as memory, \
                contextlib.redirect_stdout(io.StringIO()):
            memory.return_value.available = 2 * 2**30
            artifact, report = train_gbdt(self.store, artifacts=root / "artifacts", reports=root / "reports",
                                         max_positives=30, threads=1, batch_size=8, validation_impressions=5)
        self.assertEqual(artifact["model"].n_iter_, 80)
        self.assertFalse(report["official_development_test_evaluated"])
        self.assertEqual(report["internal_validation"]["impressions"], 5)
        self.assertEqual(report["internal_validation"]["candidate_exposures"], 20)
        self.assertTrue(all(self.store.split[row] == 1 for row in self.store.label_reads))
        saved = joblib.load(root / "artifacts" / "gbdt.joblib")
        rows = np.arange(40, 48)
        with threadpool_limits(limits=1):
            _, values = evaluate_gbdt(self.store, artifact, rows, batch_size=3)
            _, loaded_values = evaluate_gbdt(self.store, saved, rows, batch_size=1)
            np.testing.assert_array_equal(values, loaded_values)
            saved["feature_version"] = 1
            del saved["feature_mode"]
            _, legacy_values = evaluate_gbdt(self.store, saved, rows, batch_size=1)
            np.testing.assert_array_equal(values, legacy_values)
            with self.assertRaisesRegex(ValueError, "evaluate-test"):
                evaluate_gbdt(self.store, saved, np.arange(48, 52))
            test_report, test_values = evaluate_gbdt(self.store, saved, np.arange(48, 52), allow_test=True)
        self.assertEqual(test_values.shape, (4, 4))
        self.assertEqual(test_report["candidate_exposures"], 16)
        self.assertTrue(np.isfinite(test_values).all())
        self.assertEqual(json.loads((root / "reports" / "gbdt_training.json").read_text())["config"]["max_iter"], 80)
        self.store.manifest["changed"] = True
        with self.assertRaisesRegex(ValueError, "manifest"):
            evaluate_gbdt(self.store, saved, rows)


if __name__ == "__main__":
    unittest.main()
