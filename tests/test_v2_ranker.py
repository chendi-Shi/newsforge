"""Small fixtures only; v1 parity imports Torch exclusively inside that test."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from newsforge.v2.ranker import (
    CausalRanker, CONTENT_NAMES, CONTEXT_NAMES, FULL_FEATURE_NAMES, FeatureBuilder,
    INTERNAL_MIN_TIMESTAMP, MODEL_CONFIG, SEEDS, VARIANTS, build_training_matrix,
    build_validation_cache, content_features, evaluate_internal, json_hash,
    memory_preflight, select_models, sha256, training_samples, validation_rows,
)
from tests.support import workspace_temporary_directory


class TinyStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir()
        self.features = np.array([[0, 0, 0, 0], [1, 0, 1, 0], [-1, 0, 0, 1],
            [0, 1, 1, 0], [0, -1, 0, 1]], dtype=np.float32)
        self.history = np.tile([[1, 3, 0], [2, 4, 0]], (26, 1))
        self.history[40] = 0
        self.split = np.array([0] * 40 + [1] * 8 + [2] * 4)
        self.timestamps = np.r_[INTERNAL_MIN_TIMESTAMP - 7200 + np.arange(40) * 60,
            INTERNAL_MIN_TIMESTAMP + np.arange(12) * 60].astype(np.int64)
        self.candidates = np.tile([1, 2, 3, 4], 52).astype(np.int32)
        self.labels = np.tile([[1, 0, 0, 0], [0, 1, 0, 0]], (26, 1)).ravel().astype(np.uint8)
        self.offsets = np.arange(53, dtype=np.int64) * 4
        self.pool = np.tile([[2, 3, 4, 2], [1, 3, 4, 1]], (26, 1))
        self.positive_rows = np.arange(40)
        self.positive_items = np.tile([1, 2], 20)
        self.first_seen = np.array([np.iinfo(np.int64).max, INTERNAL_MIN_TIMESTAMP - 86400,
            INTERNAL_MIN_TIMESTAMP - 3600, INTERNAL_MIN_TIMESTAMP - 7200,
            np.iinfo(np.int64).max], dtype=np.int64)
        self.manifest = {"features": {"text_dimensions": 2}, "train_cutoff_timestamp": INTERNAL_MIN_TIMESTAMP - 1}
        self.label_reads = []
        (self.path / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.path / "training_cache.json").write_text(json.dumps({"cache_version": 3}), encoding="utf-8")
        np.save(self.path / "news_features.npy", self.features)
        np.save(self.path / "news_first_seen.npy", self.first_seen)

    def ensure_training_cache(self):
        pass

    def padded_candidates(self, rows):
        lengths = self.offsets[rows + 1] - self.offsets[rows]
        positions = self.offsets[rows, None] + np.arange(int(lengths.max()))[None]
        mask = np.arange(positions.shape[1])[None] < lengths[:, None]
        return np.where(mask, self.candidates[np.minimum(positions, len(self.candidates) - 1)], 0), mask

    def candidates_for(self, row):
        self.label_reads.append(int(row))
        start, end = self.offsets[row:row + 2]
        return self.candidates[start:end], self.labels[start:end]


class CountingArray(np.ndarray):
    def __new__(cls, values, counter):
        obj = np.asarray(values, dtype=np.float32).view(cls)
        obj.counter = counter
        return obj

    def __array_finalize__(self, obj):
        self.counter = getattr(obj, "counter", None)

    def sum(self, *args, **kwargs):
        self.counter["sums"] += 1
        return super().sum(*args, **kwargs)


class TinyTemporal:
    def __init__(self, path, store):
        self.path = Path(path)
        self.path.mkdir()
        self.manifest = {"data_manifest_sha256": sha256(store.path / "manifest.json"),
            "arrays": {"clicks.npy": {"sha256": "fixture-clicks"}, "exposures.npy": {"sha256": "fixture-exposures"}},
            "policy": {"delay_seconds": 3600, "half_life_seconds": 43200}}
        (self.path / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.version = sha256(self.path / "manifest.json")
        self.policy_hash = json_hash(self.manifest["policy"])
        self.counter = {"sums": 0}
        self.clicks = CountingArray([0, 2, 0, 1, 0], self.counter)
        self.exposures = CountingArray([0, 10, 0, 5, 0], self.counter)
        self.clicks.flags.writeable = self.exposures.flags.writeable = False

    def snapshot(self, as_of):
        start = as_of // 3600 * 3600
        return SimpleNamespace(index=start // 3600, as_of_start=start, cutoff=start - 3600,
                               clicks=self.clicks, exposures=self.exposures)


class DummyEstimator:
    def __init__(self, features):
        self.n_features_in_ = features

    def predict_proba(self, values):
        self.last_shape = values.shape
        scores = 1 / (1 + np.exp(-np.asarray(values).sum(axis=1) * .02))
        return np.column_stack((1 - scores, scores))


class V2RankerTests(unittest.TestCase):
    def setUp(self):
        self.context = workspace_temporary_directory("v2-ranker-")
        self.path = Path(self.context.__enter__())
        self.addCleanup(self.context.__exit__, None, None, None)
        self.store = TinyStore(self.path / "processed")
        self.temporal = TinyTemporal(self.path / "temporal", self.store)
        self.builder = FeatureBuilder(self.store, self.temporal)

    def test_v1_content_definition_is_equivalent(self):
        # Run by the coordinator serially: v1's import brings in PyTorch.
        from newsforge.gbdt import FEATURE_NAMES, pair_features
        self.assertEqual(CONTENT_NAMES, FEATURE_NAMES)
        rng = np.random.default_rng(12)
        history = rng.normal(size=(4, 7, 4)).astype(np.float32)
        candidates = rng.normal(size=(4, 6, 4)).astype(np.float32)
        mask = rng.integers(0, 2, size=(4, 7)).astype(bool)
        mask[0] = False
        history[1, -1] = 0
        actual = content_features(history, candidates, mask, text_dimensions=2)
        expected = pair_features(history, candidates, mask, text_dimensions=2, feature_mode="similarity")
        np.testing.assert_array_equal(actual, expected)
        empty = np.zeros((4, 0, 4), dtype=np.float32)
        np.testing.assert_array_equal(content_features(empty, candidates, text_dimensions=2),
                                      pair_features(empty, candidates, text_dimensions=2))

    def test_context_cutoff_age_prior_padding_and_hour_cache(self):
        features = self.builder.candidate_features([1, 3], [0, 1, 2, 3, 4], INTERNAL_MIN_TIMESTAMP + 59)
        self.assertEqual(features.shape, (5, 22))
        np.testing.assert_array_equal(features[0], 0)
        context = features[:, 15:]
        prior = 3 / 15
        self.assertAlmostEqual(context[1, 0], np.log1p(2), places=6)
        self.assertAlmostEqual(context[1, 1], np.log1p(10), places=6)
        self.assertAlmostEqual(context[1, 2], (2 + 20 * prior) / 30, places=6)
        self.assertAlmostEqual(context[1, 3], np.log1p(24), places=6)
        self.assertEqual(context[1, 4], 1)
        self.assertEqual(context[2, 4], 0)  # first_seen == cutoff is excluded.
        self.assertEqual(context[4, 3], 0)  # MAX_INT unavailable age does not overflow.
        self.assertEqual(context[4, 5], 1)
        self.assertAlmostEqual(context[4, 6], prior, places=6)
        self.builder.candidate_features([], [1], INTERNAL_MIN_TIMESTAMP + 3599)
        self.assertEqual(self.temporal.counter["sums"], 2)  # Once each, not per row/candidate.

    def test_candidate_batch_padding_and_variant_invariance(self):
        for variant in VARIANTS:
            builder = FeatureBuilder(self.store, self.temporal, variant)
            whole = builder.candidate_features([1, 3], [1, 2, 3], INTERNAL_MIN_TIMESTAMP)
            separate = np.concatenate([builder.candidate_features([1, 3, 0, 0], [candidate], INTERNAL_MIN_TIMESTAMP)
                                       for candidate in [1, 2, 3]])
            np.testing.assert_allclose(whole, separate, rtol=0, atol=1e-6)
            self.assertEqual(whole.shape[1], {"content_only": 15, "context_only": 7, "full": 22}[variant])
            empty = builder.candidate_features([], [1], INTERNAL_MIN_TIMESTAMP)
            self.assertTrue(np.isfinite(empty).all())
        with self.assertRaises(ValueError):
            self.builder.candidate_features([999], [1], INTERNAL_MIN_TIMESTAMP)

    def test_training_sample_sequence_shared_and_no_current_label_features(self):
        first = training_samples(self.store, 12)
        second = training_samples(self.store, 12)
        np.testing.assert_array_equal(first.rows, second.rows)
        np.testing.assert_array_equal(first.candidate_ids, second.candidate_ids)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(first.labels.shape, (60,))
        self.assertEqual(self.store.label_reads, [])
        features = self.builder.row_features(self.store, first.rows, first.candidate_ids)
        self.store.labels[:] = 1 - self.store.labels
        np.testing.assert_array_equal(features, self.builder.row_features(self.store, first.rows, first.candidate_ids))
        with self.assertRaisesRegex(ValueError, "shrink"):
            training_samples(self.store, 41)
        self.store.positive_rows[:] = 40
        with self.assertRaisesRegex(ValueError, "held-out"):
            training_samples(self.store, 12)

    def test_hour_batched_memmap_preserves_sample_order_and_dtype(self):
        samples = training_samples(self.store, 12)
        first = build_training_matrix(self.builder, samples, self.path / "features_a.npy", batch_size=3)
        second = build_training_matrix(self.builder, samples, self.path / "features_b.npy", batch_size=7)
        try:
            self.assertEqual(first.dtype, np.float64)
            self.assertTrue(first.flags.f_contiguous)
            np.testing.assert_array_equal(first, second)
            expected = self.builder.row_features(self.store, samples.rows, samples.candidate_ids).reshape(-1, 22)
            np.testing.assert_array_equal(first, expected)
            self.assertTrue(np.shares_memory(first[:, :15], first))
            self.assertTrue(first[:, 15:].flags.f_contiguous)
        finally:
            first._mmap.close()
            second._mmap.close()

    def test_internal_full_candidate_cache_and_dev_guard(self):
        rows = validation_rows(self.store, 6)
        features, offsets = build_validation_cache(self.builder, rows, self.path / "val.npy", batch_size=2)
        try:
            ranker = CausalRanker(DummyEstimator(22), self.builder)
            report, values = evaluate_internal(ranker, self.store, rows, features, offsets)
            self.assertEqual(report["impressions"], 6)
            self.assertEqual(report["candidate_exposures"], 24)
            self.assertEqual(values.shape, (6, 4))
            self.assertTrue(all(self.store.split[row] == 1 for row in self.store.label_reads))
            for position, row in enumerate(rows):
                ids, _ = self.store.candidates_for(row)
                expected = self.builder.candidate_features(self.store.history[row], ids, int(self.store.timestamps[row]))
                np.testing.assert_array_equal(features[offsets[position]:offsets[position + 1]], expected)
            with self.assertRaisesRegex(ValueError, "internal"):
                evaluate_internal(ranker, self.store, np.arange(48, 52), features, offsets)
        finally:
            features._mmap.close()

    def test_model_roundtrip_score_reuse_and_tamper_rejection(self):
        for variant in VARIANTS:
            builder = FeatureBuilder(self.store, self.temporal, variant)
            ranker = CausalRanker(DummyEstimator(len(builder.feature_names)), builder, {"seed": 17})
            path = self.path / f"{variant}.joblib"
            ranker.save(path)
            loaded = CausalRanker.load(path, self.store, self.temporal)
            full = self.builder.candidate_features([1, 3], [1, 2, 3], INTERNAL_MIN_TIMESTAMP)
            np.testing.assert_array_equal(loaded.score([1, 3], [1, 2, 3], INTERNAL_MIN_TIMESTAMP),
                                          loaded.predict_full_features(full))
            self.assertEqual(loaded.model_version, sha256(path))
            self.assertEqual(loaded.score([], [], INTERNAL_MIN_TIMESTAMP).shape, (0,))
            with self.assertRaises(FileExistsError):
                loaded.save(path)
        path = self.path / "full.joblib"
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA"):
            CausalRanker.load(path, self.store, self.temporal)

    def test_temporal_policy_and_feature_file_changes_reject_loading(self):
        ranker = CausalRanker(DummyEstimator(22), self.builder)
        path = self.path / "model.joblib"
        ranker.save(path)
        self.temporal.policy_hash = "changed-policy"
        with self.assertRaisesRegex(ValueError, "binding"):
            CausalRanker.load(path, self.store, self.temporal)
        self.temporal.policy_hash = json_hash(self.temporal.manifest["policy"])
        values = self.store.features.copy()
        values[1, 0] += 1
        np.save(self.store.path / "news_features.npy", values)
        with self.assertRaisesRegex(ValueError, "binding"):
            CausalRanker.load(path, self.store, self.temporal)

    def test_selection_preserves_all_ablation_rows_and_deterministic_ties(self):
        records = {}
        for variant in VARIANTS:
            for seed in SEEDS:
                run = f"{variant}-seed{seed}"
                records[run] = {"run_id": run, "variant": variant, "seed": seed,
                    "sampling": {"shared": True}, "bindings": {"shared": True},
                    "config": {**MODEL_CONFIG, "random_state": seed},
                    "model_path": f"artifacts/{run}.joblib", "model_sha256": run,
                    "internal_validation": {"row_indices_sha256": "rows", "candidate_exposures": 30,
                        "metrics": {"ndcg10": {"mean": .5, "eligible_impressions": 6}}}}
        selected = select_models(records)
        self.assertEqual(selected["selected"]["variant"], "content_only")
        self.assertEqual(selected["selected"]["seed"], 17)
        self.assertEqual(len(selected["models"]), 9)
        self.assertFalse(selected["official_dev_used_for_selection"])
        records["full-seed43"]["sampling"] = {"different": True}
        with self.assertRaisesRegex(ValueError, "budgets"):
            select_models(records)

    def test_low_memory_preflight_refuses_declared_budget(self):
        with patch("newsforge.v2.ranker.psutil.virtual_memory") as virtual:
            virtual.return_value.available = 0
            with self.assertRaises(MemoryError):
                memory_preflight(500000)
            virtual.return_value.available = 8 * 2**30
            record = memory_preflight(500000)
            self.assertEqual(record["float64_memmap_bytes"], 88000000)

    def test_runtime_module_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import newsforge.v2.ranker; assert 'torch' not in sys.modules"],
            # This checks dependency isolation, not cold-start latency. Shared
            # Windows desktops may page imported libraries during the full suite.
            cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
