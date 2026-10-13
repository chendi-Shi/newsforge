"""Synthetic v3 temporal sampling, logged pairs, bounded ranking and cache audits."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from newsforge.v2.ranker import content_features
from newsforge.v2.retrieval import CandidateConfig
from newsforge.v3 import experiment
from tests.support import workspace_temporary_directory


class LoggedStore:
    def __init__(self, groups):
        self.groups = groups
        self.read_rows = []

    def candidates_for(self, row):
        self.read_rows.append(row)
        ids, labels = self.groups[row]
        return np.asarray(ids, dtype=np.int64), np.asarray(labels)


class V3ExperimentTest(unittest.TestCase):
    def test_fixed_samples_enforce_split_half_open_windows_and_count(self):
        store = SimpleNamespace(
            timestamps=np.array([99, 100, 101, 109, 110, 104, 105, 108, 100, 106]),
            split=np.array([1, 1, 1, 1, 1, 0, 2, 1, 0, 1], dtype=np.uint8))
        with patch.dict(experiment.WINDOWS, {"fit": (100, 110, 4)}):
            selected = experiment.fixed_sample_rows(store, "fit")
            eligible = np.array([1, 2, 3, 7, 9])
            expected = np.sort(np.random.default_rng(2026).choice(eligible, 4, replace=False))
            np.testing.assert_array_equal(selected, expected)
            np.testing.assert_array_equal(selected, experiment.fixed_sample_rows(store, "fit"))
            self.assertEqual(len(np.unique(selected)), 4)
            self.assertTrue(np.all(store.split[selected] == 1))
            self.assertTrue(np.all((store.timestamps[selected] >= 100) & (store.timestamps[selected] < 110)))
        with patch.dict(experiment.WINDOWS, {"fit": (100, 110, 5)}):
            selected = experiment.fixed_sample_rows(store, "fit")
            self.assertIn(1, selected)  # Exact lower bound is included.
            self.assertNotIn(4, selected)  # Exact upper bound is excluded.
        with patch.dict(experiment.WINDOWS, {"fit": (100, 110, 6)}):
            with self.assertRaisesRegex(ValueError, "insufficient"):
                experiment.fixed_sample_rows(store, "fit")
        with self.assertRaisesRegex(ValueError, "unknown"):
            experiment.fixed_sample_rows(store, "dev")

    def test_all_three_internal_windows_are_disjoint_with_maturity_gaps(self):
        fit_low, fit_high, _ = experiment.WINDOWS["fit"]
        tune_low, tune_high, _ = experiment.WINDOWS["tune"]
        confirm_low, confirm_high, _ = experiment.WINDOWS["confirm"]
        self.assertEqual(tune_low - fit_high, 3600)
        self.assertEqual(confirm_low - tune_high, 3600)
        self.assertEqual(1573776000 - confirm_high, 3600)
        timestamps = np.array([fit_low, fit_high - 1, fit_high, tune_low,
                               tune_high - 1, tune_high, confirm_low, confirm_high - 1, confirm_high])
        store = SimpleNamespace(timestamps=timestamps, split=np.ones(len(timestamps), dtype=np.uint8))
        windows = {name: (low, high, 2) for name, (low, high, _) in experiment.WINDOWS.items()}
        with patch.dict(experiment.WINDOWS, windows):
            rows = {name: experiment.fixed_sample_rows(store, name) for name in windows}
        np.testing.assert_array_equal(rows["fit"], [0, 1])
        np.testing.assert_array_equal(rows["tune"], [3, 4])
        np.testing.assert_array_equal(rows["confirm"], [6, 7])

    def test_pairs_are_same_exposure_conflict_free_and_four_replacement_draws(self):
        store = LoggedStore({
            7: ([2, 2, 4, 5], [1, 0, 1, 0]),
            8: ([6, 7], [0, 0]),
            9: ([8, 8], [1, 0]),
            10: ([11, 12, 13, 14], [1, 0, 0, 0]),
            999: ([25, 26], [1, 0]),  # Must never be read.
        })
        rows = np.array([7, 8, 9, 10], dtype=np.int64)
        pairs, metadata = experiment.sample_pairs(store, rows, 17)
        self.assertEqual(store.read_rows, rows.tolist())
        self.assertEqual(pairs.dtype, np.dtype("int64"))
        self.assertEqual(pairs.shape, (12, 3))
        np.testing.assert_array_equal(pairs[:4], np.tile([7, 2, 5], (4, 1)))
        np.testing.assert_array_equal(pairs[4:8], np.tile([7, 4, 5], (4, 1)))
        self.assertEqual(metadata["negative_draws_per_positive"], 4)
        self.assertTrue(metadata["replacement"])
        self.assertEqual(metadata["skipped_no_positive"], 1)
        self.assertEqual(metadata["skipped_no_negative"], 1)
        for row, positive, negative in pairs:
            ids, labels = store.candidates_for(int(row))
            positive_ids = set(ids[labels == 1].tolist())
            self.assertIn(positive, positive_ids)
            self.assertIn(negative, ids[labels == 0])
            self.assertNotIn(negative, positive_ids)
        repeated, _ = experiment.sample_pairs(store, rows, 17)
        np.testing.assert_array_equal(pairs, repeated)

    def test_pair_sampling_preserves_multi_click_and_duplicate_positive_occurrences(self):
        store = LoggedStore({2: ([1, 1, 2, 3], [1, 1, 1, 0])})
        pairs, metadata = experiment.sample_pairs(store, np.array([2]), 29)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(int((pairs[:, 1] == 1).sum()), 8)
        self.assertEqual(int((pairs[:, 1] == 2).sum()), 4)
        self.assertTrue(np.all(pairs[:, 2] == 3))
        self.assertIn("multi-click", metadata["weighting"])
        for groups in ({0: ([1], [1])}, {0: ([1], [0])}):
            with self.assertRaisesRegex(ValueError, "no valid"):
                experiment.sample_pairs(LoggedStore(groups), np.array([0]), 17)
        for groups in ({0: ([1, 2], [1, 2])}, {0: ([0, 2], [1, 0])}):
            with self.assertRaisesRegex(ValueError, "invalid logged"):
                experiment.sample_pairs(LoggedStore(groups), np.array([0]), 17)

    def test_order_gate_empty_history_delta_zero_stable_ties_and_same_candidates(self):
        residual = np.array([-1., .25, 1., -.75, .5])
        for delta in (0, 2, 5, 10):
            order, scores = experiment.order_from_residual(residual, 5, delta)
            np.testing.assert_array_equal(scores, -np.arange(5) + delta * .5 * residual)
            np.testing.assert_array_equal(np.sort(order), np.arange(5))
        for gate in (False, True):
            for length, delta in ((0, 10), (5, 0)):
                order, scores = experiment.order_from_residual(residual, length, delta, gate)
                np.testing.assert_array_equal(order, np.arange(5))
                np.testing.assert_array_equal(scores, -np.arange(5))
        order, scores = experiment.order_from_residual(np.array([-1., 0., 1.]), 5, 1, gate=False)
        np.testing.assert_array_equal(scores, [-1., -1., -1.])
        np.testing.assert_array_equal(order, [0, 1, 2])
        order, scores = experiment.order_from_residual(np.empty(0), 0, 0)
        self.assertEqual(order.shape, (0,))
        self.assertEqual(scores.shape, (0,))

    def test_ranking_rejects_invalid_residuals_history_and_options(self):
        for residual in (np.zeros((2, 2)), np.array([np.nan]), np.array([1.01]), np.array([-1.01])):
            with self.assertRaises(ValueError):
                experiment.order_from_residual(residual, 5, 2)
        for length in (-1, 21, 1.5, True):
            with self.assertRaises(ValueError):
                experiment.order_from_residual([0.], length, 2)
        for delta in (-1, np.inf, np.nan, True, [2]):
            with self.assertRaises(ValueError):
                experiment.order_from_residual([0.], 5, delta)
        with self.assertRaises(ValueError):
            experiment.order_from_residual([0.], 5, 2, gate="false")

    def test_score_cache_averages_seed_axis_and_keeps_candidate_hit100_invariant(self):
        cache = {
            "rows": np.array([2, 8]), "lengths": np.array([12, 12]),
            "history_length": np.array([20, 0]),
            "hit": np.zeros((2, 100), dtype=bool),
            "clipped_scores": np.zeros((2, 3, 3, 100)),
        }
        cache["hit"][:, 10] = True
        cache["clipped_scores"][:, 0, :, 10] = [-1., 1., 1.]
        hits, hit100, invariant = experiment.score_cache(cache, 10, all_ablations=True)
        np.testing.assert_array_equal(hits["baseline"], [0, 0])
        np.testing.assert_array_equal(hits["full13"], [1, 0])
        np.testing.assert_array_equal(hits["no_category"], [0, 0])
        np.testing.assert_array_equal(hits["no_recent"], [0, 0])
        np.testing.assert_array_equal(hits["no_gate"], [1, 0])
        np.testing.assert_array_equal(hits["full13_seed17"], [0, 0])
        np.testing.assert_array_equal(hits["full13_seed29"], [1, 0])
        np.testing.assert_array_equal(hits["full13_seed43"], [1, 0])
        np.testing.assert_array_equal(hit100, [1, 1])
        self.assertTrue(invariant["per_request_candidate_and_hit100_invariant"])
        self.assertEqual(invariant["empty_history_exact_fallback_requests"], 1)
        zero_hits, _, _ = experiment.score_cache(cache, 0, all_ablations=True)
        for values in zero_hits.values():
            np.testing.assert_array_equal(values, [0, 0])

    def test_cache_roundtrip_bindings_hash_damage_and_unsafe_names(self):
        with workspace_temporary_directory() as temporary:
            directory = Path(temporary)
            original = np.arange(18, dtype=np.float32).reshape(3, 6)
            np.save(directory / "features.npy", original)
            np.save(directory / "rows.npy", np.array([1, 3, 5], dtype=np.int64))
            bindings = {"phase": "synthetic", "model_sha256": "test", "row_sha256": "test-rows"}
            manifest = experiment.save_cache_manifest(directory, bindings)
            loaded = experiment.load_cache(directory, bindings)
            try:
                self.assertEqual(set(loaded), {"features", "rows"})
                np.testing.assert_array_equal(loaded["features"], original)
                self.assertFalse(loaded["features"].flags.writeable)
            finally:
                experiment.close_arrays(loaded.values())
            with self.assertRaisesRegex(ValueError, "bindings mismatch"):
                experiment.load_cache(directory, {"phase": "different"})
            with self.assertRaises(FileExistsError):
                experiment.save_cache_manifest(directory, bindings)
            path = directory / "features.npy"
            before = path.read_bytes()
            corrupted = bytearray(before)
            corrupted[-1] ^= 1
            path.write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                experiment.load_cache(directory, bindings)
            path.write_bytes(before)
            manifest["arrays"] = {"../features.npy": "unused"}
            (directory / "cache.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid cache array name"):
                experiment.load_cache(directory, bindings)

    def test_build_cache_clips_each_seed_before_mean_and_uses_matching_effective_history(self):
        # Item 1 is non-PAD but has an all-zero feature vector; it must not count
        # as an effective history event. Items outside the last 20 are excluded.
        features = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0],
                             [0, 1, 0, 1], [1, 1, 1, 0], [1, -1, 0, 1]], dtype=np.float32)
        history = np.zeros((2, 22), dtype=np.int64)
        history[:, :2] = 3
        history[0, -2:] = [1, 2]
        store = LoggedStore({0: ([4, 5], [1, 0]), 1: ([4, 5], [0, 1])})
        store.history, store.features = history, features
        store.timestamps, store.users = np.array([110, 111]), np.array([7, 8])
        store.first_seen = np.array([999, 1, 1, 1, 1, 1])
        store.manifest = {"features": {"text_dimensions": 2}}

        class FakeGenerator:
            version, bindings = "synthetic-generator", {"synthetic": True}

            def retrieve(self, supplied, as_of):
                count = int(np.any(features[supplied[-20:]] != 0, axis=-1).sum())
                return SimpleNamespace(ids=np.array([4, 5]), cutoff=100,
                                       metadata={"history_length_used": count})

            def close(self):
                pass

        class FakeTemporal:
            version, manifest = "synthetic-temporal", {"arrays": {}}

            def close(self):
                pass

        class FakeModel:
            def __init__(self, values):
                self.values = np.array(values)

            def predict_raw(self, supplied):
                if supplied.shape != (2, 13):
                    raise AssertionError("unexpected candidate feature shape")
                return self.values.copy()

        models = {variant: [FakeModel([-100, 1]), FakeModel([1, 1]), FakeModel([1, 1])]
                  for variant in experiment.VARIANTS}
        config = SimpleNamespace(use_reranker=False, encoder_path="synthetic.npz",
                                 candidate_config=CandidateConfig(channels=("popular",), recent_hours=24))
        with workspace_temporary_directory() as temporary:
            args = SimpleNamespace(artifacts=temporary, v2_selection="unused.json", temporal="unused", data="unused")
            bindings = {"phase": "synthetic-build"}
            with patch.object(experiment.V2ServiceConfig, "from_selection", return_value=config), \
                    patch.object(experiment, "TemporalStore", return_value=FakeTemporal()), \
                    patch.object(experiment.FastCandidateGenerator, "from_paths", return_value=FakeGenerator()), \
                    patch.object(experiment, "verify_generator_selection"):
                directory, _ = experiment.build_cache(args, "synthetic", store, np.array([0, 1]), models, bindings)
            cache = experiment.load_cache(directory, bindings)
            try:
                np.testing.assert_array_equal(cache["history_length"], [1, 0])
                np.testing.assert_array_equal(cache["clipped_scores"][0, 0, :, 0], [-1, 1, 1])
                self.assertAlmostEqual(cache["clipped_scores"][0, 0, :, 0].mean(), 1 / 3)
                for row in (0, 1):
                    expected = content_features(features[history[row:row + 1, -20:]],
                                                features[np.array([[4, 5]])],
                                                history_mask=history[row:row + 1, -20:] > 0,
                                                text_dimensions=2)[0, :, :13]
                    np.testing.assert_array_equal(cache["features"][row, :2], expected)
                np.testing.assert_array_equal(cache["lengths"], [2, 2])
                self.assertTrue(np.all(cache["ids"][:, 2:] == 0))
            finally:
                experiment.close_arrays(cache.values())


if __name__ == "__main__":
    unittest.main()
