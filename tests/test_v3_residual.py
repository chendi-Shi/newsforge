"""Pairwise gradient, training boundaries, bounded reordering and model provenance."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from newsforge.v3.residual import FEATURE_NAMES, L2, PairwiseResidual, _loss_gradient
from tests.support import workspace_temporary_directory


def pairs():
    rng = np.random.default_rng(802)
    negative = rng.normal(size=(80, 13))
    positive = negative + rng.normal(scale=.15, size=(80, 13))
    positive[:, 0] += 1.
    positive[:, 10] += .4
    positive[:, 12] += .3
    positive[:, 3] = negative[:, 3] = 7.  # An exact constant dimension.
    return positive, negative


class V3ResidualTest(unittest.TestCase):
    def fitted(self, variant="full13"):
        return PairwiseResidual(variant).fit(*pairs(), bindings={"data": "synthetic", "fit_max_exclusive": 16})

    def test_analytic_gradient_matches_central_difference(self):
        rng = np.random.default_rng(10)
        differences, weights = rng.normal(size=(40, 7)), rng.normal(size=7)
        loss, gradient = _loss_gradient(weights, differences)
        expected = []
        for column in range(len(weights)):
            step = np.zeros_like(weights)
            step[column] = 1e-6
            expected.append((_loss_gradient(weights + step, differences)[0]
                             - _loss_gradient(weights - step, differences)[0]) / 2e-6)
        np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-8)
        self.assertGreater(loss, .5 * L2 * np.dot(weights, weights))

    def test_fit_only_normalization_and_deterministic_preference(self):
        positive, negative = pairs()
        model, repeat = self.fitted(), self.fitted()
        all_fit = np.concatenate((positive, negative))
        np.testing.assert_allclose(model.mean, all_fit.mean(axis=0), atol=1e-14)
        expected_scale = all_fit.std(axis=0)
        expected_scale[expected_scale == 0] = 1.
        np.testing.assert_allclose(model.scale, expected_scale, atol=1e-14)
        np.testing.assert_array_equal(model.weights, repeat.weights)
        self.assertGreater(np.mean(model.predict_raw(positive) > model.predict_raw(negative)), .95)
        before = (model.mean.copy(), model.scale.copy(), model.weights.copy())
        model.predict_raw(np.full((3, 13), 1000.))
        for old, now in zip(before, (model.mean, model.scale, model.weights)):
            np.testing.assert_array_equal(old, now)
        self.assertTrue(model.fit_metadata["optimizer_success"])
        with self.assertRaisesRegex(ValueError, "refitting"):
            model.fit(positive, negative)

    def test_feature_schema_matches_frozen_v2_and_ablations_ignore_columns(self):
        from newsforge.v2.ranker import CONTENT_NAMES
        self.assertEqual(FEATURE_NAMES, CONTENT_NAMES[:13])
        positive, negative = pairs()
        for variant, deleted in (("no_category", [10, 11]), ("no_recent", [12])):
            model = self.fitted(variant)
            changed_positive, changed_negative = positive.copy(), negative.copy()
            changed_positive[:, deleted] = 1000
            changed_negative[:, deleted] = -1000
            altered = PairwiseResidual(variant).fit(changed_positive, changed_negative)
            np.testing.assert_array_equal(model.weights, altered.weights)
            np.testing.assert_array_equal(model.predict_raw(positive), model.predict_raw(changed_positive))

    def test_gate_clipping_and_exact_popularity_fallback(self):
        model = self.fitted()
        values = np.tile(model.mean, (80, 1))
        values[:, 0] += np.linspace(-100, 100, len(values))
        raw = model.predict_raw(values)
        plain = model.score(values, history_length=5, gate=False)
        np.testing.assert_array_equal(plain, np.clip(raw, -1., 1.))
        np.testing.assert_array_equal(model.score(values, history_length=5), plain * .5)
        for gate in (True, False):
            np.testing.assert_array_equal(model.score(values, history_length=0, gate=gate), np.zeros(80))
            for history, delta in ((5, 0), (0, 10)):
                order, scores = model.rank(values, history_length=history, delta=delta, gate=gate)
                np.testing.assert_array_equal(order, np.arange(80))
                np.testing.assert_array_equal(scores, -np.arange(80))

    def test_pairwise_inversion_bound_and_exact_ties_keep_popularity_order(self):
        model = self.fitted()
        values = np.zeros((60, 13))
        residual = np.tile([-1., 1., -.2, .2], 15)
        for delta in (2, 5, 10):
            with patch.object(model, "score", return_value=residual):
                order, scores = model.rank(values, history_length=20, delta=delta, gate=False)
            position = np.argsort(order)
            for earlier in range(len(values)):
                for later in range(earlier + 1, len(values)):
                    if position[later] < position[earlier]:
                        self.assertLess(later - earlier, 2 * delta)
                    if scores[earlier] == scores[later]:
                        self.assertLess(position[earlier], position[later])
        # Rank 2 receives +1 and rank 0 receives -1, hence an exact delta=1 tie.
        with patch.object(model, "score", return_value=np.array([-1., 0., 1.])):
            order, scores = model.rank(np.zeros((3, 13)), history_length=1, delta=1, gate=False)
        np.testing.assert_array_equal(scores, [-1., -1., -1.])
        np.testing.assert_array_equal(order, [0, 1, 2])

    def test_invalid_shapes_nonfinite_history_and_delta_are_rejected(self):
        positive, negative = pairs()
        for bad in (np.zeros((0, 13)), np.zeros((3, 12)), np.zeros(13), np.full((3, 13), np.nan)):
            with self.assertRaises(ValueError):
                PairwiseResidual().fit(bad, bad)
        with self.assertRaises(ValueError):
            PairwiseResidual().fit(positive, negative[:-1])
        with self.assertRaises(ValueError):
            PairwiseResidual("unknown")
        with self.assertRaises(ValueError):
            PairwiseResidual().predict_raw(positive)
        model = self.fitted()
        for history in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                model.score(positive, history_length=history)
        for delta in (-1, np.inf, np.nan, True, "2"):
            with self.assertRaises(ValueError):
                model.rank(positive, history_length=2, delta=delta)
        with self.assertRaises(ValueError):
            model.score(positive, history_length=2, gate="false")
        with self.assertRaises(ValueError):
            model.predict_raw(np.full((1, 13), np.inf))
        order, scores = model.rank(np.empty((0, 13)), history_length=0, delta=0)
        self.assertEqual(order.shape, (0,))
        self.assertEqual(scores.shape, (0,))

    def test_optimizer_failure_is_not_published_as_fitted(self):
        failure = SimpleNamespace(success=False, status=1, message="iteration budget exhausted")
        model = PairwiseResidual()
        with patch("newsforge.v3.residual.minimize", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "optimizer failed.*status=1"):
                model.fit(*pairs())
        self.assertIsNone(model.weights)
        self.assertEqual(model.fit_metadata, {})

    def test_save_load_bindings_no_overwrite_and_no_pickle(self):
        model = self.fitted("no_category")
        with workspace_temporary_directory() as temporary:
            path = Path(temporary) / "residual.npz"
            metadata = model.save(path)
            loaded = PairwiseResidual.load(path, expected_bindings=model.bindings)
            np.testing.assert_array_equal(model.predict_raw(pairs()[0]), loaded.predict_raw(pairs()[0]))
            self.assertEqual(metadata["variant"], "no_category")
            self.assertEqual(model.fit_metadata, loaded.fit_metadata)
            with np.load(path, allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), {"mean", "scale", "weights"})
            with self.assertRaises(FileExistsError):
                model.save(path)
            with self.assertRaisesRegex(ValueError, "bindings mismatch"):
                PairwiseResidual.load(path, expected_bindings={"data": "different"})

    def test_same_size_corruption_schema_tampering_and_invalid_scale_rejected(self):
        model = self.fitted()
        with workspace_temporary_directory() as temporary:
            path = Path(temporary) / "residual.npz"
            metadata = model.save(path)
            original = path.read_bytes()
            damaged = bytearray(original)
            damaged[-25] ^= 1
            path.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                PairwiseResidual.load(path)
            path.write_bytes(original)
            sidecar = path.with_suffix(".npz.json")
            wrong_schema = dict(metadata, feature_names=list(reversed(metadata["feature_names"])))
            sidecar.write_text(json.dumps(wrong_schema), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "variant mismatch"):
                PairwiseResidual.load(path)
            # Recompute the outer hash to exercise semantic validation too.
            with path.open("wb") as stream:
                np.savez(stream, mean=model.mean, scale=np.zeros_like(model.scale), weights=model.weights)
            metadata["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scales must be positive"):
                PairwiseResidual.load(path)


if __name__ == "__main__":
    unittest.main()
