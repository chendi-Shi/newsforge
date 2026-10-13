"""Causal catalogue, fusion and NumPy inference boundary tests, no real labels."""

from dataclasses import fields, replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from newsforge.v2.retrieval import (CandidateConfig, CandidateGenerator, NumpyMeanEncoder,
                                   CHANNEL_BITS, rerank_order, sha256_file)
from tests.support import workspace_temporary_directory


class FakeTemporal:
    def __init__(self, clicks, exposures, start=720000):
        self.clicks, self.exposures, self.start = clicks, exposures, start
        self.train_cutoff = 700000

    def frozen_snapshot(self, as_of):
        snapshot = self.snapshot(as_of)
        if snapshot.cutoff <= self.train_cutoff:
            raise ValueError("frozen feedback cutoff must exceed training cutoff")
        snapshot.clicks = self.clicks[::-1].copy()
        snapshot.exposures = self.exposures[::-1].copy()
        return snapshot

    def snapshot(self, as_of):
        if not self.start <= as_of < self.start + 7200:
            raise ValueError("snapshot hour outside coverage")
        hour = as_of // 3600 * 3600
        return SimpleNamespace(index=hour // 3600, as_of_start=hour, cutoff=hour - 3600,
                               clicks=self.clicks, exposures=self.exposures)


def fixture(size=260, config=None):
    rng = np.random.default_rng(21)
    features = rng.uniform(.1, 1, size=(size, 3)).astype(np.float32)
    features[0] = 0
    embeddings = features[:, :2].copy()
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    first_seen = np.full(size, 710000, dtype=np.int64)
    first_seen[0] = np.iinfo(np.int64).max
    first_seen[-2:] = [716400, 725000]  # Exact cutoff and future: both disallowed.
    clicks = np.arange(size, dtype=np.float32)
    exposures = clicks * 2
    encoder = NumpyMeanEncoder({"w0": np.eye(3, dtype=np.float32), "b0": np.zeros(3, np.float32),
                               "w2": np.eye(2, 3, dtype=np.float32), "b2": np.zeros(2, np.float32)})
    temporal = FakeTemporal(clicks, exposures)
    generator = CandidateGenerator(features, embeddings, first_seen, clicks[::-1].copy(),
                                   exposures[::-1].copy(), temporal, encoder, config=config)
    return generator


class V2RetrievalTest(unittest.TestCase):
    def assert_batches_equal(self, before, after):
        for field in fields(before):
            left, right = getattr(before, field.name), getattr(after, field.name)
            with self.subTest(field=field.name):
                if isinstance(left, np.ndarray):
                    np.testing.assert_array_equal(left, right)
                else:
                    self.assertEqual(left, right)

    def test_fixed_pool_strict_cutoff_all_history_seen_and_sources(self):
        generator = fixture()
        history = list(range(1, 26)) + [1, 0]
        result = generator.retrieve(history, 720100)
        self.assertEqual(len(result.ids), 100)
        self.assertEqual(len(set(result.ids)), 100)
        self.assertFalse(set(history) & set(result.ids))
        self.assertTrue((generator.first_seen[result.ids] < result.cutoff).all())
        self.assertNotIn(258, result.ids)
        self.assertNotIn(259, result.ids)
        self.assertEqual(result.metadata["history_events_encoded"], 20)
        self.assertEqual(result.metadata["route_counts"], {"popular": 200, "content": 200, "tower": 200})
        self.assertTrue((result.source_mask > 0).all())

    def test_rrf_votes_and_tie_policy_match_declared_routes(self):
        generator = fixture()
        result = generator.retrieve([1, 2], 720100)
        expected, masks = {}, {}
        for channel, ids in result.sources.items():
            self.assertNotEqual(channel, "fallback")
            for rank, item in enumerate(ids, 1):
                expected[item] = expected.get(item, 0.) + 1 / (60 + rank)
                masks[item] = masks.get(item, 0) | CHANNEL_BITS[channel]
        ids = np.array(list(expected), dtype=np.int64)
        scores = np.array([expected[int(item)] for item in ids])
        wanted = ids[rerank_order(ids, scores)[:100]]
        np.testing.assert_array_equal(result.ids, wanted)
        np.testing.assert_allclose(result.scores, [expected[int(item)] for item in wanted], atol=0, rtol=0)
        np.testing.assert_array_equal(result.source_mask, [masks[int(item)] for item in wanted])
        np.testing.assert_array_equal(rerank_order([8, 2, 5], [1., 1., 2.]), [2, 1, 0])

    def test_window_backfills_same_cutoff_without_changing_denominator(self):
        generator = fixture(config=CandidateConfig(recent_hours=24))
        generator.first_seen[1:250] = 600000  # Older than 24h, still valid fallback.
        result = generator.retrieve([250], 720100)
        self.assertEqual(len(result.ids), 100)
        self.assertEqual(result.metadata["recent_unseen_catalogue_count"], 7)
        self.assertEqual(result.metadata["fallback_count"], 93)
        self.assertEqual(result.sources["fallback"][0], 249)
        self.assertTrue((generator.first_seen[result.ids] < 716400).all())
        self.assertTrue((result.scores[-93:] == 0).all())

    def test_no_freshness_removes_only_window(self):
        generator = fixture()
        generator.first_seen[1:240] = 500000
        result = generator.retrieve([2], 720100, config=CandidateConfig(recent_hours=None))
        self.assertEqual(result.metadata["recent_unseen_catalogue_count"], 256)
        self.assertEqual(result.metadata["fallback_count"], 0)
        self.assertTrue((generator.first_seen[result.ids] < result.cutoff).all())

    def test_freeze_feedback_changes_popularity_without_catalogue_change(self):
        generator = fixture(config=CandidateConfig(channels=("popular",)))
        dynamic = generator.retrieve([], 720100)
        frozen = generator.retrieve([], 720100, config=replace(generator.config, freeze_feedback=True))
        self.assertEqual(dynamic.ids[0], 257)
        self.assertEqual(frozen.ids[0], 1)
        self.assertEqual(dynamic.metadata["visible_catalogue_count"], frozen.metadata["visible_catalogue_count"])
        self.assertEqual(frozen.metadata["feedback_source"], "frozen_training")
        generator.temporal.train_cutoff = 720000
        with self.assertRaisesRegex(ValueError, "frozen feedback cutoff"):
            generator.retrieve([], 720100, config=replace(generator.config, freeze_feedback=True))

    def test_zero_click_popularity_uses_exposure_then_first_seen_then_id(self):
        generator = fixture(size=8, config=CandidateConfig(pool_size=3, route_size=4, channels=("popular",)))
        generator.temporal.clicks[:] = 0
        generator.temporal.exposures[:] = 0
        generator.temporal.exposures[[1, 2, 3]] = 7
        generator.first_seen[[1, 2, 3]] = [710001, 710001, 710002]
        result = generator.retrieve([], 720100)
        np.testing.assert_array_equal(result.ids, [3, 1, 2])

    def test_zero_vectors_fallback_is_counted_and_does_not_raise(self):
        generator = fixture(config=CandidateConfig(channels=("tower",)))
        generator.encoder.arrays["w2"][:] = 0
        result = generator.retrieve([1], 720100)
        self.assertEqual(result.metadata["zero_query_channels"], ["tower"])
        self.assertEqual(result.metadata["fallback_count"], 100)
        self.assertEqual(result.source_mask.tolist(), [CHANNEL_BITS["fallback"]] * 100)
        self.assertFalse(result.metadata["empty_usable_history"])

    def test_empty_history_small_catalogue_and_no_future_padding(self):
        result = fixture(size=8).retrieve([], 720100)
        self.assertEqual(result.ids.tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(result.shortfall, 95)
        self.assertTrue(result.metadata["empty_usable_history"])
        self.assertEqual(result.metadata["zero_query_channels"], ["content", "tower"])

    def test_snapshot_and_history_validation(self):
        generator = fixture()
        for history, as_of in (([1], 900000), ([999], 720100), ([-1], 720100),
                               ([1.5], 720100), ([259], 720100), ([1], 720100.5)):
            with self.subTest(history=history, as_of=as_of):
                with self.assertRaises(ValueError):
                    generator.retrieve(history, as_of)
        generator.encoder.metadata["train_cutoff_timestamp"] = 720200
        with self.assertRaisesRegex(ValueError, "training cutoff"):
            generator.retrieve([1], 720100)

    def test_cache_reuses_hour_recipe_and_bounds_entries(self):
        generator = fixture()
        generator.retrieve([1], 720001)
        value = next(iter(generator._cache.values()))
        generator.retrieve([2], 720002)
        self.assertIs(next(iter(generator._cache.values())), value)
        generator._cache_size = 1
        generator.retrieve([1], 723700)
        self.assertEqual(len(generator._cache), 1)

    def test_model_feedback_must_precede_cutoff_including_exact_hour(self):
        generator = fixture()
        for training_cutoff in (716400, 716401):
            generator.encoder.metadata["train_cutoff_timestamp"] = training_cutoff
            for as_of in (720000, 723599):
                with self.subTest(training_cutoff=training_cutoff, as_of=as_of):
                    with self.assertRaisesRegex(ValueError, "training cutoff"):
                        generator.retrieve([], as_of)
            self.assertEqual(len(generator.retrieve([], 723600).ids), 100)

    def test_query_route_cache_is_exact_across_all_nine_ablation_recipes(self):
        reference, cached = fixture(), fixture()
        full = CandidateConfig()
        configs = [full, replace(full, channels=("content", "tower")),
                   replace(full, channels=("popular", "tower")),
                   replace(full, channels=("popular", "content")),
                   replace(full, channels=("popular",)), replace(full, channels=("content",)),
                   replace(full, channels=("tower",)), replace(full, recent_hours=None),
                   replace(full, freeze_feedback=True)]
        with patch.object(reference, "_dot", wraps=reference._dot) as uncached_dot, \
                patch.object(cached, "_dot", wraps=cached._dot) as cached_dot:
            for config in configs:
                before = reference.retrieve([1, 2, 3], 720100, config=config)
                after = cached.retrieve([1, 2, 3], 720100, config=config, reuse_query_routes=True)
                self.assert_batches_equal(before, after)
            self.assertEqual(uncached_dot.call_count, 12)
            self.assertEqual(cached_dot.call_count, 4)
        self.assertEqual(len(cached._query_routes), 4)
        self.assertTrue(all(len(ids) <= full.route_size for ids in cached._query_routes.values()))
        # The public response cannot mutate the internal cached top IDs.
        after.sources["content"][0] = 999
        self.assert_batches_equal(reference.retrieve([1, 2, 3], 720100),
                                  cached.retrieve([1, 2, 3], 720100, reuse_query_routes=True))

    def test_query_route_cache_keys_cover_full_history_time_window_and_budget(self):
        reference, cached = fixture(), fixture()
        history = list(range(1, 26))
        changed_history = [26, *range(2, 26)]  # Same last twenty, different seen set.
        cases = [(history, 720100, CandidateConfig()),
                 (changed_history, 720100, CandidateConfig()),
                 (changed_history, 720101, CandidateConfig()),
                 (changed_history, 720101, CandidateConfig(recent_hours=24)),
                 (changed_history, 720101, CandidateConfig(recent_hours=72)),
                 (changed_history, 720101, CandidateConfig(recent_hours=None)),
                 (changed_history, 720101, CandidateConfig(recent_hours=None, route_size=201))]
        with patch.object(cached, "_dot", wraps=cached._dot) as dot:
            for index, (history, as_of, config) in enumerate(cases, 1):
                before = reference.retrieve(history, as_of, config=config)
                after = cached.retrieve(history, as_of, config=config, reuse_query_routes=True)
                self.assert_batches_equal(before, after)
                self.assertEqual(dot.call_count, index * 2)
            # Pool truncation and RRF constant do not alter either semantic route.
            config = replace(cases[-1][2], pool_size=50, rank_constant=40)
            before = reference.retrieve(history, as_of, config=config)
            after = cached.retrieve(history, as_of, config=config, reuse_query_routes=True)
            self.assert_batches_equal(before, after)
            self.assertEqual(dot.call_count, len(cases) * 2)
            cached.retrieve([1], 720101, reuse_query_routes=True)
            self.assertEqual(len(cached._query_routes), 2)
            self.assertEqual(dot.call_count, len(cases) * 2 + 2)

    def test_query_route_cache_preserves_empty_zero_and_exact_tie_cases(self):
        for history in ([], [1]):
            reference, cached = fixture(), fixture()
            reference.encoder.arrays["w2"][:] = 0
            cached.encoder.arrays["w2"][:] = 0
            for channels in (("content", "tower"), ("tower",), ("popular", "tower")):
                config = CandidateConfig(channels=channels)
                self.assert_batches_equal(reference.retrieve(history, 720100, config=config),
                    cached.retrieve(history, 720100, config=config, reuse_query_routes=True))
        reference, cached = fixture(), fixture()
        def tied(matrix, ids, query, **kwargs):
            return np.zeros(len(ids), dtype=np.float32)
        with patch.object(reference, "_dot", side_effect=tied), patch.object(cached, "_dot", side_effect=tied) as dot:
            for config in (CandidateConfig(), CandidateConfig(channels=("content", "tower")),
                           CandidateConfig(freeze_feedback=True)):
                self.assert_batches_equal(reference.retrieve([1], 720100, config=config),
                    cached.retrieve([1], 720100, config=config, reuse_query_routes=True))
            self.assertEqual(dot.call_count, 2)

    def test_default_path_does_not_reuse_or_retain_query_routes(self):
        generator = fixture()
        generator.retrieve([1], 720100, reuse_query_routes=True)
        with patch.object(generator, "_dot", wraps=generator._dot) as dot:
            generator.retrieve([1], 720100)
            generator.retrieve([1], 720100)
            self.assertEqual(dot.call_count, 4)
        self.assertEqual(len(generator._query_routes), 0)
        self.assertIsNone(generator._route_query_key)
        with self.assertRaisesRegex(ValueError, "boolean"):
            generator.retrieve([1], 720100, reuse_query_routes=1)

    def test_numpy_encoder_mask_empty_and_hash_binding(self):
        encoder = fixture().encoder
        values = np.array([[[3., 4., 0.], [300., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]]], np.float32)
        result = encoder.encode_user(values, np.array([[True, False], [True, True]]))
        np.testing.assert_allclose(result, [[.6, .8], [0., 0.]], atol=1e-7)
        np.testing.assert_array_equal(encoder.encode_user(np.zeros((2, 0, 3), np.float32)), np.zeros((2, 2)))
        with workspace_temporary_directory(prefix="v2-encoder-") as directory:
            path = Path(directory) / "encoder.npz"
            np.savez(path, **encoder.arrays)
            path.with_suffix(".npz.json").write_text(json.dumps({"npz_sha256": sha256_file(path)}))
            loaded = NumpyMeanEncoder.load(path)
            np.testing.assert_array_equal(loaded.encode_user(values), encoder.encode_user(values))
            path.with_suffix(".npz.json").write_text(json.dumps({"npz_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                NumpyMeanEncoder.load(path)

    def test_invalid_recipe(self):
        for arguments in ({"pool_size": True}, {"route_size": 10}, {"channels": ()},
                          {"channels": ("tower", "tower")}, {"recent_hours": 6}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    CandidateConfig(**arguments)


if __name__ == "__main__":
    unittest.main()
