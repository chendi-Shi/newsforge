"""Exact parity, causal boundaries, bounded work and ownership for v3 serving."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import numpy as np

from newsforge.v2.retrieval import CandidateConfig
from newsforge.v3.fast_retrieval import FastCandidateGenerator
from tests.support import workspace_temporary_directory
from tests.test_v2_retrieval import fixture


POPULAR = CandidateConfig(channels=("popular",), recent_hours=24)


def fast_fixture(size=260, config=POPULAR, cache_size=8):
    original = fixture(size=size, config=config)
    return FastCandidateGenerator(
        original.features, original.article_embeddings, original.first_seen,
        original.train_clicks, original.train_exposures, original.temporal,
        original.encoder, config=config, cache_size=cache_size)


class V3FastRetrievalTest(unittest.TestCase):
    def assert_batches_equal(self, expected, actual):
        for field in fields(expected):
            left, right = getattr(expected, field.name), getattr(actual, field.name)
            with self.subTest(field=field.name):
                if isinstance(left, np.ndarray):
                    self.assertEqual(left.dtype, right.dtype)
                    np.testing.assert_array_equal(left, right)
                else:
                    self.assertEqual(left, right)

    def test_exact_parity_all_windows_budgets_frozen_feedback_and_histories(self):
        original, fast = fixture(config=POPULAR), fast_fixture()
        for recent_hours in (None, 24, 48, 72):
            for frozen in (False, True):
                for pool, route, constant in ((100, 200, 60), (5, 5, 1), (7, 11, 73)):
                    config = replace(POPULAR, recent_hours=recent_hours, freeze_feedback=frozen,
                                     pool_size=pool, route_size=route, rank_constant=constant)
                    for history in ([], [0, 0], [1], [2, 1, 2, 0, 1],
                                    list(range(1, 26)), list(range(1, 250))):
                        for as_of in (720000, 723599, 723600):
                            self.assert_batches_equal(original.retrieve(history, as_of, config=config),
                                                      fast.retrieve(history, as_of, config=config))

    def test_recency_cutoff_ties_fallback_and_zero_feature_history(self):
        original, fast = fixture(size=14, config=POPULAR), fast_fixture(size=14)
        for generator in (original, fast):
            generator.first_seen[:] = [np.iinfo(np.int64).max, 600000, 633599, 633600,
                                       716399, 716400, 720000, 725000, 633600, 633600,
                                       633600, -1, 700000, 700000]
            generator.temporal.clicks[:] = 0
            generator.temporal.exposures[:] = 0
            generator.features[1] = 0
        configs = [POPULAR, replace(POPULAR, pool_size=4, route_size=6),
                   replace(POPULAR, pool_size=2, route_size=2)]
        for config in configs:
            for history in ([], [1], [1, 1, 3, 9], [4, 12, 13, 3, 8, 9, 10], [5, 6]):
                expected = original.retrieve(history, 720000, config=config)
                actual = fast.retrieve(history, 720000, config=config)
                self.assert_batches_equal(expected, actual)
                self.assertNotIn(5, actual.ids)
                self.assertNotIn(6, actual.ids)
                self.assertNotIn(7, actual.ids)
        empty = fast.retrieve([], 720000)
        self.assertEqual(empty.ids.tolist(), [4, 12, 13, 3, 8, 9, 10, 2, 1])
        self.assertEqual(empty.sources["fallback"], [2, 1])
        self.assertEqual(empty.scores[-2:].tolist(), [0., 0.])

    def test_randomized_tie_heavy_catalogues_match_every_batch_field(self):
        rng = np.random.default_rng(820)
        for size in (1, 7, 101, 411):
            # The old fixture assumes at least two real rows.
            original, fast = fixture(size=max(size, 3), config=POPULAR), fast_fixture(size=max(size, 3))
            seen = rng.choice([600000, 633599, 633600, 650000, 700000, 716399, 716400],
                              size=len(original.features)).astype(np.int64)
            clicks = rng.integers(0, 3, size=len(seen)).astype(np.float32)
            exposures = clicks + rng.integers(0, 3, size=len(seen)).astype(np.float32)
            for generator in (original, fast):
                generator.first_seen[:] = seen
                generator.first_seen[0] = np.iinfo(np.int64).max
                generator.temporal.clicks[:] = clicks
                generator.temporal.exposures[:] = exposures
                if size == 1:
                    generator.first_seen[1:] = 725000
            for _ in range(12):
                history = ([] if size == 1 else rng.integers(0, len(seen), size=40).tolist())
                self.assert_batches_equal(original.retrieve(history, 720100), fast.retrieve(history, 720100))

    def test_large_rank_constant_preserves_float_rounding_id_tie_break(self):
        original, fast = fixture(config=POPULAR), fast_fixture()
        for constant in (2 ** 53, 10 ** 30, 10 ** 100):
            config = replace(POPULAR, rank_constant=constant)
            self.assert_batches_equal(original.retrieve([], 720100, config=config),
                                      fast.retrieve([], 720100, config=config))

    def test_nonpopular_recipes_delegate_without_query_cache_contamination(self):
        original, fast = fixture(), fast_fixture(config=CandidateConfig())
        cases = [(CandidateConfig(), [1, 2], 720100, True),
                 (POPULAR, [5], 720101, True),
                 (CandidateConfig(channels=("tower",)), [4, 3], 720101, True),
                 (CandidateConfig(channels=("content", "popular")), [3], 723700, False),
                 (POPULAR, [], 723700, False)]
        for config, history, as_of, reuse in cases:
            self.assert_batches_equal(
                original.retrieve(history, as_of, config=config, reuse_query_routes=reuse),
                fast.retrieve(history, as_of, config=config, reuse_query_routes=reuse))
        self.assertEqual(len(fast._query_routes), 0)

    def test_invalid_inputs_and_model_maturity_match_reference(self):
        original, fast = fixture(config=POPULAR), fast_fixture()
        for history, as_of in (([1], 900000), ([999], 720100), ([-1], 720100),
                              ([1.5], 720100), ([259], 720100), ([1], 720100.5),
                              ([[1]], 720100), ([True], 720100), ([1], True)):
            for generator in (original, fast):
                with self.subTest(history=history, as_of=as_of, generator=type(generator).__name__):
                    with self.assertRaises(ValueError):
                        generator.retrieve(history, as_of)
        with self.assertRaisesRegex(ValueError, "boolean"):
            fast.retrieve([], 720100, reuse_query_routes=1)
        for generator in (original, fast):
            generator.encoder.metadata["train_cutoff_timestamp"] = 716400
            with self.assertRaisesRegex(ValueError, "training cutoff"):
                generator.retrieve([], 720000)
        self.assert_batches_equal(original.retrieve([], 723600), fast.retrieve([], 723600))

    def test_invalid_temporal_statistics_fail_before_cache_population(self):
        for bad in (np.nan, -1, np.inf):
            fast = fast_fixture()
            fast.temporal.clicks[1] = bad
            with self.assertRaisesRegex(ValueError, "popularity statistics"):
                fast.retrieve([], 720100)
            self.assertEqual(fast.cache_info()["entries"], 0)
        fast = fast_fixture()
        fast.temporal.exposures = np.zeros(2)
        with self.assertRaisesRegex(ValueError, "popularity statistics"):
            fast.retrieve([], 720100)

    def test_popular_path_skips_neural_encoder_and_bounds_history_filter_work(self):
        fast = fast_fixture(size=10001)
        with patch.object(fast.encoder, "encode_user", side_effect=AssertionError("unused encoder")), \
                patch("newsforge.v3.fast_retrieval.np.isin", wraps=np.isin) as isin:
            fast.retrieve([1, 2, 2, 3], 720100)
            self.assertEqual(isin.call_count, 1)
            self.assertEqual(len(isin.call_args.args[0]), 203)
        self.assertEqual(fast.cache_info()["entries"], 1)

    def test_cached_arrays_are_readonly_bounded_and_outputs_are_independent(self):
        fast = fast_fixture(cache_size=1)
        first = fast.retrieve([1], 720100)
        expected = fast.retrieve([1], 720100)
        first.ids[:] = 0
        first.scores[:] = -1
        first.source_mask[:] = 0
        first.sources["popular"][:] = [999]
        first.metadata["route_counts"]["popular"] = -1
        self.assert_batches_equal(expected, fast.retrieve([1], 720100))
        for entry in fast._popular_cache.values():
            for array in (entry.global_order, entry.recent_order, entry.recent_membership):
                self.assertFalse(array.flags.writeable)
        stats = fast.cache_info()
        self.assertEqual((stats["misses"], stats["hits"]), (1, 2))
        self.assertLessEqual(stats["array_bytes"], len(fast.first_seen) * 17)
        fast.retrieve([1], 723700)
        self.assertEqual(fast.cache_info()["entries"], 1)
        self.assertEqual(fast.cache_info()["misses"], 2)
        fast.clear_cache()
        self.assertEqual(fast.cache_info()["array_bytes"], 0)

    def test_batch_reuses_only_snapshot_state_and_matches_concurrent_requests(self):
        fast, original = fast_fixture(), fixture(config=POPULAR)
        requests = [([index, index, 0], 720100 + index) for index in range(1, 17)]
        expected = [original.retrieve(*request) for request in requests]
        actual = fast.retrieve_batch(requests)
        self.assertEqual(fast.cache_info()["misses"], 1)
        self.assertEqual(fast.cache_info()["hits"], 15)
        for before, after in zip(expected, actual):
            self.assert_batches_equal(before, after)
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(lambda request: fast.retrieve(*request), requests))
        for before, after in zip(expected, parallel):
            self.assert_batches_equal(before, after)
        for invalid in ([], requests * 3):
            with self.assertRaisesRegex(ValueError, "batch size"):
                fast.retrieve_batch(invalid)

    def test_close_is_idempotent_serialized_and_rejects_later_reads(self):
        fast = fast_fixture()
        entered, release = threading.Event(), threading.Event()
        implementation = fast._retrieve_popular

        def delayed(*args):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release request")
            return implementation(*args)

        with patch.object(fast, "_retrieve_popular", side_effect=delayed), ThreadPoolExecutor(max_workers=2) as pool:
            reading = pool.submit(fast.retrieve, [], 720100)
            self.assertTrue(entered.wait(timeout=5))
            closing = pool.submit(fast.close)
            self.assertFalse(closing.done())
            release.set()
            self.assertEqual(len(reading.result(timeout=5).ids), 100)
            closing.result(timeout=5)
        fast.close()
        self.assertTrue(fast.cache_info()["closed"])
        self.assertEqual(fast.cache_info()["array_bytes"], 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            fast.retrieve([], 720100)

    def test_close_releases_owned_maps_but_not_borrowed_temporal_store(self):
        fast = fast_fixture()
        with workspace_temporary_directory(prefix="v3-fast-mapping-") as directory:
            features_path, embeddings_path = Path(directory) / "features.npy", Path(directory) / "embeddings.npy"
            np.save(features_path, fast.features)
            np.save(embeddings_path, fast.article_embeddings)
            fast.features = np.load(features_path, mmap_mode="r", allow_pickle=False)
            fast.article_embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
            with patch.object(fast.temporal, "close", create=True) as close_temporal:
                fast.retrieve([], 720100)
                fast.close()
                self.assertTrue(fast.features._mmap.closed)
                self.assertTrue(fast.article_embeddings._mmap.closed)
                close_temporal.assert_not_called()

    def test_version_distinguishes_optimization_from_frozen_reference(self):
        original, fast = fixture(config=POPULAR), fast_fixture()
        self.assertEqual(original.bindings, fast.bindings)
        self.assertNotEqual(original.version, fast.version)
        self.assertEqual(fast.version, fast.version)


if __name__ == "__main__":
    unittest.main()
