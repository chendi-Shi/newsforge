"""Negative-cache contracts and an optional, explicitly synthetic benchmark."""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import unittest

import numpy as np
from numpy.lib.format import open_memmap

from newsforge.store import Store
from newsforge.train import choose_negatives
from tests.support import workspace_temporary_directory


def close_cache(store):
    for name in ("pool", "positive_rows", "positive_items", "mean_history"):
        mapping = getattr(getattr(store, name, None), "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()


@contextmanager
def fixture(candidates, labels, offsets, split, history=None, features=None):
    with workspace_temporary_directory(prefix="newsforge-sampling-") as directory:
        store = Store.__new__(Store)
        store.path = Path(directory)
        store.candidates = np.asarray(candidates, dtype=np.int32)
        store.labels = np.asarray(labels, dtype=np.int8)
        store.offsets = np.asarray(offsets, dtype=np.int64)
        store.split = np.asarray(split, dtype=np.uint8)
        if features is None:
            count = int(store.candidates.max(initial=0)) + 1
            features = np.arange(count * 4, dtype=np.float32).reshape(count, 4)
            features[0] = 0
        store.features = np.asarray(features, dtype=np.float32)
        store.history = (np.zeros((len(split), 2), dtype=np.int32) if history is None
                         else np.asarray(history, dtype=np.int32))
        (store.path / "manifest.json").write_text(json.dumps({"fixture": True, "rows": len(split)}), encoding="utf-8")
        np.save(store.path / "news_features.npy", store.features)
        try:
            yield store
        finally:
            close_cache(store)


def assert_cache_contract(test, store, pool_size):
    expected_rows, expected_items = [], []
    for row in range(len(store.split)):
        candidates, labels = store.candidates_for(row)
        positives = candidates[labels == 1]
        negatives = candidates[(labels == 0) & ~np.isin(candidates, positives)]
        if store.split[row] != 0 or not len(positives) or not len(negatives):
            np.testing.assert_array_equal(store.pool[row], 0)
            continue
        test.assertTrue(np.isin(store.pool[row], negatives).all())
        test.assertFalse(np.isin(store.pool[row], positives).any())
        test.assertTrue((store.pool[row] > 0).all())
        if len(negatives) >= pool_size:
            available, selected = Counter(negatives.tolist()), Counter(store.pool[row].tolist())
            test.assertTrue(all(count <= available[item] for item, count in selected.items()))
        expected_rows.extend([row] * len(positives))
        expected_items.extend(positives.tolist())
    np.testing.assert_array_equal(store.positive_rows, expected_rows)
    np.testing.assert_array_equal(store.positive_items, expected_items)
    features = store.features[store.history]
    expected_mean = features.sum(1) / np.maximum(np.any(features != 0, axis=-1).sum(1, keepdims=True), 1)
    np.testing.assert_allclose(store.mean_history, expected_mean, atol=1e-6, rtol=1e-6)


class SamplingTest(unittest.TestCase):
    def test_hard_negatives_are_top_two_and_stay_inside_pool(self):
        pool = np.array([[3, 4, 5, 6], [7, 8, 9, 10]])
        scores = np.array([[1, 4, 3, 2], [8, 9, 6, 5]])
        selected = choose_negatives(pool, scores, np.random.default_rng(17), True)
        np.testing.assert_array_equal(selected[:, :2], [[4, 5], [8, 7]])
        for row in range(2):
            self.assertTrue(np.isin(selected[row], pool[row]).all())

    def test_cache_excludes_clicked_ids_and_heldout_impressions(self):
        features = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
        with fixture([1, 1, 2, 3, 2, 3], [1, 0, 0, 1, 0, 1], [0, 3, 4, 6], [0, 0, 2],
                     [[1, 0], [2, 3], [0, 0]], features) as store:
            store.ensure_training_cache(4)
            assert_cache_contract(self, store, 4)
            np.testing.assert_array_equal(store.pool[0], 2)
            np.testing.assert_allclose(store.mean_history[1], [.5, 1])
            marker = json.loads((store.path / "training_cache.json").read_text())
            self.assertEqual(marker["positive_training_examples"], 1)
            self.assertEqual(marker["skipped_no_positive_or_negative"], 1)

    def test_ragged_empty_multiple_positive_and_duplicate_boundaries(self):
        groups = [[], [1, 2], [3, 4], [5, 6, 5, 6, 7, 8],
                  [1, 2, 2, 3, 4, 4, 5], [8, 9, 10, 11], [1, 2, 3]]
        labels = [[], [0, 0], [1, 1], [1, 1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0]]
        offsets = np.r_[0, np.cumsum([len(group) for group in groups])]
        for pool_size in (1, 3, 4, 16):
            with self.subTest(pool_size=pool_size), fixture(np.concatenate(groups), np.concatenate(labels), offsets,
                                                          [0, 0, 0, 0, 0, 1, 2]) as store:
                store.ensure_training_cache(pool_size)
                assert_cache_contract(self, store, pool_size)

    def test_all_skipped_groups_and_no_training_rows(self):
        for split in ([0, 0], [1, 2]):
            with self.subTest(split=split), fixture([1, 2], [1, 0], [0, 1, 2], split) as store:
                store.ensure_training_cache(4)
                assert_cache_contract(self, store, 4)
                self.assertEqual(len(store.positive_rows), 0)

    def test_fixed_seed_and_heldout_labels_cannot_change_cache(self):
        candidates = np.tile(np.arange(1, 11), 20)
        labels = np.tile([1] + [0] * 9, 20)
        split = np.array([0, 1, 0, 2] * 5)
        offsets = np.arange(21) * 10
        with fixture(candidates, labels, offsets, split) as first:
            first.ensure_training_cache(6)
            expected = np.array(first.pool)
        changed = labels.copy().reshape(20, 10)
        changed[split != 0] = 9  # Must not inspect even invalid held-out labels.
        with fixture(candidates, changed.ravel(), offsets, split) as second:
            second.ensure_training_cache(6)
            np.testing.assert_array_equal(second.pool, expected)
            close_cache(second)
            second.ensure_training_cache(6)
            np.testing.assert_array_equal(second.pool, expected)

    def test_uniform_long_and_short_sampling(self):
        count = 1500
        with fixture(np.tile([1, 2, 3, 4, 5, 6], count), np.tile([1, 0, 0, 0, 0, 0], count),
                     np.arange(count + 1) * 6, np.zeros(count)) as store:
            store.ensure_training_cache(4)
            self.assertTrue(all(len(set(row)) == 4 for row in store.pool))
            observed = np.bincount(store.pool.ravel(), minlength=7)[2:7]
            self.assertTrue(np.all(np.abs(observed - count * 4 / 5) < 110), observed)
        with fixture(np.tile([1, 2, 3], count), np.tile([1, 0, 0], count),
                     np.arange(count + 1) * 3, np.zeros(count)) as store:
            store.ensure_training_cache(4)
            observed = np.bincount(store.pool.ravel(), minlength=4)[2:4]
            self.assertTrue(np.all(np.abs(observed - count * 2) < 200), observed)

    def test_sampling_across_exposure_chunk_boundary(self):
        count, width = 20001, 21
        with fixture(np.tile(np.arange(1, width + 1), count), np.tile([1] + [0] * (width - 1), count),
                     np.arange(count + 1) * width, np.zeros(count, dtype=np.uint8)) as store:
            store.ensure_training_cache(16)
            marker = json.loads((store.path / "training_cache.json").read_text())
            self.assertGreaterEqual(marker["sampling_chunks"], 2)
            np.testing.assert_array_equal(store.positive_rows, np.arange(count))
            np.testing.assert_array_equal(store.positive_items, 1)
            self.assertTrue(np.all((store.pool >= 2) & (store.pool <= width)))
            self.assertTrue(np.all(np.diff(np.sort(store.pool, axis=1), axis=1) > 0))

    def test_signature_rejects_feature_manifest_and_pool_changes(self):
        with fixture([1, 2], [1, 0], [0, 2], [0]) as store:
            store.ensure_training_cache(4)
            close_cache(store)
            with self.assertRaisesRegex(ValueError, "does not match"):
                store.ensure_training_cache(8)
            original = (store.path / "manifest.json").read_bytes()
            (store.path / "manifest.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "does not match"):
                store.ensure_training_cache(4)
            (store.path / "manifest.json").write_bytes(original)
            np.save(store.path / "news_features.npy", store.features + 1)
            with self.assertRaisesRegex(ValueError, "does not match"):
                store.ensure_training_cache(4)

    def test_invalid_pool_size_and_training_ids_rejected(self):
        with fixture([1, 2], [1, 0], [0, 2], [0]) as store:
            for size in (0, -1, True):
                with self.assertRaises(ValueError):
                    store.ensure_training_cache(size)
        with fixture([1, 0], [1, 0], [0, 2], [0]) as store:
            with self.assertRaisesRegex(ValueError, "non-PAD"):
                store.ensure_training_cache(4)
            self.assertFalse((store.path / "training_cache.json").exists())


def legacy_cache_build(store, pool_size=16):
    """Former scalar sampler, including hashes, I/O, history means and loads."""
    def hashes():
        return {name: hashlib.sha256((store.path / name).read_bytes()).hexdigest()
                for name in ("manifest.json", "news_features.npy")}
    signature = hashes()
    pool = open_memmap(store.path / "negative_pool.npy", mode="w+", dtype=np.int32,
                       shape=(len(store.split), pool_size))
    pool[:] = 0
    rows, items = [], []
    rng = np.random.default_rng(20261010)
    for row in np.flatnonzero(store.split == 0):
        candidates, labels = store.candidates_for(row)
        positives = candidates[labels == 1]
        negatives = candidates[(labels == 0) & ~np.isin(candidates, positives)]
        if not len(negatives) or not len(positives):
            continue
        pool[row] = rng.choice(negatives, pool_size, replace=len(negatives) < pool_size)
        rows.extend([int(row)] * len(positives))
        items.extend(positives.tolist())
    pool.flush()
    pool._mmap.close()
    np.save(store.path / "positive_rows.npy", np.asarray(rows, dtype=np.int32))
    np.save(store.path / "positive_items.npy", np.asarray(items, dtype=np.int32))
    mean = open_memmap(store.path / "mean_history.npy", mode="w+", dtype=np.float32,
                       shape=(len(store.split), store.features.shape[1]))
    for start in range(0, len(store.split), 8192):
        features = store.features[store.history[start:start + 8192]]
        valid = np.any(features != 0, axis=-1)
        mean[start:start + 8192] = features.sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)
    mean.flush()
    mean._mmap.close()
    (store.path / "training_cache.json").write_text(json.dumps(signature))
    assert hashes() == signature
    for name, filename in (("pool", "negative_pool"), ("positive_rows", "positive_rows"),
                           ("positive_items", "positive_items"), ("mean_history", "mean_history")):
        setattr(store, name, np.load(store.path / (filename + ".npy"), mmap_mode="r"))


def benchmark_cache(rows=20000, width=50, repeats=3, output="reports/sampling_cache_benchmark.json"):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    rng = np.random.default_rng(622)
    candidates = rng.integers(1, 2048, size=(rows, width), dtype=np.int32)
    candidates[::13, 1] = candidates[::13, 0]
    labels = np.zeros((rows, width), dtype=np.int8)
    labels[:, 0] = 1
    labels[::5, -2:] = 1
    labels[::41] = 0
    labels[::43] = 1
    split = np.zeros(rows, dtype=np.uint8)
    split[::5], split[::11] = 1, 2
    history = rng.integers(0, 2048, size=(rows, 5), dtype=np.int32)
    features = rng.normal(size=(2048, 32)).astype(np.float32)
    features[0] = 0
    timings = {"reference": [], "vectorized": []}
    expected = None
    for repeat in range(repeats):
        names = ("reference", "vectorized") if repeat % 2 == 0 else ("vectorized", "reference")
        for name in names:
            with fixture(candidates.ravel(), labels.ravel(), np.arange(rows + 1) * width,
                         split, history, features) as store:
                started = time.perf_counter()
                (legacy_cache_build if name == "reference" else Store.ensure_training_cache)(store, 16)
                timings[name].append(time.perf_counter() - started)
                actual = [np.array(store.positive_rows), np.array(store.positive_items), np.array(store.mean_history)]
                if expected is None:
                    expected = actual
                else:
                    for left, right in zip(actual, expected):
                        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-6)
                assert_cache_contract(unittest.TestCase(), store, 16)
    report = {"fixture": "synthetic sampling/IO benchmark, not MIND model or serving latency",
              "rows": rows, "candidate_exposures": rows * width, "train_rows": int((split == 0).sum()),
              "pool_size": 16, "repeats": repeats, "threads": 1,
              "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
              "scope": "input hashes, negative sampling, positive indices, writes, mean history, memmap loads; fixture creation and verification excluded",
              "reference": "former per-impression np.isin + Generator.choice retained in tests/test_sampling.py",
              "timings_seconds": timings,
              "reference_median_seconds": statistics.median(timings["reference"]),
              "vectorized_median_seconds": statistics.median(timings["vectorized"]),
              "speedup": statistics.median(timings["reference"]) / statistics.median(timings["vectorized"]),
              "same_positive_examples_and_mean_history": True,
              "both_pass_eligibility_and_replacement_contracts": True,
              "rng_note": "Fixed shared seed per implementation; distributions match, individual draws need not",
              "limitations": ["Synthetic fixture; full-MIND wall time must be measured separately.",
                              "Shared desktop background load and filesystem caching are uncontrolled."]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--benchmark", action="store_true")
        parser.add_argument("--rows", type=int, default=20000)
        parser.add_argument("--width", type=int, default=50)
        parser.add_argument("--repeats", type=int, default=3)
        parser.add_argument("--output", default="reports/sampling_cache_benchmark.json")
        args = parser.parse_args()
        if args.rows < 1 or args.width < 3 or args.repeats < 1:
            parser.error("rows/repeats must be positive and width >=3")
        benchmark_cache(args.rows, args.width, args.repeats, args.output)
    else:
        unittest.main()
