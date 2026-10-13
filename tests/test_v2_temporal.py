"""Causal boundary and artifact-integrity checks on small synthetic logs only."""
import json
from pathlib import Path
import unittest

import numpy as np

from newsforge.v2.temporal import HOUR, TemporalStore, build_temporal
from tests.support import workspace_temporary_directory


def fixture(root, *, change_future=False):
    root.mkdir()
    # Deliberately not timestamp-ordered; duplicate timestamps are independent rows.
    times = np.array([0, 3599, 7200, 3600, 7199, 3600, 10800, 18000], dtype=np.int64)
    candidates = np.array([[1, 2], [1, 3], [3, 4], [2, 3], [1, 4], [2, 4], [2, 4], [1, 2]], dtype=np.int32)
    labels = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.uint8)
    if change_future:
        labels[times >= 3600] = 1 - labels[times >= 3600]
    train = times <= 3599
    clicks = np.bincount(candidates[train][labels[train] == 1], minlength=5).astype(np.int64)
    exposures = np.bincount(candidates[train].reshape(-1), minlength=5).astype(np.int64)
    arrays = {"all_timestamps": times, "all_offsets": np.arange(0, 17, 2, dtype=np.int64),
              "all_candidates": candidates.reshape(-1), "all_labels": labels.reshape(-1),
              "news_first_seen": np.array([np.iinfo(np.int64).max, 0, 0, 3599, 3600], dtype=np.int64),
              "news_train_clicks": clicks, "news_train_exposures": exposures}
    for name, array in arrays.items():
        np.save(root / f"{name}.npy", array)
    manifest = {"rows": len(times), "candidates": candidates.size, "positives": int(labels.sum()),
                "news_including_padding": 5, "train_cutoff_timestamp": 3599,
                "splits": {"0": {"positives": int(clicks.sum()), "candidates": int(exposures.sum())}}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TemporalTests(unittest.TestCase):
    def test_strict_cutoff_ties_hour_end_decay_and_readonly(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            fixture(root / "source")
            result = build_temporal(root / "source", root / "temporal", chunk_candidates=3)
            self.assertTrue(result["counts"]["histogram_counts_match_source"])
            self.assertEqual(result["counts"]["source_candidate_exposures"], 16)
            self.assertEqual(result["n_snapshots"], 6)
            with TemporalStore(root / "temporal", root / "source") as store:
                for when in (0, 3599, 3600, 7199):
                    self.assertEqual(float(store.snapshot(when).clicks.sum()), 0)
                earlier = store.snapshot(7200)
                self.assertEqual((earlier.index, earlier.as_of_start, earlier.cutoff), (2, 7200, 3600))
                np.testing.assert_array_equal(earlier.clicks, [0, 1, 0, 1, 0])
                np.testing.assert_array_equal(earlier.exposures, [0, 2, 1, 1, 0])
                # Both rows at exactly source_time=3600 are excluded above and enter together here.
                later = store.snapshot(10800)
                decay = 2 ** (-1 / 12)
                np.testing.assert_allclose(later.clicks, [0, decay + 1, 1, decay, 1], rtol=1e-7)
                np.testing.assert_allclose(later.exposures, [0, 2 * decay + 1, decay + 2, decay + 1, 2], rtol=1e-7)
                self.assertFalse(earlier.clicks.flags.writeable)
                with self.assertRaises(ValueError):
                    earlier.clicks[1] = 99
                np.testing.assert_allclose(store.age_hours(np.array([0, 1, 3, 4]), 7200),
                                           [0, 2, (7200 - 3599) / HOUR, 0])

    def test_future_label_perturbation_cannot_change_earlier_snapshot(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            for name, altered in (("original", False), ("changed", True)):
                fixture(root / name, change_future=altered)
                build_temporal(root / name, root / (name + "-time"), chunk_candidates=3)
            with TemporalStore(root / "original-time", root / "original") as original, \
                    TemporalStore(root / "changed-time", root / "changed") as changed:
                for when in (0, 3600, 7200):
                    np.testing.assert_array_equal(original.snapshot(when).clicks, changed.snapshot(when).clicks)
                    np.testing.assert_array_equal(original.snapshot(when).exposures, changed.snapshot(when).exposures)
                self.assertFalse(np.array_equal(original.snapshot(10800).clicks, changed.snapshot(10800).clicks))

    def test_snapshot_bounds_and_frozen_training_availability(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            fixture(root / "source")
            build_temporal(root / "source", root / "temporal", chunk_candidates=4)
            with TemporalStore(root / "temporal", root / "source") as store:
                for timestamp in (-1, 18001, 21600, 7200.0, True):
                    with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                        store.snapshot(timestamp)
                with self.assertRaisesRegex(ValueError, "not yet available"):
                    store.frozen_snapshot(3600)
                frozen = store.frozen_snapshot(18000)
                self.assertEqual(frozen.mode, "frozen")
                np.testing.assert_array_equal(frozen.clicks, [0, 1, 0, 1, 0])
                self.assertFalse(frozen.exposures.flags.writeable)

    def test_same_size_source_and_snapshot_corruption_are_rejected(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            fixture(root / "source")
            build_temporal(root / "source", root / "temporal")
            for path, message in ((root / "source" / "all_labels.npy", "source SHA256 mismatch"),
                                  (root / "temporal" / "snapshot_clicks.npy", "snapshot SHA256 mismatch")):
                with self.subTest(path=path.name):
                    with path.open("r+b") as stream:
                        stream.seek(-1, 2); previous = stream.read(1)
                        stream.seek(-1, 2); stream.write(bytes([previous[0] ^ 1]))
                    try:
                        with self.assertRaisesRegex(ValueError, message):
                            TemporalStore(root / "temporal", root / "source")
                    finally:
                        with path.open("r+b") as stream:
                            stream.seek(-1, 2); stream.write(previous)

    def test_budget_counts_and_no_overwrite(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            fixture(root / "source")
            with self.assertRaisesRegex(ValueError, "200000"):
                build_temporal(root / "source", root / "too-large", chunk_candidates=200001)
            build_temporal(root / "source", root / "temporal")
            with self.assertRaisesRegex(ValueError, "new directory"):
                build_temporal(root / "source", root / "temporal")
            path = root / "source" / "manifest.json"
            manifest = json.loads(path.read_text()); manifest["positives"] += 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "histogram counts"):
                build_temporal(root / "source", root / "inconsistent")


if __name__ == "__main__":
    unittest.main()
