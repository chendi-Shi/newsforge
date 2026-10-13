"""Tests for temporal isolation and the memmap data contract, using no real data."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from newsforge.data import MAX_TIME, download_dataset, prepare, time_split
from tests.support import workspace_temporary_directory


class DownloadIntegrityTests(unittest.TestCase):
    def test_same_size_corrupt_archives_are_rejected_before_extraction(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            for source in ("train", "dev"):
                (root / f"MINDlarge_{source}.zip").write_bytes(b"corrupt")
            with patch("newsforge.data.EXPECTED_BYTES", {"train": 7, "dev": 7}), \
                 patch("newsforge.data.EXPECTED_SHA256", {"train": "0" * 64, "dev": "0" * 64}):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    download_dataset(root, workers=1)
            self.assertFalse((root / "train").exists())
            self.assertFalse((root / "dev").exists())


class TemporalSplitTests(unittest.TestCase):
    def test_timestamp_ties_and_locked_official_split(self):
        stamps = np.asarray([10, 30, 20, 30, 40, 50, 60, 70, 80, 80, 90, 100])
        split, cutoff = time_split(stamps, 11, fraction=.7)
        self.assertEqual(cutoff, 70)
        self.assertTrue(np.all(split[stamps == 80] == 1))
        self.assertEqual(split[-1], 2)
        self.assertLess(stamps[split == 0].max(), stamps[split == 1].min())


def write_fixture(root):
    for source in ("train", "dev"):
        (root / source).mkdir(parents=True)
        lines = [f"N{i}\tsports\tball\tcommon title token{i}\tshared story news\turl\t[]\t[]\n"
                 for i in range(1, 24)]
        lines += ["N99\tFUTURE_CATEGORY\tx\tuniqueheldoutsecret\tsecretfutureword\turl\t[]\t[]\n",
                  "N100\tx\tx\tunobservedsecret\textra\turl\t[]\t[]\n"]
        (root / source / "news.tsv").write_text("".join(lines), encoding="utf-8")
    train = []
    for i in range(12):
        history = " ".join(f"N{j}" for j in range(1, 24)) if i == 0 else "N1 N2"
        candidate = "N99-1 N2-0" if i == 11 else "N1-1 N2-0"
        hour = i if i else 12
        train.append(f"{i+1}\tU{i%3+1}\t11/09/2019 {hour:02d}:00:00 AM\t{history}\t{candidate}\n")
    (root / "train" / "behaviors.tsv").write_text("".join(train), encoding="utf-8")
    dev = ["1\tU4\t11/10/2019 01:00:00 AM\t\tN99-1 N2-0\n",
           "2\tU1\t11/10/2019 02:00:00 AM\tN1\tN2-1 N99-0\n"]
    (root / "dev" / "behaviors.tsv").write_text("".join(dev), encoding="utf-8")


class PipelineTests(unittest.TestCase):
    def test_contract_and_heldout_text_isolation(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory); raw = root / "raw"; output = root / "processed"
            write_fixture(raw)
            manifest = prepare(raw, output, components=4, max_features=1000, min_df=1, threads=1)
            ids = json.loads((output / "news_ids.json").read_text())
            index = {v: i for i, v in enumerate(ids)}
            self.assertEqual(ids[0], "[PAD]")
            history = np.load(output / "all_history.npy")
            self.assertEqual(history.dtype, np.int32)
            self.assertEqual(history.shape, (14, 20))
            self.assertEqual(history[0].tolist(), [index[f"N{i}"] for i in range(4, 24)])
            self.assertTrue(np.all(history[12] == 0))
            self.assertEqual(history[13, 0], index["N1"])
            self.assertTrue(np.all(history[13, 1:] == 0))
            offsets = np.load(output / "all_offsets.npy")
            candidates = np.load(output / "all_candidates.npy")
            labels = np.load(output / "all_labels.npy")
            split = np.load(output / "all_split.npy")
            self.assertEqual(offsets.dtype, np.int64)
            self.assertEqual(candidates.dtype, np.int32)
            self.assertEqual(labels.dtype, np.uint8)
            np.testing.assert_array_equal(offsets, np.arange(0, 29, 2))
            self.assertEqual(int(offsets[-1]), len(candidates))
            np.testing.assert_array_equal(split[-2:], [2, 2])
            original_ids = np.load(output / "all_impression_ids.npy")
            self.assertEqual(original_ids[0], original_ids[12])
            features = np.load(output / "news_features.npy")
            self.assertEqual(features.dtype, np.float32)
            self.assertTrue(np.all(features[0] == 0))
            vectorizer = joblib.load(output / "tfidf.joblib")
            self.assertNotIn("uniqueheldoutsecret", vectorizer.vocabulary_)
            self.assertNotIn("secretfutureword", vectorizer.vocabulary_)
            observed = np.load(output / "news_train_observed.npy")
            self.assertFalse(observed[index["N99"]])
            self.assertFalse(observed[index["N100"]])
            first_seen = np.load(output / "news_first_seen.npy")
            self.assertEqual(first_seen[index["N100"]], MAX_TIME)
            self.assertEqual(first_seen[0], MAX_TIME)
            self.assertEqual(np.load(output / "news_train_clicks.npy")[index["N99"]], 0)
            self.assertEqual(np.load(output / "news_train_exposures.npy")[index["N99"]], 0)
            positive_rows = np.load(output / "train_positive_rows.npy")
            positive_items = np.load(output / "train_positive_items.npy")
            self.assertTrue(np.all(split[positive_rows] == 0))
            self.assertTrue(np.all(positive_items == index["N1"]))
            self.assertEqual(manifest["candidates"], 28)
            self.assertEqual(manifest["positives"], 14)
            self.assertIn("train_cutoff_timestamp", manifest)

    def test_heldout_labels_do_not_change_training_statistics_or_text(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory); raw = root / "raw"
            write_fixture(raw)
            first, second = root / "first", root / "second"
            prepare(raw, first, components=4, max_features=1000, min_df=1, threads=1)
            path = raw / "dev" / "behaviors.tsv"
            content = path.read_text().replace("N99-1 N2-0", "N99-0 N2-1")
            path.write_text(content, encoding="utf-8")
            prepare(raw, second, components=4, max_features=1000, min_df=1, threads=1)
            for name in ("news_features", "news_train_clicks", "news_train_exposures", "news_first_seen"):
                np.testing.assert_array_equal(np.load(first / f"{name}.npy"), np.load(second / f"{name}.npy"))


if __name__ == "__main__":
    unittest.main()
