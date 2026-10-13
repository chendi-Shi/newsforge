"""Synthetic integration: matched warmup and locked-test/resume behaviour."""
import gc
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

from newsforge.train import main


class TrainingIntegrationTest(unittest.TestCase):
    def test_matched_warmup_full_epochs_and_test_lock(self):
        root = Path(__file__).resolve().parent / ("training-test-" + uuid.uuid4().hex)
        root.mkdir()
        self.addCleanup(shutil.rmtree, root)
        data, artifacts, reports = root / "data", root / "artifacts", root / "reports"
        data.mkdir()
        rng = np.random.default_rng(5)
        features = rng.normal(size=(10, 6)).astype(np.float32)
        features[0] = 0
        arrays = {
            "news_features": features,
            "all_history": np.tile([[1, 2, 0], [0, 0, 0], [3, 4, 5]], (8, 1)).astype(np.int32),
            "all_candidates": np.tile([6, 7, 8], 24).astype(np.int32),
            "all_labels": np.tile([1, 0, 0], 24).astype(np.uint8),
            "all_offsets": np.arange(0, 73, 3, dtype=np.int64),
            "all_timestamps": np.repeat([10, 20, 30], [12, 6, 6]).astype(np.int64),
            "all_split": np.repeat([0, 1, 2], [12, 6, 6]).astype(np.uint8),
            "all_users": np.arange(24, dtype=np.int32) // 2,
            "news_first_seen": np.array([2**63-1] + [10] * 9, dtype=np.int64),
            "news_train_clicks": np.array([0] * 6 + [12, 0, 0, 0]),
            "news_train_exposures": np.array([0] * 6 + [12, 12, 12, 0]),
        }
        for name, array in arrays.items():
            np.save(data / (name + ".npy"), array)
        (data / "manifest.json").write_text(json.dumps({"train_cutoff_timestamp": 10}))
        (data / "news_ids.json").write_text(json.dumps(["PAD"] + [f"N{i}" for i in range(1, 10)]))
        arguments = ["train", "--data", str(data), "--artifacts", str(artifacts), "--output", str(reports),
                     "--variants", "mean_random", "mean_mixed", "--seeds", "17", "--epochs", "2",
                     "--batch-size", "8", "--threads", "1", "--validation-impressions", "6"]
        with patch.object(sys, "argv", arguments + ["--skip-test"]):
            main()
        self.assertFalse((artifacts / "evaluation_rows.npy").exists())
        checks = json.loads((reports / "matched_warmup_checks.json").read_text())
        self.assertTrue(checks["17"]["identical"])
        mixed = torch.load(artifacts / "mean_mixed_seed17.pt", weights_only=True)
        self.assertTrue(mixed["trained_on_all_eligible_clicks"])
        self.assertEqual([epoch["training_examples"] for epoch in mixed["history"]], [12, 12])
        self.assertEqual([epoch["negative_policy"] for epoch in mixed["history"]], ["four_random", "two_hard_two_random"])
        with patch.object(sys, "argv", arguments + ["--resume"]):
            main()
        np.testing.assert_array_equal(np.load(artifacts / "evaluation_rows.npy"), np.arange(18, 24))
        result = json.loads((reports / "ranking_experiment.json").read_text())
        self.assertEqual(result["runs"]["mean_random-seed17"]["impressions"], 6)
        self.assertIn("mean_mixed", result["paired_user_bootstrap"])
        # A changed fit budget must not silently reuse an old checkpoint.
        changed = list(arguments)
        changed[changed.index("--epochs") + 1] = "3"
        with patch.object(sys, "argv", changed + ["--resume", "--skip-test"]):
            with self.assertRaisesRegex(ValueError, "Resume configuration/data mismatch"):
                main()
        gc.collect()


if __name__ == "__main__":
    unittest.main()
