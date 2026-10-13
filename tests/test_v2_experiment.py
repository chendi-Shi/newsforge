"""Statistical tests use synthetic counts, never tune on dev labels."""
import unittest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tests.support import workspace_temporary_directory


class BootstrapTests(unittest.TestCase):
    def test_internal_selection_labels_mature_before_first_dev_cutoff(self):
        from newsforge.v2.experiment import sample_rows, MIN_VALID_TIME, MAX_VALID_TIME
        store = SimpleNamespace(split=np.array([1, 1, 1, 1, 2]), timestamps=np.array([
            MIN_VALID_TIME - 1, MIN_VALID_TIME, MAX_VALID_TIME - 1, MAX_VALID_TIME,
            MAX_VALID_TIME + 3600]))
        np.testing.assert_array_equal(sample_rows(store, 1, 0), [1, 2])
        np.testing.assert_array_equal(sample_rows(store, 1, 100), [1, 2])
        np.testing.assert_array_equal(sample_rows(store, 2, 0), [4])

    def test_zero_and_constant_ratio(self):
        from newsforge.v2.experiment import recall_interval
        before = np.array([0, 1, 0, 2])
        den = np.array([2, 4, 2, 4])
        users = np.array([1, 1, 2, 3])
        zero = recall_interval(before, before, den, users, repeats=100)
        self.assertEqual(zero["delta"], 0)
        self.assertEqual(zero["ci95"], [0., 0.])
        result = recall_interval(before, before + den / 2, den, users, repeats=100)
        self.assertEqual(result["delta"], .5)
        self.assertEqual(result["ci95"], [.5, .5])

    def test_ratio_is_micro_not_query_mean(self):
        from newsforge.v2.experiment import recall_interval
        result = recall_interval([0, 0], [1, 0], [1, 9], [1, 2], repeats=100)
        self.assertAlmostEqual(result["delta"], .1)

    def test_invalid_shape_and_denominator(self):
        from newsforge.v2.experiment import recall_interval
        with self.assertRaises(ValueError):
            recall_interval([0], [0], [0], [1])
        with self.assertRaises(ValueError):
            recall_interval([0, 1], [0], [1], [1])


class FrozenSelectionTests(unittest.TestCase):
    def test_selected_model_rejects_a_self_consistent_replacement(self):
        from newsforge.v2.experiment import load_selected_model, sha256_file
        with workspace_temporary_directory() as directory:
            path = Path(directory) / "model.joblib"
            path.write_bytes(b"internally selected model")
            entry = {"model_path": str(path), "model_sha256": sha256_file(path),
                     "variant": "full", "seed": 17}
            model = SimpleNamespace(model_version=entry["model_sha256"], variant="full", metadata={"seed": 17})
            with patch("newsforge.v2.experiment.CausalRanker.load", return_value=model) as loader:
                self.assertIs(load_selected_model(entry, None, None), model)
                loader.assert_called_once()
            # Even if a replacement has a valid matching sidecar, its bytes must
            # match the SHA fixed in the earlier internal selection document.
            path.write_bytes(b"a different internally valid model")
            with patch("newsforge.v2.experiment.CausalRanker.load", return_value=model) as loader:
                with self.assertRaisesRegex(ValueError, "Selected ranker model SHA"):
                    load_selected_model(entry, None, None)
                loader.assert_not_called()

    def test_loaded_model_must_match_selected_variant_and_seed(self):
        from newsforge.v2.experiment import load_selected_model
        entry = {"model_path": "selected.joblib", "model_sha256": "frozen-hash", "variant": "full", "seed": 17}
        for variant, seed, version in (("content_only", 17, "frozen-hash"),
                                       ("full", 29, "frozen-hash"), ("full", 17, "other-hash")):
            model = SimpleNamespace(model_version=version, variant=variant, metadata={"seed": seed})
            with self.subTest(variant=variant, seed=seed, version=version), \
                    patch("newsforge.v2.experiment.sha256_file", return_value="frozen-hash"), \
                    patch("newsforge.v2.experiment.CausalRanker.load", return_value=model):
                with self.assertRaisesRegex(ValueError, "identity"):
                    load_selected_model(entry, None, None)

    def test_bindings_cover_population_retrieval_and_model_sidecars(self):
        from newsforge.v2.experiment import bindings
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            selection_path = root / "ranker-selection.json"
            args = SimpleNamespace(data=str(root / "data"), temporal=str(root / "temporal"),
                                   reports=str(root / "reports"),
                                   ranker_selection=str(selection_path))
            changed = {}
            def digest(path):
                return hashlib.sha256((str(Path(path)) + changed.get(str(Path(path)), "original")).encode()).hexdigest()
            model_path = root / "selected.joblib"
            entry = {"model_path": str(model_path), "model_sha256": digest(model_path),
                     "variant": "full", "seed": 17}
            selected = {"models": [entry], "selected": entry, "best_by_variant": {"full": entry}}
            selection_path.write_text(json.dumps(selected), encoding="utf-8")
            with patch("newsforge.v2.experiment.sha256_file", side_effect=digest):
                original = bindings(args, None)
                required_data = ("all_history.npy", "all_users.npy", "all_split.npy", "all_labels.npy",
                                 "news_features.npy", "news_first_seen.npy", "news_train_clicks.npy",
                                 "news_train_exposures.npy", "news_ids.json", "negative_pool.npy",
                                 "positive_rows.npy", "positive_items.npy")
                for name in required_data:
                    self.assertIn(name, original["processed_files"])
                watched = [Path(args.data) / name for name in required_data] + [
                    Path("artifacts/article_embeddings.npy"), Path("artifacts/retriever.pt"),
                    Path("artifacts/v2/user_encoder.npz"), Path("artifacts/v2/user_encoder.npz.json"),
                    Path(str(model_path) + ".manifest.json")]
                for path in watched:
                    with self.subTest(path=str(path)):
                        changed[str(path)] = "changed"
                        self.assertNotEqual(bindings(args, None), original)
                        changed.clear()
                changed[str(model_path)] = "replacement"
                with self.assertRaisesRegex(ValueError, "Selected ranker model SHA"):
                    bindings(args, None)
                changed.clear()
                selected["selected"] = {**entry, "model_sha256": "a different frozen model"}
                selection_path.write_text(json.dumps(selected), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    bindings(args, None)


if __name__ == "__main__":
    unittest.main()
