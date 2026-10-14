"""Small invented numerical states only; no fit, transform or real data read."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_preprocessing.py"
SPEC = importlib.util.spec_from_file_location("compare_preprocessing_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(root, perturb=False, mismatch=False):
    root.mkdir()
    features = np.array([[0, 0, 0, 0], [1, 0, 1, 0], [.6, .8, 0, 1]], dtype=np.float32)
    if perturb:
        features[1, 1] = 1e-5
    np.save(root / "news_features.npy", features, allow_pickle=False)
    (root / "news_ids.json").write_text(json.dumps(["pad", "private_fixture_article_b" if mismatch else "private_fixture_article_a", "private_fixture_article_c"]))
    (root / "category_vocabulary.json").write_text('["unknown", "category"]')
    (root / "manifest.json").write_text('{"features":{"text_dimensions":2,"dimensions":4}}')
    vectorizer = TfidfVectorizer(dtype=np.float32)
    vectorizer.vocabulary_ = {"private_fixture_token_one": 0, "private_fixture_token_two": 1}
    vectorizer._tfidf = TfidfTransformer()
    vectorizer._tfidf.idf_ = np.array([1., 2.], dtype=np.float32)
    joblib.dump(vectorizer, root / "tfidf.joblib")
    svd = TruncatedSVD(n_components=2, random_state=2026)
    svd.components_ = np.eye(2, dtype=np.float32)
    svd.singular_values_ = np.array([2., 1.], dtype=np.float32)
    svd.explained_variance_ = np.array([.4, .3], dtype=np.float32)
    svd.explained_variance_ratio_ = np.array([.4, .3], dtype=np.float32)
    joblib.dump(svd, root / "svd.joblib")


class ComparePreprocessingTests(unittest.TestCase):
    def test_exact_equal_hash_binding_privacy_and_no_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "old"); fixture(root / "new")
            with patch.object(TfidfVectorizer, "fit", side_effect=AssertionError("no fit")), \
                 patch.object(TfidfVectorizer, "transform", side_effect=AssertionError("no transform")), \
                 patch.object(TruncatedSVD, "fit", side_effect=AssertionError("no fit")), \
                 patch.object(TruncatedSVD, "transform", side_effect=AssertionError("no transform")):
                result = MODULE.compare(root / "old", root / "new", root / "comparison.json")
            self.assertTrue(result["numerical_comparison"]["all_features"]["exact_equal"])
            self.assertTrue(result["tfidf"]["vocabulary_equal"])
            self.assertTrue(result["svd"]["components"]["exact_equal"])
            self.assertEqual(result["script_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
            self.assertEqual(result["input_sha256"]["feature_array"]["old_sha256"], hashlib.sha256((root / "old/news_features.npy").read_bytes()).hexdigest())
            output = (root / "comparison.json").read_text()
            self.assertNotIn("private_fixture", output)
            self.assertNotIn(str(root), output)
            self.assertFalse(result["analysis_environment_only"]["historical_run_environment"])

    def test_small_text_perturbation_is_retained_and_categories_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "old"); fixture(root / "new", perturb=True)
            result = MODULE.compare(root / "old", root / "new", root / "comparison.json")
            numerical = result["numerical_comparison"]
            self.assertFalse(numerical["text"]["exact_equal"])
            self.assertAlmostEqual(numerical["text"]["max_absolute_difference"], 1e-5, places=10)
            self.assertTrue(numerical["category"]["exact_equal"])
            self.assertEqual(numerical["row_geometry"]["text_rows_with_any_difference"], 1)

    def test_mismatched_index_refuses_comparison_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "old"); fixture(root / "new", mismatch=True)
            with patch.object(joblib, "load", side_effect=AssertionError("must validate identity first")):
                with self.assertRaisesRegex(ValueError, "mapping differs"):
                    MODULE.compare(root / "old", root / "new", root / "comparison.json")
            self.assertFalse((root / "comparison.json").exists())

    def test_existing_output_is_preserved_before_any_input_access(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "comparison.json"
            output.write_text("preserve existing evidence")
            with self.assertRaises(FileExistsError):
                MODULE.compare(Path(folder) / "missing-old", Path(folder) / "missing-new", output)
            self.assertEqual(output.read_text(), "preserve existing evidence")


if __name__ == "__main__":
    unittest.main()
