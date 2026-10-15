"""Invented article titles only; the original Adressa data is never accessed."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from newsforge.adaptation import representation
from tests.support import workspace_temporary_directory


SMALL = representation.RepresentationConfig(n_components=2, batch_size=2)


def invented_database(future_title="zzzzzz uniquezzzz", *, path=":memory:"):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE articles(id INTEGER PRIMARY KEY,first_seen INTEGER,title TEXT,category TEXT)")
    records = [
        (1, 10, "alpha meadow blue", "one"),
        (2, 11, "alpha meadow green", "two"),
        (3, 12, "beta forest blue", "three"),
        (4, 13, "beta forest green", "one"),
        (5, 14, "", ""),
        (6, 20, future_title, "future"),
        (7, 21, future_title, "future"),
    ]
    # Deliberately insert backwards: model input order must come from ORDER BY.
    connection.executemany("INSERT INTO articles VALUES(?,?,?,?)", reversed(records))
    connection.commit()
    first_seen = np.array([-1] + [row[1] for row in records], dtype=np.int64)
    return connection, first_seen


class LocalRepresentationTests(unittest.TestCase):
    def test_formal_defaults_and_timestamp_are_frozen(self):
        self.assertEqual(representation.FIT_CUTOFF, int(datetime(2017, 1, 1, 23, tzinfo=timezone.utc).timestamp()))
        self.assertEqual(representation.DEFAULT_CONFIG.n_components, 96)
        self.assertEqual(representation.DEFAULT_CONFIG.category_dimensions, 19)
        self.assertEqual(representation.DEFAULT_CONFIG.batch_size, 4096)
        with self.assertRaises(FrozenInstanceError):
            representation.DEFAULT_CONFIG.n_components = 2
        vectorizer = representation._vectorizer()
        params = vectorizer.get_params()
        expected = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
                    "max_features": 40000, "sublinear_tf": True, "norm": "l2",
                    "dtype": np.float32, "stop_words": None}
        self.assertEqual({key: params[key] for key in expected}, expected)

    def test_strict_cutoff_excludes_future_vocabulary_and_idf(self):
        connection, first_seen = invented_database()
        self.addCleanup(connection.close)
        with workspace_temporary_directory() as directory:
            features, metadata = representation._fit_local_representation(connection, first_seen, directory, 20, SMALL)
            fitted = joblib.load(Path(directory) / "vectorizer.joblib")
            expected = representation._vectorizer()
            titles = [row[0] for row in connection.execute("SELECT title FROM articles WHERE first_seen < 20 ORDER BY id")]
            expected.fit(titles)
            self.assertEqual(fitted.vocabulary_, expected.vocabulary_)
            np.testing.assert_array_equal(fitted.idf_, expected.idf_)
            self.assertNotIn("zzz", fitted.vocabulary_)
            self.assertEqual(metadata["fit_docs"], 5)
            self.assertEqual(metadata["fit_ids_array_sha256"], hashlib.sha256(np.arange(1, 6, dtype="<i8").tobytes()).hexdigest())
            np.testing.assert_array_equal(features[6:, :2], 0)
            self.assertEqual(metadata["coverage"]["all"]["zero_tfidf"], 3)
            self.assertEqual(metadata["coverage"]["fit"]["zero_tfidf"], 1)

    def test_future_title_changes_cannot_change_earlier_vectors_or_fitted_models(self):
        first, first_seen = invented_database("zzzzzz uniquezzzz")
        second, second_seen = invented_database("alpha meadow alpha meadow blue")
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        with workspace_temporary_directory() as directory:
            paths = [Path(directory) / "first", Path(directory) / "second"]
            before, before_meta = representation._fit_local_representation(first, first_seen, paths[0], 20, SMALL)
            after, after_meta = representation._fit_local_representation(second, second_seen, paths[1], 20, SMALL)
            np.testing.assert_array_equal(before[:6], after[:6])
            self.assertFalse(np.array_equal(before[6:, :2], after[6:, :2]))
            for name in ("vectorizer.joblib", "svd.joblib"):
                self.assertEqual((paths[0] / name).read_bytes(), (paths[1] / name).read_bytes())
            for key in ("fit_ids_array_sha256", "fit_source_sha256"):
                self.assertEqual(before_meta[key], after_meta[key])
            self.assertNotEqual(before_meta["source_sha256"], after_meta["source_sha256"])
            self.assertEqual(before_meta["coverage"]["fit"], after_meta["coverage"]["fit"])

    def test_padding_empty_title_and_unknown_category_are_finite(self):
        connection, first_seen = invented_database()
        self.addCleanup(connection.close)
        with workspace_temporary_directory() as directory:
            features, metadata = representation._fit_local_representation(connection, first_seen, directory, 20, SMALL)
            self.assertEqual(features.shape, (8, 21))
            self.assertEqual(features.dtype, np.dtype("float32"))
            self.assertTrue(np.isfinite(features).all())
            np.testing.assert_array_equal(features[0], 0)
            np.testing.assert_array_equal(features[1:, 2], 1)
            np.testing.assert_array_equal(features[:, 3:], 0)
            np.testing.assert_array_equal(features[5, :2], 0)
            np.testing.assert_allclose(np.linalg.norm(features[1:5, :2], axis=1), 1, atol=2e-7)
            np.testing.assert_array_equal(np.load(Path(directory) / "features.npy", allow_pickle=False), features)
            flags = np.load(Path(directory) / "coverage.npy", allow_pickle=False)
            self.assertEqual(flags.dtype, np.dtype("uint8"))
            self.assertEqual(flags.shape, (8, 4))
            np.testing.assert_array_equal(flags[0], 0)
            np.testing.assert_array_equal(flags[5], [1, 1, 1, 1])
            np.testing.assert_array_equal(flags[6:], [[0, 1, 1, 1], [0, 1, 1, 1]])
            for index, key in enumerate(metadata["coverage_columns"]):
                self.assertEqual(int(flags[1:, index].sum()), metadata["coverage"]["all"][key])
            self.assertEqual(metadata["category_unknown_column"], 2)
            self.assertEqual(metadata["coverage"]["all"]["unknown_category"], 7)
            serialized = json.dumps(metadata, allow_nan=False)
            for source_text in ("alpha meadow blue", "zzzzzz uniquezzzz", '"zzz"'):
                self.assertNotIn(source_text, serialized)
            self.assertEqual(set(Path(directory).iterdir()), {Path(directory) / name for name in representation._ARTIFACT_NAMES})

    def test_existing_artifact_is_never_overwritten_or_fitted_over(self):
        connection, first_seen = invented_database()
        self.addCleanup(connection.close)
        with workspace_temporary_directory() as directory:
            for name in representation._ARTIFACT_NAMES:
                with self.subTest(name=name):
                    folder = Path(directory) / name.replace(".", "_")
                    folder.mkdir()
                    existing = folder / name
                    existing.write_bytes(b"immutable evidence")
                    with patch.object(representation, "_vectorizer") as build:
                        with self.assertRaises(FileExistsError):
                            representation._fit_local_representation(connection, first_seen, folder, 20, SMALL)
                        build.assert_not_called()
                    self.assertEqual(existing.read_bytes(), b"immutable evidence")
                    self.assertEqual(list(folder.iterdir()), [existing])

    def test_too_few_fit_documents_fail_without_lowering_dimensions(self):
        connection, first_seen = invented_database()
        self.addCleanup(connection.close)
        with workspace_temporary_directory() as directory:
            with self.assertRaisesRegex(ValueError, "at least 97"):
                representation.fit_local_representation(connection, first_seen, directory, 20)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_public_api_keeps_full_115_dimensions_on_synthetic_titles(self):
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        connection.execute("CREATE TABLE articles(id INTEGER PRIMARY KEY,first_seen INTEGER,title TEXT)")
        records = [(index, index, f"invented article {index % 50:03d} meadow forest blue green")
                   for index in range(1, 101)]
        connection.executemany("INSERT INTO articles VALUES(?,?,?)", records)
        first_seen = np.array([-1] + list(range(1, 101)), dtype=np.int64)
        with workspace_temporary_directory() as directory:
            features, metadata = representation.fit_local_representation(connection, first_seen, directory)
            self.assertEqual(features.shape, (101, 115))
            self.assertEqual(metadata["text_dimensions"], 96)
            self.assertEqual(metadata["category_unknown_column"], 96)
            self.assertEqual(metadata["params"]["batch_size"], 4096)
            np.testing.assert_array_equal(features[0], 0)
            np.testing.assert_array_equal(features[1:, 97:], 0)
            np.testing.assert_array_equal(features[1:, 96], 1)
            np.testing.assert_allclose(np.linalg.norm(features[1:, :96], axis=1), 1, atol=2e-7)
            self.assertTrue(np.isfinite(features).all())

    def test_too_few_vocabulary_features_fail_without_relaxing_min_df(self):
        connection, first_seen = invented_database()
        self.addCleanup(connection.close)
        connection.execute("UPDATE articles SET title='a' WHERE first_seen < 20")
        with workspace_temporary_directory() as directory:
            with self.assertRaisesRegex(ValueError, "insufficient fit vocabulary"):
                representation._fit_local_representation(connection, first_seen, directory, 20, SMALL)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_first_seen_mismatch_and_missing_article_are_rejected(self):
        connection, first_seen = invented_database()
        self.addCleanup(connection.close)
        with workspace_temporary_directory() as directory:
            mismatched = first_seen.copy()
            mismatched[6] -= 1
            with self.assertRaisesRegex(ValueError, "first_seen does not match"):
                representation._fit_local_representation(connection, mismatched, directory, 20, SMALL)
            connection.execute("DELETE FROM articles WHERE id=6")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                representation._fit_local_representation(connection, first_seen, directory, 20, SMALL)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_read_only_source_database_is_unchanged(self):
        with workspace_temporary_directory() as directory:
            database = Path(directory) / "invented.sqlite"
            writable, first_seen = invented_database(path=database)
            writable.close()
            before = database.read_bytes()
            readonly = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
            try:
                features, _ = representation._fit_local_representation(readonly, first_seen, Path(directory) / "output", 20, SMALL)
            finally:
                readonly.close()
            self.assertEqual(database.read_bytes(), before)
            self.assertEqual(features.shape, (8, 21))


if __name__ == "__main__":
    unittest.main()
