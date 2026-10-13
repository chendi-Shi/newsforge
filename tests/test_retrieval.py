"""Retrieval correctness and artifact-integrity tests; no timing assertions."""

import json
from pathlib import Path
import unittest

import numpy as np

from newsforge.retrieval import CatalogIndex, UNKNOWN_TIME
from tests.support import workspace_temporary_directory


def catalogue(kind="exact"):
    embeddings = np.array([[1, 0], [.99, .1], [.9, .3], [.8, .5], [.95, .05],
                           [.7, .7], [.6, .8], [0, 0], [np.nan, 1]], dtype=np.float32)
    ids = ["PAD", "N1", "N2", "N3", "FUTURE", "UNKNOWN", "OOV", "ZERO", "NAN"]
    timestamps = np.array([0, 10, 10, 15, 30, UNKNOWN_TIME, 0, 0, 0], dtype=np.int64)
    clicks = np.array([1000, 3, 8, 5, 999, 999, 999, 999, 999], dtype=np.int64)
    return CatalogIndex.build(embeddings, ids, timestamps, clicks,
                               default_as_of=20, kind=kind, threads=1, hnsw_m=8,
                               ef_search=64, model_version="test-model")


class RetrievalTests(unittest.TestCase):
    def test_exact_matches_numpy_dot_products(self):
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(301, 16)).astype(np.float32)
        matrix[0] = 0
        index = CatalogIndex.build(matrix, ["PAD"] + [f"N{i}" for i in range(1, 301)],
                                   np.full(301, 10), np.zeros(301, dtype=np.int64),
                                   default_as_of=10, kind="exact")
        query = rng.normal(size=16).astype(np.float32)
        result = index.search(query, 20, seen={2, 4, 6})
        normalized = matrix[1:] / np.linalg.norm(matrix[1:], axis=1, keepdims=True)
        expected_scores = normalized @ (query / np.linalg.norm(query))
        order = np.argsort(-expected_scores)
        order = [i for i in order if int(i + 1) not in {2, 4, 6}][:20]
        self.assertEqual(result.indices, [int(i + 1) for i in order])
        np.testing.assert_allclose(result.scores, expected_scores[order], atol=2e-6, rtol=2e-6)

    def test_timestamp_seen_pad_oov_unknown_and_bad_vectors_filtered(self):
        for kind in ("exact", "hnsw"):
            with self.subTest(kind=kind):
                index = catalogue(kind)
                result = index.search(np.array([1, 0]), 10, seen={1})
                self.assertEqual(set(result.indices), {2, 3})
                self.assertEqual(result.shortfall, 8)
                self.assertTrue(result.exhausted)
                self.assertEqual(set(index.indexed_ids), {1, 2, 3, 4})

    def test_explicit_cutoff_can_expand_or_shrink_observed_catalogue(self):
        index = catalogue()
        self.assertEqual(index.search([1, 0], 10, as_of=5).indices, [])
        self.assertEqual(set(index.search([1, 0], 10, as_of=30).indices), {1, 2, 3, 4})
        self.assertEqual(index.default_as_of, 20)

    def test_popularity_uses_training_counts_and_same_filters(self):
        index = catalogue()
        result = index.popular(10, seen={2})
        self.assertEqual(result.indices, [3, 1])
        np.testing.assert_allclose(result.scores, np.log1p([5, 3]))
        self.assertEqual(index.popular(10, as_of=1).indices, [])

    def test_adaptive_overfetch_and_explicit_cap_do_not_fill_future_items(self):
        angles = np.linspace(0.001, 1.5, 150, dtype=np.float32)
        matrix = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        matrix[0] = 0
        index = CatalogIndex.build(matrix, ["PAD"] + [f"N{i}" for i in range(1, 150)],
                                   np.full(150, 10), np.zeros(150, dtype=np.int64),
                                   default_as_of=10, kind="exact")
        result = index.search([1, 0], 5, seen=set(range(1, 121)))
        self.assertEqual(result.indices, list(range(121, 126)))
        self.assertGreaterEqual(result.examined, 125)
        capped = index.search([1, 0], 5, seen=set(range(1, 121)), max_overfetch=32)
        self.assertEqual(capped.indices, [])
        self.assertEqual(capped.shortfall, 5)
        self.assertFalse(capped.exhausted)

    def test_persistence_hnsw_and_exact_keep_ids_scores_and_versions(self):
        for kind in ("exact", "hnsw"):
            with self.subTest(kind=kind), workspace_temporary_directory() as folder:
                original = catalogue(kind)
                original.save(folder)
                restored = CatalogIndex.load(folder, threads=1)
                self.assertEqual(original.version, restored.version)
                self.assertEqual(original.search([1, 0], 3), restored.search([1, 0], 3))
                restored.warmup(1)

    def test_corrupted_artifact_is_rejected_before_faiss_load(self):
        with workspace_temporary_directory() as folder:
            catalogue().save(folder)
            path = Path(folder) / "catalogue.faiss"
            with path.open("ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                CatalogIndex.load(folder)

    def test_manifest_tampering_is_rejected(self):
        with workspace_temporary_directory() as folder:
            catalogue().save(folder)
            path = Path(folder) / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["default_as_of"] = 999999
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                CatalogIndex.load(folder)

    def test_invalid_query_and_k_are_rejected(self):
        index = catalogue()
        for query in ([0, 0], [np.nan, 0], [1, 2, 3]):
            with self.assertRaises(ValueError):
                index.search(query, 2)
        for k in (0, -1, True):
            with self.assertRaises(ValueError):
                index.search([1, 0], k)
        with self.assertRaises(ValueError):
            index.search([1, 0], 1, as_of=UNKNOWN_TIME)

    def test_hnsw_matches_high_recall_reference_on_fixed_small_fixture(self):
        rng = np.random.default_rng(104)
        matrix = rng.normal(size=(401, 16)).astype(np.float32)
        matrix[0] = 0
        settings = (matrix, ["PAD"] + [f"N{i}" for i in range(1, 401)],
                    np.full(401, 10), np.zeros(401, dtype=np.int64))
        exact = CatalogIndex.build(*settings, default_as_of=10, kind="exact")
        approx = CatalogIndex.build(*settings, default_as_of=10, kind="hnsw", hnsw_m=16,
                                    ef_construction=80, ef_search=128)
        recalls = []
        for query in rng.normal(size=(5, 16)).astype(np.float32):
            expected = set(exact.search(query, 10).indices)
            actual = set(approx.search(query, 10).indices)
            recalls.append(len(expected & actual) / len(expected))
        self.assertGreaterEqual(float(np.mean(recalls)), .95)


if __name__ == "__main__":
    unittest.main()
