"""Persisted FAISS retrieval with observable-catalogue and history filters.

An index may contain items first observed after the default training cutoff.
Every search, including popularity fallback, filters using the request cutoff.
First exposure is an availability proxy, not a publisher publication timestamp.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


UNKNOWN_TIME = np.iinfo(np.int64).max
RESERVED_IDS = {"", "PAD", "[PAD]", "<PAD>", "__PAD__", "UNK", "[UNK]", "<UNK>",
                "__UNK__", "OOV", "[OOV]", "<OOV>", "__OOV__"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RetrievalResult:
    indices: list[int]
    scores: list[float]
    requested: int
    examined: int
    exhausted: bool
    effective_as_of: int

    @property
    def shortfall(self) -> int:
        return max(0, self.requested - len(self.indices))


class CatalogIndex:
    """FAISS inner-product index over unit vectors, preserving dataset IDs.

    ``exact`` is IndexFlatIP; ``hnsw`` is IndexHNSWFlat with the same IP metric.
    Numeric IDs are row numbers in news_features/article_embeddings. Row 0 and
    reserved string IDs cannot enter the index. Unknown availability, zero and
    non-finite embeddings cannot be returned. Search never fills a shortfall
    with a future or previously seen item.
    """

    def __init__(self, index, news_ids, first_seen, train_clicks, indexed_ids, manifest):
        self.index = index
        self.news_ids = list(news_ids)
        self.first_seen = np.asarray(first_seen, dtype=np.int64)
        self.train_clicks = np.asarray(train_clicks, dtype=np.int64)
        self.indexed_ids = np.asarray(indexed_ids, dtype=np.int64)
        self.manifest = dict(manifest)
        self.threads = int(manifest["faiss_threads"])
        self.id_to_index = {news_id: i for i, news_id in enumerate(self.news_ids)
                            if i > 0 and news_id.upper() not in RESERVED_IDS}
        self._indexed_mask = np.zeros(len(self.news_ids), dtype=bool)
        self._indexed_mask[self.indexed_ids] = True
        # Primary key descending training clicks; ties ascending row number.
        self._popular_ids = self.indexed_ids[np.lexsort((self.indexed_ids,
                                                        -self.train_clicks[self.indexed_ids]))]

    @property
    def version(self) -> str:
        return self.manifest.get("index_version", "unsaved")

    @property
    def default_as_of(self) -> int:
        return int(self.manifest["default_as_of"])

    @classmethod
    def build(cls, embeddings: np.ndarray, news_ids: list[str], first_seen: np.ndarray,
              train_clicks: np.ndarray, *, default_as_of: int, kind: str = "hnsw",
              hnsw_m: int = 32, ef_construction: int = 160, ef_search: int = 128,
              threads: int = 1, model_version: str = "unknown", source_hashes: dict | None = None):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        first_seen = np.asarray(first_seen, dtype=np.int64)
        train_clicks = np.asarray(train_clicks, dtype=np.int64)
        if embeddings.ndim != 2 or embeddings.shape[1] < 1:
            raise ValueError("embeddings must have shape (number_of_items, dimension)")
        n = len(news_ids)
        if embeddings.shape[0] != n or first_seen.shape != (n,) or train_clicks.shape != (n,):
            raise ValueError("catalogue arrays must have identical row counts")
        if any(not isinstance(value, str) for value in news_ids) or len(set(news_ids)) != n:
            raise ValueError("news_ids must be unique strings, including the PAD row")
        if np.any(train_clicks < 0):
            raise ValueError("train_clicks must be nonnegative")
        if kind not in {"exact", "hnsw"}:
            raise ValueError("kind must be exact or hnsw")
        if default_as_of < 0 or default_as_of >= UNKNOWN_TIME:
            raise ValueError("default_as_of must be a known Unix timestamp")
        if threads < 1 or hnsw_m < 2 or ef_search < 1 or ef_construction < 1:
            raise ValueError("invalid FAISS threading/HNSW settings")
        faiss.omp_set_num_threads(threads)
        norms = np.linalg.norm(embeddings, axis=1)
        valid = ((np.arange(n) > 0) & (first_seen >= 0) & (first_seen < UNKNOWN_TIME)
                 & np.isfinite(embeddings).all(axis=1) & np.isfinite(norms) & (norms > 1e-12))
        valid &= np.array([value.upper() not in RESERVED_IDS for value in news_ids])
        item_ids = np.flatnonzero(valid).astype(np.int64)
        if len(item_ids) == 0:
            raise ValueError("catalogue contains no eligible non-PAD items")
        matrix = np.ascontiguousarray(embeddings[item_ids] / norms[item_ids, None])
        dimension = embeddings.shape[1]
        if kind == "exact":
            base = faiss.IndexFlatIP(dimension)
        else:
            base = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efConstruction = ef_construction
            base.hnsw.efSearch = ef_search
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(matrix, item_ids)
        manifest = {
            "format_version": 1, "kind": kind, "metric": "inner_product_on_unit_vectors",
            "dimension": dimension, "catalogue_rows": n, "indexed_items": len(item_ids),
            "default_as_of": int(default_as_of),
            "minimum_first_seen": int(first_seen[item_ids].min()),
            "maximum_first_seen": int(first_seen[item_ids].max()),
            "availability_semantics": "first observed timestamp; publication time unavailable",
            "model_version": model_version, "faiss_version": faiss.__version__,
            "faiss_threads": threads, "hnsw_m": hnsw_m if kind == "hnsw" else None,
            "ef_construction": ef_construction if kind == "hnsw" else None,
            "ef_search": ef_search if kind == "hnsw" else None,
            "source_hashes": dict(source_hashes or {}),
        }
        return cls(index, news_ids, first_seen, train_clicks, item_ids, manifest)

    def save(self, directory: str | Path) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "catalogue.faiss"))
        (directory / "news_ids.json").write_text(json.dumps(self.news_ids, ensure_ascii=False), encoding="utf-8")
        np.save(directory / "first_seen.npy", self.first_seen, allow_pickle=False)
        np.save(directory / "train_clicks.npy", self.train_clicks, allow_pickle=False)
        np.save(directory / "indexed_ids.npy", self.indexed_ids, allow_pickle=False)
        manifest = {k: v for k, v in self.manifest.items() if k not in {"files", "index_version", "created_at_utc"}}
        manifest["files"] = {name: sha256_file(directory / name) for name in
                             ("catalogue.faiss", "news_ids.json", "first_seen.npy",
                              "train_clicks.npy", "indexed_ids.npy")}
        manifest["index_version"] = _canonical_hash(manifest)
        manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.manifest = manifest
        return manifest

    @classmethod
    def load(cls, directory: str | Path, *, threads: int | None = None):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        unhashed = {key: value for key, value in manifest.items() if key not in {"index_version", "created_at_utc"}}
        if _canonical_hash(unhashed) != manifest.get("index_version"):
            raise ValueError("index manifest hash mismatch")
        expected_files = {"catalogue.faiss", "news_ids.json", "first_seen.npy", "train_clicks.npy", "indexed_ids.npy"}
        if set(manifest.get("files", {})) != expected_files:
            raise ValueError("index manifest must contain the expected local artifact set")
        for name, expected in manifest["files"].items():
            if sha256_file(directory / name) != expected:
                raise ValueError(f"index artifact hash mismatch: {name}")
        worker_threads = int(threads if threads is not None else manifest["faiss_threads"])
        if worker_threads < 1:
            raise ValueError("threads must be positive")
        faiss.omp_set_num_threads(worker_threads)
        index = faiss.read_index(str(directory / "catalogue.faiss"))
        news_ids = json.loads((directory / "news_ids.json").read_text(encoding="utf-8"))
        first_seen = np.load(directory / "first_seen.npy", allow_pickle=False)
        train_clicks = np.load(directory / "train_clicks.npy", allow_pickle=False)
        indexed_ids = np.load(directory / "indexed_ids.npy", allow_pickle=False)
        n = manifest["catalogue_rows"]
        if (len(news_ids) != n or first_seen.shape != (n,) or train_clicks.shape != (n,)
                or index.ntotal != manifest["indexed_items"] or index.d != manifest["dimension"]
                or indexed_ids.shape != (index.ntotal,) or not np.array_equal(
                    faiss.vector_to_array(index.id_map), indexed_ids)):
            raise ValueError("index/catalogue shape or ID-map mismatch")
        if len(set(news_ids)) != n or np.any(indexed_ids <= 0) or np.any(indexed_ids >= n):
            raise ValueError("invalid catalogue ID map")
        result = cls(index, news_ids, first_seen, train_clicks, indexed_ids, manifest)
        result.threads = worker_threads
        return result

    def _cutoff(self, as_of: int | None) -> int:
        cutoff = self.default_as_of if as_of is None else int(as_of)
        if cutoff < 0 or cutoff >= UNKNOWN_TIME:
            raise ValueError("as_of must be a known nonnegative Unix timestamp")
        return cutoff

    def eligible(self, indices: np.ndarray, *, as_of: int | None = None,
                 seen: Iterable[int] = ()) -> np.ndarray:
        """Boolean eligibility mask; FAISS -1 and other invalid IDs are safe."""
        indices = np.asarray(indices, dtype=np.int64)
        valid = (indices > 0) & (indices < len(self.news_ids))
        safe = np.clip(indices, 0, len(self.news_ids) - 1)
        valid &= self._indexed_mask[safe] & (self.first_seen[safe] <= self._cutoff(as_of))
        seen_array = np.fromiter((int(value) for value in seen), dtype=np.int64)
        if len(seen_array):
            valid &= ~np.isin(indices, seen_array)
        return valid

    def search(self, query: np.ndarray, k: int, *, as_of: int | None = None,
               seen: Iterable[int] = (), max_overfetch: int | None = None) -> RetrievalResult:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape != (self.index.d,) or not np.isfinite(vector).all():
            raise ValueError("query must be a finite vector of index dimension")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("query vector must have nonzero norm; use popularity for cold history")
        vector = np.ascontiguousarray((vector / norm)[None, :])
        # OpenMP thread settings are per caller thread. FastAPI runs synchronous
        # handlers in worker threads, so configure the actual search thread.
        faiss.omp_set_num_threads(self.threads)
        cutoff = self._cutoff(as_of)
        seen = set(int(value) for value in seen)
        limit = self.index.ntotal if max_overfetch is None else min(self.index.ntotal, int(max_overfetch))
        if limit < 1:
            raise ValueError("max_overfetch must be positive")
        fetch = min(limit, max(k * 2, k + 16))
        while True:
            scores, indices = self.index.search(vector, fetch)
            valid = self.eligible(indices[0], as_of=cutoff, seen=seen)
            chosen_ids, chosen_scores = indices[0][valid][:k], scores[0][valid][:k]
            if len(chosen_ids) >= k or fetch >= limit:
                return RetrievalResult(chosen_ids.astype(int).tolist(),
                                       chosen_scores.astype(float).tolist(), k, fetch,
                                       fetch >= self.index.ntotal, cutoff)
            fetch = min(limit, max(fetch + 1, fetch * 2))

    def popular(self, k: int, *, as_of: int | None = None, seen: Iterable[int] = ()) -> RetrievalResult:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        cutoff = self._cutoff(as_of)
        valid = self.eligible(self._popular_ids, as_of=cutoff, seen=seen)
        selected = self._popular_ids[valid][:k]
        return RetrievalResult(selected.astype(int).tolist(),
                               np.log1p(self.train_clicks[selected]).astype(float).tolist(),
                               k, len(self._popular_ids), True, cutoff)

    def warmup(self, iterations: int = 3) -> None:
        """Touch the index through real searches, without changing the weights."""
        if iterations < 0:
            raise ValueError("iterations must be nonnegative")
        vector = np.asarray(self.index.reconstruct(int(self.indexed_ids[0])), dtype=np.float32)
        for _ in range(iterations):
            self.search(vector, min(10, self.index.ntotal), max_overfetch=min(256, self.index.ntotal))


def build_from_files(processed_dir: str | Path, embedding_path: str | Path, output_dir: str | Path,
                     *, default_as_of: int, kind="hnsw", threads=1, model_version="unknown", **kwargs):
    processed_dir, embedding_path = Path(processed_dir), Path(embedding_path)
    paths = {"article_embeddings": embedding_path,
             "news_ids": processed_dir / "news_ids.json",
             "first_seen": processed_dir / "news_first_seen.npy",
             "train_clicks": processed_dir / "news_train_clicks.npy",
             "news_features": processed_dir / "news_features.npy"}
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    catalogue = CatalogIndex.build(np.load(embedding_path, mmap_mode="r", allow_pickle=False),
                                   json.loads(paths["news_ids"].read_text(encoding="utf-8")),
                                   np.load(paths["first_seen"], allow_pickle=False),
                                   np.load(paths["train_clicks"], allow_pickle=False),
                                   default_as_of=default_as_of, kind=kind, threads=threads,
                                   model_version=model_version, source_hashes=source_hashes, **kwargs)
    catalogue.save(output_dir)
    return catalogue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--embeddings", default="artifacts/article_embeddings.npy")
    parser.add_argument("--output", default="artifacts/index_hnsw")
    parser.add_argument("--kind", choices=["exact", "hnsw"], default="hnsw")
    parser.add_argument("--default-as-of", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=160)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--model-version", default="unknown")
    args = parser.parse_args()
    catalogue = build_from_files(args.processed_dir, args.embeddings, args.output,
                                 default_as_of=args.default_as_of, kind=args.kind,
                                 threads=args.threads, model_version=args.model_version,
                                 hnsw_m=args.hnsw_m, ef_construction=args.ef_construction,
                                 ef_search=args.ef_search)
    print(json.dumps(catalogue.manifest, indent=2))


if __name__ == "__main__":
    main()
