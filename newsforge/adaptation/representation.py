"""Causal, CPU-only title representation for the Adressa adaptation experiment.

The public entry point fixes the formal representation at 96 text dimensions and
19 category dimensions. Serialized estimators contain source-derived vocabulary
and are trusted-local artifacts, never publication artifacts. Only aggregate
counts and hashes are returned as metadata. The caller retains the existing
``content_features`` implementation to derive its original 13 scalar features.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
import struct

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


FIT_CUTOFF = 1483311600  # 2017-01-01 23:00:00 UTC, exclusive.
_ARTIFACT_NAMES = ("vectorizer.joblib", "svd.joblib", "features.npy", "coverage.npy")


@dataclass(frozen=True)
class RepresentationConfig:
    """Frozen formal defaults; smaller dimensions are for synthetic tests only."""

    n_components: int = 96
    category_dimensions: int = 19
    batch_size: int = 4096


DEFAULT_CONFIG = RepresentationConfig()


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=40000, sublinear_tf=True, norm="l2",
        dtype=np.float32, stop_words=None,
    )


def _update_source_hash(digest, row) -> None:
    """Hash representation inputs in a documented, unambiguous byte encoding."""
    article_id, stamp, title = row
    payload = title.encode("utf-8")
    digest.update(struct.pack("<qqQ", article_id, stamp, len(payload)))
    digest.update(payload)


def _validate_row(row, first_seen: np.ndarray) -> None:
    article_id, stamp, title = row
    if (not isinstance(article_id, int) or article_id <= 0
            or article_id >= len(first_seen)):
        raise ValueError("article IDs must index nonpadding feature rows")
    if not isinstance(stamp, int) or stamp < 0 or stamp != first_seen[article_id]:
        raise ValueError("article first_seen does not match the supplied array")
    if not isinstance(title, str):
        raise ValueError("first-observed article titles must be strings")


def _empty_coverage() -> dict:
    return {
        "articles": 0, "empty_title": 0, "known_vocabulary_rows": 0,
        "zero_tfidf": 0, "zero_svd": 0, "unknown_category": 0,
        "analyzed_ngrams": 0, "known_vocabulary_ngrams": 0,
    }


def _finish_coverage(value: dict) -> dict:
    articles = value["articles"]
    ngrams = value["analyzed_ngrams"]
    return {
        **value,
        "known_vocabulary_row_fraction": value["known_vocabulary_rows"] / articles if articles else 0.0,
        "known_vocabulary_ngram_fraction": value["known_vocabulary_ngrams"] / ngrams if ngrams else 0.0,
        "zero_tfidf_fraction": value["zero_tfidf"] / articles if articles else 0.0,
        "zero_svd_fraction": value["zero_svd"] / articles if articles else 0.0,
    }


def fit_local_representation(
    connection: sqlite3.Connection,
    first_seen: np.ndarray,
    out_dir: str | Path,
    fit_cutoff: int = FIT_CUTOFF,
) -> tuple[np.ndarray, dict]:
    """Fit on first titles strictly before ``fit_cutoff``, then transform all.

    ``connection`` is the caller's read-only original events database. Only
    SELECT queries are issued. Its article IDs must be the contiguous integers
    1..len(first_seen)-1; row 0 is padding. The caller may supply a new directory
    or an existing stage directory, but none of the four artifacts may exist.
    Insufficient fit rows or vocabulary fail rather than changing parameters.
    """
    return _fit_local_representation(connection, first_seen, out_dir, fit_cutoff, DEFAULT_CONFIG)


def _fit_local_representation(
    connection: sqlite3.Connection,
    first_seen: np.ndarray,
    out_dir: str | Path,
    fit_cutoff: int,
    config: RepresentationConfig,
) -> tuple[np.ndarray, dict]:
    """Implementation with a private configuration hook for synthetic tests."""
    if (not isinstance(config.n_components, int) or config.n_components < 1
            or not isinstance(config.category_dimensions, int) or config.category_dimensions < 1
            or not isinstance(config.batch_size, int) or config.batch_size < 1):
        raise ValueError("representation dimensions and batch size must be positive integers")
    if isinstance(fit_cutoff, bool) or not isinstance(fit_cutoff, (int, np.integer)):
        raise ValueError("fit cutoff must be an integer UTC timestamp")
    fit_cutoff = int(fit_cutoff)
    cutoff_utc = datetime.fromtimestamp(fit_cutoff, timezone.utc).isoformat()
    first_seen = np.asarray(first_seen)
    if (first_seen.ndim != 1 or len(first_seen) < 2
            or not np.issubdtype(first_seen.dtype, np.integer)
            or np.any(first_seen[1:] < 0)):
        raise ValueError("first_seen must be an integer array with nonnegative article timestamps")
    out_dir = Path(out_dir)
    for name in _ARTIFACT_NAMES:
        if (out_dir / name).exists() or (out_dir / name).is_symlink():
            raise FileExistsError("refusing to overwrite a representation artifact")

    # The SQL predicate prevents future documents from influencing vocabulary,
    # document frequencies, IDF values, or SVD axes.
    fit_rows = list(connection.execute(
        "SELECT id,first_seen,title FROM articles WHERE first_seen < ? ORDER BY id", (fit_cutoff,)))
    required = config.n_components + 1
    if len(fit_rows) < required:
        raise ValueError(f"insufficient fit documents: at least {required} are required")
    previous_id = 0
    fit_source = hashlib.sha256()
    for row in fit_rows:
        _validate_row(row, first_seen)
        if row[0] <= previous_id:
            raise ValueError("fit article IDs must be unique and ordered")
        previous_id = row[0]
        _update_source_hash(fit_source, row)
    fit_ids = np.asarray([row[0] for row in fit_rows], dtype="<i8")
    fit_ids_hash = hashlib.sha256(fit_ids.tobytes(order="C")).hexdigest()
    fit_source_hash = fit_source.hexdigest()
    fit_docs = len(fit_rows)
    vectorizer = _vectorizer()
    svd = TruncatedSVD(n_components=config.n_components, n_iter=7, random_state=2026)
    dimensions = config.n_components + config.category_dimensions
    features = np.zeros((len(first_seen), dimensions), dtype=np.float32)
    flags = np.zeros((len(first_seen), 4), dtype=np.uint8)
    coverage = {"all": _empty_coverage(), "fit": _empty_coverage()}
    source = hashlib.sha256()
    transformed_fit_source = hashlib.sha256()

    with threadpool_limits(limits=1):
        try:
            fit_matrix = vectorizer.fit_transform([row[2] for row in fit_rows])
        except ValueError as exc:
            # sklearn's empty/pruned-vocabulary errors must not result in a
            # relaxed min_df or a different dimensionality.
            raise ValueError("insufficient fit vocabulary for the fixed representation") from exc
        if fit_matrix.shape[1] < required:
            raise ValueError(f"insufficient fit vocabulary: at least {required} features are required")
        svd.fit(fit_matrix)
        del fit_rows, fit_matrix, fit_ids
        if svd.components_.shape != (config.n_components, len(vectorizer.vocabulary_)):
            raise ValueError("SVD did not produce the fixed text dimensionality")
        if not np.isfinite(svd.components_).all():
            raise ValueError("nonfinite fitted SVD components")

        analyzer = vectorizer.build_analyzer()
        vocabulary = vectorizer.vocabulary_
        next_id = 1
        cursor = connection.execute("SELECT id,first_seen,title FROM articles ORDER BY id")
        while True:
            rows = cursor.fetchmany(config.batch_size)
            if not rows:
                break
            for row in rows:
                _validate_row(row, first_seen)
                if row[0] != next_id:
                    raise ValueError("article IDs must be contiguous and unique")
                next_id += 1
                _update_source_hash(source, row)
                if row[1] < fit_cutoff:
                    _update_source_hash(transformed_fit_source, row)
            ids = np.asarray([row[0] for row in rows], dtype=np.int64)
            matrix = vectorizer.transform([row[2] for row in rows])
            text = normalize(svd.transform(matrix), norm="l2", copy=False).astype(np.float32, copy=False)
            if text.shape != (len(rows), config.n_components) or not np.isfinite(text).all():
                raise ValueError("invalid local text representation")
            features[ids, :config.n_components] = text
            features[ids, config.n_components] = 1.0  # Original first category column is UNKNOWN.
            zero_tfidf = np.asarray(matrix.getnnz(axis=1) == 0)
            zero_svd = np.count_nonzero(text, axis=1) == 0
            flags[ids] = np.column_stack((
                np.asarray([not bool(row[2]) for row in rows]),
                zero_tfidf, zero_svd, np.ones(len(rows), dtype=bool)))
            for index, row in enumerate(rows):
                ngrams = analyzer(row[2])
                known = sum(ngram in vocabulary for ngram in ngrams)
                values = {
                    "articles": 1, "empty_title": int(not bool(row[2])),
                    "known_vocabulary_rows": int(not zero_tfidf[index]),
                    "zero_tfidf": int(zero_tfidf[index]), "zero_svd": int(zero_svd[index]),
                    "unknown_category": 1, "analyzed_ngrams": len(ngrams),
                    "known_vocabulary_ngrams": known,
                }
                groups = ("all", "fit") if row[1] < fit_cutoff else ("all",)
                for group in groups:
                    for key, value in values.items():
                        coverage[group][key] += value
        if next_id != len(first_seen):
            raise ValueError("article table does not cover the supplied first_seen array")
        if transformed_fit_source.hexdigest() != fit_source_hash:
            raise ValueError("fit article inputs changed between fitting and transformation")

    metadata = {
        "schema_version": 1,
        "fit_docs": fit_docs,
        "articles": len(first_seen) - 1,
        "dimensions": dimensions,
        "feature_dimensions": dimensions,
        "text_dimensions": config.n_components,
        "category_dimensions": config.category_dimensions,
        "category_unknown_column": config.n_components,
        "fit_cutoff": fit_cutoff,
        "fit_cutoff_utc": cutoff_utc,
        "fit_predicate": "first_seen < fit_cutoff",
        "fit_ids_array_sha256": fit_ids_hash,
        "fit_ids_hash_encoding": "ordered signed int64 little-endian array, without header",
        "fit_source_sha256": fit_source_hash,
        "source_sha256": source.hexdigest(),
        "source_hash_encoding": "ordered articles: int64le id, int64le first_seen, uint64le UTF-8 title byte count, title bytes",
        "vocabulary_features": len(vectorizer.vocabulary_),
        "coverage": {group: _finish_coverage(value) for group, value in coverage.items()},
        "coverage_columns": ["empty_title", "zero_tfidf", "zero_svd", "unknown_category"],
        "params": {
            "vectorizer": {key: np.dtype(value).name if key == "dtype"
                else list(value) if isinstance(value, tuple) else value
                for key, value in vectorizer.get_params(deep=False).items()},
            "svd": svd.get_params(deep=False),
            "text_normalization": "l2", "batch_size": config.batch_size, "numeric_threads": 1,
        },
        "category_policy": "all nonpadding articles use the original first UNKNOWN category column",
        "scalar_feature_policy": "reuse the original content_features 13 scalars; SVD axes are not ranker features",
        "artifact_policy": "trusted local only; do not publish estimators, titles, vocabulary, IDs or feature rows",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive file creation also protects against a concurrent writer after
    # the initial existence check. File objects keep joblib from using a
    # nonexclusive filename-based write or generating auxiliary files.
    with ExitStack() as stack:
        streams = {name: stack.enter_context((out_dir / name).open("xb")) for name in _ARTIFACT_NAMES}
        joblib.dump(vectorizer, streams["vectorizer.joblib"], compress=0)
        joblib.dump(svd, streams["svd.joblib"], compress=0)
        np.save(streams["features.npy"], features, allow_pickle=False)
        np.save(streams["coverage.npy"], flags, allow_pickle=False)
    return features, metadata
