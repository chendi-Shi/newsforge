"""Streaming MIND-large preprocessing with a locked official development set.

The official train file is divided by time; no labels or text from the internal
validation or official development periods fit the text encoder. Catalog IDs are
array addresses, not learned item-ID features. The source provides no timezone;
naive timestamps are interpreted as UTC only to obtain a reproducible ordering.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

BASE_URL = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main"
SOURCE_URL = "https://github.com/recommenders-team/recommenders/blob/main/recommenders/datasets/mind.py"
TERMS_URL = "https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf"
EXPECTED_BYTES = {"train": 531360717, "dev": 103592887}
EXPECTED_SHA256 = {
    "train": "7e7da0ecc45db1d8769eebf92301675e7af443b0d5266ac12b6e102bb8399233",
    "dev": "67b0437b48b6baee5655b6063a56d29edbcc92308fc0140555b0ae27263cf9e9",
}
MAX_TIME = np.iinfo(np.int64).max


def _json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(raw: Path, *, workers=2):
    """Download publicly accessible, upstream-documented files; no credentials."""
    raw.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        url = f"{BASE_URL}/MINDlarge_{name}.zip"
        archive = raw / f"MINDlarge_{name}.zip"
        if not archive.exists() or archive.stat().st_size != EXPECTED_BYTES[name]:
            temporary = archive.with_suffix(".download")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if temporary.stat().st_size != EXPECTED_BYTES[name]:
                raise ValueError(f"Incomplete download: {name}")
            if sha256(temporary) != EXPECTED_SHA256[name]:
                raise ValueError(f"Archive SHA256 mismatch: {name}")
            temporary.replace(archive)
        digest = sha256(archive)
        if digest != EXPECTED_SHA256[name]:
            raise ValueError(f"Archive SHA256 mismatch: {name}; remove the invalid archive and retry")
        destination = raw / name
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for member in ("news.tsv", "behaviors.tsv"):
                target = destination / member
                if target.exists() and target.stat().st_size == bundle.getinfo(member).file_size:
                    continue
                with bundle.open(member) as source, target.open("wb") as out:
                    while chunk := source.read(1024 * 1024):
                        out.write(chunk)
        receipt = {"url": url, "bytes": archive.stat().st_size, "sha256": digest,
                   "extracted": ["news.tsv", "behaviors.tsv"], "source": SOURCE_URL}
        _json(raw / f"{name}_download.json", receipt)
        print(f"Downloaded and verified MIND-large {name}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, ("train", "dev")))


@lru_cache(maxsize=65536)
def parse_timestamp(value: str) -> int:
    return int(datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p").replace(tzinfo=timezone.utc).timestamp())


def time_split(timestamps: np.ndarray, train_rows: int, fraction=.85):
    """Keep same-second impressions together and the official dev fully locked."""
    if not 0 < fraction < 1 or train_rows < 2:
        raise ValueError("Need a nonempty train period and fraction in (0, 1)")
    values = np.asarray(timestamps[:train_rows])
    rank = min(int(train_rows * fraction), train_rows - 1)
    cutoff = int(np.partition(values, rank)[rank])
    splits = np.full(len(timestamps), 2, dtype=np.uint8)
    splits[:train_rows] = np.where(values <= cutoff, 0, 1)
    if not np.any(splits == 0) or not np.any(splits == 1):
        raise ValueError("Timestamp cutoff did not produce both train and validation")
    return splits, cutoff


def read_news(raw: Path):
    """Prefer the train-file version when duplicate catalog records differ."""
    news = {"[PAD]": {"text": "", "category": ""}}
    changed = 0
    for source in ("train", "dev"):
        with (raw / source / "news.tsv").open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\n\r").split("\t")
                if len(fields) != 8:
                    raise ValueError(f"Expected 8 news fields in {source}, got {len(fields)}")
                record = {"text": fields[3] + " " + fields[4], "category": fields[1]}
                if fields[0] not in news:
                    news[fields[0]] = record
                elif news[fields[0]] != record:
                    changed += 1
    return news, changed


def _scan(raw: Path, output: Path):
    """First pass writes fixed-width scratch arrays, never expanded interactions."""
    counts, positives, stamps = [], [], []
    total_rows = total_candidates = total_positives = 0
    rows_by_source = {}
    unique_referenced = set()
    paths = [output / f".scan_{key}.bin" for key in ("counts", "positives", "timestamps")]
    with paths[0].open("wb") as count_file, paths[1].open("wb") as positive_file, paths[2].open("wb") as stamp_file:
        def flush():
            np.asarray(counts, dtype=np.int64).tofile(count_file)
            np.asarray(positives, dtype=np.int32).tofile(positive_file)
            np.asarray(stamps, dtype=np.int64).tofile(stamp_file)
            counts.clear(); positives.clear(); stamps.clear()
        for source in ("train", "dev"):
            row_count = 0
            with (raw / source / "behaviors.tsv").open(encoding="utf-8") as stream:
                for line in stream:
                    fields = line.rstrip("\r\n").split("\t")
                    if len(fields) != 5:
                        raise ValueError(f"Expected 5 behavior fields in {source}")
                    candidates = fields[4].split()
                    if not candidates:
                        raise ValueError("An impression has no candidates")
                    pos = 0
                    for candidate in candidates:
                        if candidate[-2:] not in ("-0", "-1"):
                            raise ValueError(f"Missing/invalid candidate label: {candidate}")
                        unique_referenced.add(candidate[:-2])
                        pos += candidate[-1] == "1"
                    unique_referenced.update(fields[3].split())
                    counts.append(len(candidates)); positives.append(pos); stamps.append(parse_timestamp(fields[2]))
                    row_count += 1; total_rows += 1
                    total_candidates += len(candidates); total_positives += pos
                    if len(counts) >= 100000:
                        flush()
                    if total_rows % 250000 == 0:
                        print(f"Scan: {total_rows:,} impressions, {total_candidates:,} candidates", flush=True)
            rows_by_source[source] = row_count
        flush()
    parse_timestamp.cache_clear()
    return {"rows": total_rows, "candidates": total_candidates, "positives": total_positives,
            "rows_by_source": rows_by_source, "referenced_ids": unique_referenced}


def _array(output: Path, name: str, dtype, shape):
    return np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)


def build_news_features(news, fit_mask, output: Path, *, components=96, max_features=60000,
                        min_df=2, threads=4):
    """Fit lexical statistics and SVD on text observable in split 0 only."""
    records = list(news.values())
    fit_indices = np.flatnonzero(fit_mask)
    if not len(fit_indices):
        raise ValueError("No news was observed in the training period")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=min_df,
                                 dtype=np.float32, sublinear_tf=True, strip_accents="unicode")
    print(f"TF-IDF fitting {len(fit_indices):,} training-observable news", flush=True)
    corpus = [records[i]["text"] for i in fit_indices]
    matrix = vectorizer.fit_transform(corpus)
    n_components = min(components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if n_components < 1:
        raise ValueError("Too little training text for TruncatedSVD")
    svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=2026)
    with threadpool_limits(limits=threads):
        svd.fit(matrix)
    del matrix, corpus
    category_names = sorted({records[i]["category"] for i in fit_indices})
    category_vocab = {name: index + 1 for index, name in enumerate(category_names)}
    features = _array(output, "news_features", np.float32, (len(records), components + len(category_names) + 1))
    features[:] = 0
    with threadpool_limits(limits=threads):
        for start in range(1, len(records), 8192):
            end = min(start + 8192, len(records))
            block = records[start:end]
            text = normalize(svd.transform(vectorizer.transform([r["text"] for r in block])), norm="l2")
            features[start:end, :n_components] = text
            cat = np.asarray([category_vocab.get(r["category"], 0) for r in block])
            features[np.arange(start, end), components + cat] = 1
    features.flush()
    joblib.dump(vectorizer, output / "tfidf.joblib")
    joblib.dump(svd, output / "svd.joblib")
    _json(output / "category_vocabulary.json", ["[UNKNOWN]"] + category_names)
    return {"dimensions": int(features.shape[1]), "text_dimensions": components,
            "svd_actual_components": n_components, "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
            "tfidf_max_features": max_features, "ngram_range": [1, 2], "min_df": min_df,
            "fit_news": len(fit_indices), "fit_policy": "only history/candidate IDs observable in split 0",
            "categories": len(category_names), "category_unknown_column": components,
            "text_l2_normalized": True, "padding_row_all_zero": True,
            "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum())}


def prepare(raw: Path, output: Path, *, history_length=20, components=96, max_features=60000,
            min_df=2, threads=4, download=False):
    started = time.perf_counter()
    if download:
        download_dataset(raw)
    output.mkdir(parents=True, exist_ok=True)
    news, changed_records = read_news(raw)
    scan = _scan(raw, output)
    missing = sorted(scan.pop("referenced_ids") - news.keys())
    for identifier in missing:
        news[identifier] = {"text": "", "category": "[MISSING]"}
    ids = list(news)
    index = {identifier: value for value, identifier in enumerate(ids)}
    _json(output / "news_ids.json", ids)
    n, m = scan["rows"], scan["candidates"]
    counts = np.memmap(output / ".scan_counts.bin", mode="r", dtype=np.int64)
    positive_counts = np.memmap(output / ".scan_positives.bin", mode="r", dtype=np.int32)
    timestamps = _array(output, "all_timestamps", np.int64, (n,))
    scan_timestamps = np.memmap(output / ".scan_timestamps.bin", mode="r", dtype=np.int64)
    timestamps[:] = scan_timestamps
    split_values, cutoff = time_split(timestamps, scan["rows_by_source"]["train"])
    split = _array(output, "all_split", np.uint8, (n,)); split[:] = split_values
    offsets = _array(output, "all_offsets", np.int64, (n + 1,))
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    candidates = _array(output, "all_candidates", np.int32, (m,))
    labels = _array(output, "all_labels", np.uint8, (m,))
    history = _array(output, "all_history", np.int32, (n, history_length)); history[:] = 0
    users = _array(output, "all_users", np.int32, (n,))
    impression_ids = _array(output, "all_impression_ids", np.int64, (n,))
    first_seen = np.full(len(ids), MAX_TIME, dtype=np.int64)
    train_clicks = np.zeros(len(ids), dtype=np.int64)
    train_exposures = np.zeros(len(ids), dtype=np.int64)
    train_seen = np.zeros(len(ids), dtype=bool)
    training_positive_count = int(positive_counts[split_values == 0].sum())
    positive_rows = _array(output, "train_positive_rows", np.int32, (training_positive_count,))
    positive_items = _array(output, "train_positive_items", np.int32, (training_positive_count,))
    row = positive_offset = empty_history = same_impression_repeats = 0
    split_users = [set(), set(), set()]
    split_news = [set(), set(), set()]
    split_stats = {str(i): {"impressions": 0, "candidates": 0, "positives": 0} for i in range(3)}
    for source in ("train", "dev"):
        with (raw / source / "behaviors.tsv").open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\r\n").split("\t")
                if not fields[1].startswith("U") or not fields[1][1:].isdigit():
                    raise ValueError("Expected a numeric anonymized U-prefixed user ID")
                user = int(fields[1][1:]); users[row] = user
                impression_ids[row] = int(fields[0])
                full_history = [index[value] for value in fields[3].split()]
                recent_history = full_history[-history_length:]
                history[row, :len(recent_history)] = recent_history
                empty_history += not full_history
                tokens = fields[4].split()
                items = np.fromiter((index[token[:-2]] for token in tokens), dtype=np.int32, count=len(tokens))
                target = np.fromiter((token[-1] == "1" for token in tokens), dtype=np.uint8, count=len(tokens))
                left, right = int(offsets[row]), int(offsets[row + 1])
                candidates[left:right] = items; labels[left:right] = target
                if len(set(items)) != len(items):
                    same_impression_repeats += 1
                when = int(timestamps[row]); which = int(split[row])
                np.minimum.at(first_seen, items, when)
                if full_history:
                    np.minimum.at(first_seen, full_history, when)
                split_users[which].add(user)
                split_news[which].update(items)
                stats = split_stats[str(which)]
                stats["impressions"] += 1; stats["candidates"] += len(items); stats["positives"] += int(target.sum())
                if which == 0:
                    train_seen[items] = True; train_seen[full_history] = True
                    np.add.at(train_exposures, items, 1)
                    np.add.at(train_clicks, items, target)
                    chosen = items[target == 1]
                    end = positive_offset + len(chosen)
                    positive_rows[positive_offset:end] = row; positive_items[positive_offset:end] = chosen
                    positive_offset = end
                row += 1
                if row % 250000 == 0:
                    print(f"Materialized {row:,}/{n:,} impressions", flush=True)
    assert row == n and positive_offset == training_positive_count and offsets[-1] == m
    for array in (timestamps, split, offsets, candidates, labels, history, users, impression_ids, positive_rows, positive_items):
        array.flush()
    np.save(output / "news_first_seen.npy", first_seen)
    np.save(output / "news_train_clicks.npy", train_clicks)
    np.save(output / "news_train_exposures.npy", train_exposures)
    np.save(output / "news_train_observed.npy", train_seen)
    feature_manifest = build_news_features(news, train_seen, output, components=components,
        max_features=max_features, min_df=min_df, threads=threads)
    for which in range(3):
        stats = split_stats[str(which)]
        mask = split_values == which
        stats.update({"users": len(split_users[which]), "candidate_news": len(split_news[which]),
                      "min_timestamp": int(timestamps[mask].min()), "max_timestamp": int(timestamps[mask].max()),
                      "positive_rate": stats["positives"] / stats["candidates"]})
    manifest = {"dataset": "MIND-large", "dataset_version": "upstream Recommenders public release",
        "rows": n, "impressions": n, "candidates": m, "positives": scan["positives"],
        "users": len(set.union(*split_users)), "news": len(ids) - 1, "news_including_padding": len(ids),
        "official_train_rows": scan["rows_by_source"]["train"], "official_dev_rows": scan["rows_by_source"]["dev"],
        "train_cutoff_timestamp": cutoff, "splits": split_stats,
        "split_names": {"0": "train", "1": "internal_validation", "2": "locked_official_dev"},
        "internal_split": "official train, first ~85% by timestamp; ties kept together",
        "test_policy": "official dev full candidate sets, excluded from fitting and hyperparameter selection",
        "history_length": history_length, "history_padding": "right; index 0", "history_truncation": "last 20 clicked news",
        "history_source": "provided request-time history, ordered by time; no reconstruction from held-out labels",
        "timestamp_policy": "source has no timezone; interpreted as UTC only for a deterministic epoch coordinate",
        "news_first_seen_policy": "minimum request time observing ID in history/candidates; availability proxy, not publication time",
        "unobserved_news_first_seen": int(MAX_TIME), "user_id_encoding": "integer suffix of source U-prefixed ID",
        "impression_id_policy": "original numeric ID, may repeat across train/dev; global array row is unique",
        "news_id_policy": "union catalog index only, not a learned item-ID feature",
        "features": feature_manifest,
        "data_quality": {"missing_catalog_news": len(missing), "catalog_conflicts_kept_train_version": changed_records,
                         "empty_history_impressions": empty_history, "duplicate_candidate_impressions": same_impression_repeats},
        "processing_seconds": time.perf_counter() - started}
    receipts = []
    for source in ("train", "dev"):
        receipt = raw / f"{source}_download.json"
        if receipt.exists():
            receipts.append(json.loads(receipt.read_text(encoding="utf-8")))
        else:
            receipts.append({"source_split": source, "locally_provided": True,
                "files": {name: {"bytes": (raw / source / name).stat().st_size,
                                    "sha256": sha256(raw / source / name)} for name in ("news.tsv", "behaviors.tsv")}})
    _json(output / "provenance.json", {"dataset": "MIND-large", "upstream_download_code": SOURCE_URL,
        "terms": TERMS_URL, "use": "non-commercial research/testing/personal experimentation",
        "redistribution": "Do not redistribute raw dataset; publish code and aggregate results only", "downloads": receipts})
    _json(output / "manifest.json", manifest)
    del counts, positive_counts, scan_timestamps
    for name in ("counts", "positives", "timestamps"):
        (output / f".scan_{name}.bin").unlink()
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    prepare(args.raw, args.output, download=args.download, threads=args.threads)


if __name__ == "__main__":
    main()
