"""Read-only audit of the materialized MIND arrays and train-only statistics.

This preserves the checks used for reports/data_validation.json. It does not
download data, fit features, select models, or evaluate recommendation scores.
The materialization timing diagnostic is a separate historical measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


ARRAY_NAMES = (
    "all_offsets", "all_candidates", "all_labels", "all_history",
    "all_timestamps", "all_split", "all_users", "all_impression_ids",
    "train_positive_rows", "train_positive_items", "news_features",
    "news_first_seen", "news_train_clicks", "news_train_exposures",
    "news_train_observed",
)


def require(condition, message):
    """Keep checks active even if Python is run with optimization enabled."""
    if not bool(condition):
        raise ValueError(message)


def audit_data(data: Path, *, chunk_rows=50000):
    """Audit every materialized candidate; memory grows with one row chunk."""
    require(chunk_rows > 0, "chunk_rows must be positive")
    started = time.perf_counter()
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    arrays = {name: np.load(data / f"{name}.npy", mmap_mode="r") for name in ARRAY_NAMES}
    n, news_count = manifest["rows"], manifest["news_including_padding"]
    offsets, candidates, labels = (arrays[key] for key in ("all_offsets", "all_candidates", "all_labels"))
    split, history, features = (arrays[key] for key in ("all_split", "all_history", "news_features"))
    require(offsets.shape == (n + 1,), "offset shape mismatch")
    require(int(offsets[0]) == 0 and int(offsets[-1]) == manifest["candidates"], "offset endpoints mismatch")
    require(candidates.shape == labels.shape == (manifest["candidates"],), "candidate shape mismatch")
    require(history.shape == (n, manifest["history_length"]), "history shape mismatch")
    require(features.shape == (news_count, manifest["features"]["dimensions"]), "feature shape mismatch")
    expected_dtypes = {
        "all_offsets": np.int64, "all_candidates": np.int32, "all_labels": np.uint8,
        "all_history": np.int32, "all_timestamps": np.int64, "all_split": np.uint8,
        "all_users": np.int32, "all_impression_ids": np.int64,
        "train_positive_rows": np.int32, "train_positive_items": np.int32,
        "news_features": np.float32, "news_first_seen": np.int64,
        "news_train_clicks": np.int64, "news_train_exposures": np.int64,
        "news_train_observed": np.bool_,
    }
    for name, dtype in expected_dtypes.items():
        require(arrays[name].dtype == dtype, f"dtype mismatch: {name}")
    for name in ("all_timestamps", "all_split", "all_users", "all_impression_ids"):
        require(arrays[name].shape == (n,), f"row-array shape mismatch: {name}")
    for name in ("news_first_seen", "news_train_clicks", "news_train_exposures", "news_train_observed"):
        require(arrays[name].shape == (news_count,), f"news-array shape mismatch: {name}")

    computed = {str(k): {"impressions": 0, "candidates": 0, "positives": 0} for k in range(3)}
    clicks = np.zeros(news_count, dtype=np.int64)
    exposures = np.zeros(news_count, dtype=np.int64)
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        counts = np.diff(offsets[start:end + 1])
        require(np.all(counts > 0), f"empty/non-monotonic impression at row block {start}")
        left, right = int(offsets[start]), int(offsets[end])
        items, targets = candidates[left:right], labels[left:right]
        require(np.all((items > 0) & (items < news_count)), f"candidate ID out of range at row block {start}")
        require(np.all(targets <= 1), f"nonbinary label at row block {start}")
        block_history = history[start:end]
        require(np.all((block_history >= 0) & (block_history < news_count)), "history ID out of range")
        require(not np.any((block_history[:, :-1] == 0) & (block_history[:, 1:] != 0)), "history is not right-padded")
        block_split = split[start:end]
        require(np.all(block_split <= 2), "unknown split ID")
        flat_split = np.repeat(block_split, counts)
        for which in range(3):
            mask = flat_split == which
            computed[str(which)]["impressions"] += int(np.count_nonzero(block_split == which))
            computed[str(which)]["candidates"] += int(mask.sum())
            computed[str(which)]["positives"] += int(targets[mask].sum())
        train = flat_split == 0
        exposures += np.bincount(items[train], minlength=news_count)
        clicks += np.bincount(items[train & (targets == 1)], minlength=news_count)
    for which, record in computed.items():
        for key, value in record.items():
            require(manifest["splits"][which][key] == value, f"manifest split {which} {key} mismatch")
    require(sum(item["positives"] for item in computed.values()) == manifest["positives"], "total positives mismatch")
    require(np.array_equal(clicks, arrays["news_train_clicks"]), "train click reconstruction mismatch")
    require(np.array_equal(exposures, arrays["news_train_exposures"]), "train exposure reconstruction mismatch")

    positive_rows, positive_items = arrays["train_positive_rows"], arrays["train_positive_items"]
    require(positive_rows.shape == positive_items.shape == (computed["0"]["positives"],), "positive-index shape mismatch")
    require(np.all((positive_rows >= 0) & (positive_rows < n)), "positive row out of range")
    require(np.all((positive_items > 0) & (positive_items < news_count)), "positive item out of range")
    require(np.all(split[positive_rows] == 0), "held-out row in training positives")
    require(np.array_equal(np.bincount(positive_items, minlength=news_count), clicks), "positive-index click distribution mismatch")
    require(np.isfinite(features).all() and np.all(features[0] == 0), "nonfinite features or nonzero PAD")
    text_dimensions = manifest["features"]["text_dimensions"]
    norms = np.linalg.norm(features[1:, :text_dimensions], axis=1)
    require(np.all(np.isclose(norms, 1, atol=1e-5) | np.isclose(norms, 0, atol=1e-7)), "text L2 normalization mismatch")
    categories = features[1:, text_dimensions:]
    require(np.all((categories == 0) | (categories == 1)) and np.all(categories.sum(axis=1) == 1), "category one-hot mismatch")
    require(arrays["news_first_seen"][0] == np.iinfo(np.int64).max, "PAD first-seen sentinel mismatch")
    require(not arrays["news_train_observed"][0], "PAD included in feature fit mask")
    require(int(arrays["news_train_observed"].sum()) == manifest["features"]["fit_news"], "feature fit count mismatch")
    timestamps = arrays["all_timestamps"]
    require(timestamps[split == 0].max() == manifest["train_cutoff_timestamp"], "cutoff mismatch")
    require(timestamps[split == 0].max() < timestamps[split == 1].min(), "train/validation timestamps overlap")
    require(timestamps[split == 1].max() < timestamps[split == 2].min(), "validation/dev timestamps overlap")
    return {
        "status": "passed", "validation_seconds": time.perf_counter() - started,
        "rows": n, "candidates": int(offsets[-1]), "news": news_count - 1,
        "dimensions": int(features.shape[1]), "split_recomputed": computed,
        "checks": [
            "full candidate IDs and binary labels", "full offsets and history padding",
            "full per-split row/candidate/positive totals",
            "full per-news training exposure and click count reconstruction",
            "positive index split and per-news distribution",
            "finite features, text L2 norms, category onehot, zero padding",
            "strict temporal boundaries", "training fit count and PAD first-seen sentinel",
        ],
        "scope": "materialized-array integrity; does not refit text, reread source TSV, or evaluate models",
        "chunk_rows": chunk_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunk-rows", type=int, default=50000)
    parser.add_argument("--output", type=Path, help="Optional new JSON path; an existing file is never overwritten")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new path")
    result = audit_data(args.data, chunk_rows=args.chunk_rows)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
