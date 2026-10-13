"""Memory-mapped impression store; training caches never inspect held-out labels."""
import json
import hashlib
from pathlib import Path
import time

import numpy as np
from numpy.lib.format import open_memmap


class Store:
    def __init__(self, path="data/processed"):
        self.path = Path(path)
        self.features = np.load(self.path / "news_features.npy", mmap_mode="r")
        for name in ["history", "candidates", "labels", "offsets", "timestamps", "users", "split"]:
            setattr(self, name, np.load(self.path / f"all_{name}.npy", mmap_mode="r"))
        self.first_seen = np.load(self.path / "news_first_seen.npy", mmap_mode="r")
        self.clicks = np.load(self.path / "news_train_clicks.npy")
        self.exposures = np.load(self.path / "news_train_exposures.npy")
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.news_ids = json.loads((self.path / "news_ids.json").read_text(encoding="utf-8"))

    def candidates_for(self, row):
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        return self.candidates[start:end], self.labels[start:end]

    def padded_candidates(self, rows):
        lengths = self.offsets[rows + 1] - self.offsets[rows]
        width = int(lengths.max())
        positions = self.offsets[rows, None] + np.arange(width)[None]
        mask = np.arange(width)[None] < lengths[:, None]
        positions = np.minimum(positions, len(self.candidates) - 1)
        return np.where(mask, self.candidates[positions], 0).astype(np.int64), mask

    def ensure_training_cache(self, pool_size=16):
        """Build a shared exposed-negative cache without per-impression Python.

        Groups with >= pool_size eligible exposure occurrences are sampled
        uniformly without replacement via independent random keys. Short groups
        use independent uniform draws with replacement. Repeated source IDs
        remain repeated occurrences, matching numpy.choice on the old arrays.
        Any ID clicked in an impression is excluded from every negative slot
        for that impression, even if another occurrence has label zero.
        """
        if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size < 1:
            raise ValueError("pool_size must be a positive integer")
        marker = self.path / "training_cache.json"
        def digest(path):
            result = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    result.update(chunk)
            return result.hexdigest()
        max_chunk_exposures = 400000
        signature = {"cache_version": 3, "pool_size": pool_size,
                     "sampler": "grouped_random_keys_v1", "max_chunk_exposures": max_chunk_exposures,
                     "data_manifest_sha256": digest(self.path / "manifest.json"),
                     "features_sha256": digest(self.path / "news_features.npy")}
        if marker.exists():
            saved = json.loads(marker.read_text())
            if any(saved.get(key) != value for key, value in signature.items()):
                raise ValueError("Training cache does not match current data/configuration; rebuild in a fresh processed directory.")
            self.pool = np.load(self.path / "negative_pool.npy", mmap_mode="r")
            self.positive_rows = np.load(self.path / "positive_rows.npy", mmap_mode="r")
            self.positive_items = np.load(self.path / "positive_items.npy", mmap_mode="r")
            self.mean_history = np.load(self.path / "mean_history.npy", mmap_mode="r")
            if (self.pool.shape != (len(self.split), pool_size)
                    or self.mean_history.shape != (len(self.split), self.features.shape[1])
                    or self.positive_rows.shape != self.positive_items.shape):
                raise ValueError("Training cache arrays do not match the declared shapes")
            return
        start_time = time.perf_counter()
        rows = np.flatnonzero(self.split == 0)
        lengths = np.asarray(self.offsets[rows + 1] - self.offsets[rows], dtype=np.int64)
        if np.any(lengths < 0):
            raise ValueError("Candidate offsets must be nondecreasing")
        cumulative = np.r_[0, np.cumsum(lengths)]
        pool = open_memmap(self.path / "negative_pool.npy", mode="w+", dtype="int32",
                           shape=(len(self.split), pool_size))
        pool[:] = 0
        positive_rows, positive_items = [], []
        rng = np.random.default_rng(20261010)
        skipped, chunks, cursor, next_progress = 0, 0, 0, 250000
        catalog_size = len(self.features)
        try:
            while cursor < len(rows):
                end = min(len(rows), max(cursor + 1, int(np.searchsorted(
                    cumulative, cumulative[cursor] + max_chunk_exposures, side="right")) - 1))
                selected_rows, sizes = rows[cursor:end], lengths[cursor:end]
                starts = np.r_[0, np.cumsum(sizes)[:-1]]
                groups = np.repeat(np.arange(len(selected_rows), dtype=np.int32), sizes)
                positions = np.repeat(np.asarray(self.offsets[selected_rows]) - starts, sizes)
                positions += np.arange(len(groups), dtype=np.int64)
                ids = np.asarray(self.candidates[positions], dtype=np.int64)
                labels = np.asarray(self.labels[positions])
                if np.any((ids <= 0) | (ids >= catalog_size)):
                    raise ValueError("Logged candidates must be known non-PAD article IDs")
                if np.any((labels != 0) & (labels != 1)):
                    raise ValueError("Logged candidates must have binary labels")
                # Local group IDs keep keys bounded while distinguishing the
                # same article in different impressions. Never scan held-out labels.
                keys = groups.astype(np.int64) * catalog_size + ids
                positive_mask = labels == 1
                negative_positions = np.flatnonzero(labels == 0)
                conflicting = np.isin(keys[negative_positions], keys[positive_mask],
                                      assume_unique=False, kind="sort")
                negative_positions = negative_positions[~conflicting]
                positive_counts = np.bincount(groups[positive_mask], minlength=len(selected_rows))
                negative_counts = np.bincount(groups[negative_positions], minlength=len(selected_rows))
                valid_rows = (positive_counts > 0) & (negative_counts > 0)
                skipped += int((~valid_rows).sum())
                negative_positions = negative_positions[valid_rows[groups[negative_positions]]]
                negative_ids, negative_groups = ids[negative_positions], groups[negative_positions]
                counts = np.bincount(negative_groups, minlength=len(selected_rows))
                short_rows = np.flatnonzero((counts > 0) & (counts < pool_size))
                if len(short_rows):
                    negative_starts = np.cumsum(counts) - counts
                    draws = rng.integers(0, counts[short_rows, None],
                                        size=(len(short_rows), pool_size))
                    pool[selected_rows[short_rows]] = negative_ids[negative_starts[short_rows, None] + draws]
                long_rows = np.flatnonzero(counts >= pool_size)
                if len(long_rows):
                    long_mask = counts[negative_groups] >= pool_size
                    long_ids, long_groups = negative_ids[long_mask], negative_groups[long_mask]
                    order = np.lexsort((rng.random(len(long_ids)), long_groups))
                    long_starts = np.cumsum(counts[long_rows]) - counts[long_rows]
                    chosen = order[long_starts[:, None] + np.arange(pool_size)[None, :]]
                    pool[selected_rows[long_rows]] = long_ids[chosen]
                training_positive_mask = positive_mask & valid_rows[groups]
                positive_rows.append(selected_rows[groups[training_positive_mask]].astype(np.int32))
                positive_items.append(ids[training_positive_mask].astype(np.int32))
                chunks += 1
                cursor = end
                if cursor >= next_progress or cursor == len(rows):
                    print(f"negative cache {cursor:,}/{len(rows):,} impressions ({chunks} vectorized chunks)", flush=True)
                    next_progress = cursor + 250000
            positive_row_values = np.concatenate(positive_rows) if positive_rows else np.empty(0, dtype=np.int32)
            positive_item_values = np.concatenate(positive_items) if positive_items else np.empty(0, dtype=np.int32)
            np.save(self.path / "positive_rows.npy", positive_row_values)
            np.save(self.path / "positive_items.npy", positive_item_values)
            pool.flush()
        finally:
            pool._mmap.close()
        sampling_seconds = time.perf_counter() - start_time
        mean = open_memmap(self.path / "mean_history.npy", mode="w+", dtype="float32",
                          shape=(len(self.split), self.features.shape[1]))
        # Bound the gathered [B,H,D] tensor to roughly 64 MiB.
        mean_chunk_rows = min(8192, max(1, (64 * 2**20) // max(
            1, self.history.shape[1] * self.features.shape[1] * self.features.dtype.itemsize)))
        try:
            for start in range(0, len(self.split), mean_chunk_rows):
                history = self.history[start:start + mean_chunk_rows]
                features = self.features[history]
                valid = np.any(features != 0, axis=-1)
                mean[start:start + mean_chunk_rows] = features.sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)
            mean.flush()
        finally:
            mean._mmap.close()
        marker.write_text(json.dumps({**signature, "seed": 20261010,
            "train_impressions": len(rows), "positive_training_examples": len(positive_row_values),
            "skipped_no_positive_or_negative": skipped, "seconds": time.perf_counter() - start_time,
            "sampling_seconds": sampling_seconds, "sampling_chunks": chunks,
            "mean_history_chunk_rows": mean_chunk_rows,
            "negative_policy": "Only label=0 in the same training impression; exclude every ID with any positive occurrence; no held-out labels.",
            "sampling_policy": "Uniform exposure-occurrence sample without replacement when count >= pool_size; otherwise independent draws with replacement. Fixed seed and chunk protocol shared across arms.",
            "rng_change": "RNG consumption differs from the former per-row numpy.choice implementation; the sampling distribution and eligible item policy are unchanged."}, indent=2))
        self.ensure_training_cache(pool_size)

    def history_batch(self, rows, mean=False):
        if mean:
            values = np.asarray(self.mean_history[rows])[:, None, :]
            return values, np.any(values != 0, axis=-1)
        values = self.features[self.history[rows]]
        return values, np.any(values != 0, axis=-1)
