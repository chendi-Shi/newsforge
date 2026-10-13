"""Causal, NumPy-only multi-route candidate retrieval.

No label arrays or PyTorch are imported at runtime. PyTorch is imported only
by the explicit encoder export/validation command. Every route uses the same
strict observable catalogue cutoff supplied by TemporalStore.snapshot().
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


CHANNEL_BITS = {"popular": 1, "content": 2, "tower": 4, "fallback": 8}
POPULARITY_POLICY = "EMA clicks descending, exposures descending, first_seen descending, item index ascending"


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def rerank_order(ids, scores):
    """One tie policy for offline and HTTP: descending score, ascending ID."""
    ids, scores = np.asarray(ids, dtype=np.int64), np.asarray(scores)
    if ids.ndim != 1 or scores.shape != ids.shape or not np.isfinite(scores).all():
        raise ValueError("IDs and finite scores must be matching vectors")
    return np.lexsort((ids, -scores))


def _top_ids(ids, scores, count):
    if len(ids) <= count:
        positions = np.arange(len(ids))
    else:
        threshold = np.partition(scores, -count)[-count]
        # Include boundary ties before the deterministic ID tiebreak.
        positions = np.flatnonzero(scores >= threshold)
    ordering = rerank_order(ids[positions], scores[positions])[:count]
    return ids[positions[ordering]]


@dataclass(frozen=True)
class CandidateConfig:
    pool_size: int = 100
    route_size: int = 200
    recent_hours: int | None = 48
    channels: tuple[str, ...] = ("popular", "content", "tower")
    freeze_feedback: bool = False
    rank_constant: int = 60

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        if (any(not isinstance(value, int) or isinstance(value, bool)
                for value in (self.pool_size, self.route_size, self.rank_constant))
                or self.pool_size < 1 or self.route_size < self.pool_size or self.rank_constant < 1
                or self.recent_hours not in (None, 24, 48, 72)
                or not self.channels or len(set(self.channels)) != len(self.channels)
                or set(self.channels) - {"popular", "content", "tower"}
                or not isinstance(self.freeze_feedback, bool)):
            raise ValueError("invalid candidate recipe")

    @property
    def version(self):
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class CandidateBatch:
    ids: np.ndarray
    scores: np.ndarray
    source_mask: np.ndarray
    sources: dict[str, list[int]]
    snapshot_index: int
    cutoff: int
    as_of: int
    shortfall: int
    metadata: dict


class NumpyMeanEncoder:
    """The frozen v1 mean user MLP, with PyTorch-compatible float32 masking."""
    def __init__(self, arrays, metadata=None):
        self.arrays = {name: np.asarray(arrays[name], dtype=np.float32)
                       for name in ("w0", "b0", "w2", "b2")}
        w0, b0, w2, b2 = (self.arrays[name] for name in ("w0", "b0", "w2", "b2"))
        if (w0.ndim != 2 or b0.shape != (w0.shape[0],) or w2.ndim != 2
                or w2.shape[1] != w0.shape[0] or b2.shape != (w2.shape[0],)
                or not all(np.isfinite(value).all() for value in self.arrays.values())):
            raise ValueError("invalid NumPy encoder arrays")
        self.feature_dim, self.embedding_dim = w0.shape[1], w2.shape[0]
        self.metadata = dict(metadata or {})

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if metadata.get("npz_sha256") != sha256_file(path):
            raise ValueError("NumPy encoder hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            return cls(arrays, metadata)

    def encode_user(self, history_features, history_mask=None):
        values = np.asarray(history_features, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != self.feature_dim or not np.isfinite(values).all():
            raise ValueError("history must be finite [batch, history, feature_dim]")
        mask = np.any(values != 0, axis=-1)
        if history_mask is not None:
            supplied = np.asarray(history_mask)
            if supplied.dtype != np.bool_ or supplied.shape != mask.shape:
                raise ValueError("history mask must be a matching boolean matrix")
            mask &= supplied
        pooled = np.where(mask[..., None], values, np.float32(0)).sum(axis=1, dtype=np.float32)
        pooled /= np.maximum(mask.sum(axis=1, keepdims=True), 1).astype(np.float32)
        hidden = np.maximum(pooled @ self.arrays["w0"].T + self.arrays["b0"], np.float32(0))
        vectors = hidden @ self.arrays["w2"].T + self.arrays["b2"]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), np.float32(1e-12))
        return vectors * mask.any(axis=1, keepdims=True)


class CandidateGenerator:
    def __init__(self, features, article_embeddings, first_seen, train_clicks, train_exposures,
                 temporal_store, encoder, *, config=None, bindings=None, cache_size=8):
        self.features, self.article_embeddings = features, article_embeddings
        self.first_seen = np.asarray(first_seen, dtype=np.int64)
        self.train_clicks, self.train_exposures = np.asarray(train_clicks), np.asarray(train_exposures)
        self.temporal, self.encoder = temporal_store, encoder
        self.config = config or CandidateConfig()
        self.bindings = dict(bindings or {})
        n = len(features)
        if (features.ndim != 2 or article_embeddings.ndim != 2 or len(article_embeddings) != n
                or self.first_seen.shape != (n,) or self.train_clicks.shape != (n,)
                or self.train_exposures.shape != (n,) or features.shape[1] != encoder.feature_dim
                or article_embeddings.shape[1] != encoder.embedding_dim or cache_size < 1):
            raise ValueError("candidate catalogue/model shapes differ")
        self._norms = np.empty(n, dtype=np.float32)
        for start in range(0, n, 8192):
            block = features[start:start + 8192]
            if not np.isfinite(block).all() or not np.isfinite(article_embeddings[start:start + 8192]).all():
                raise ValueError("catalogue contains non-finite features or embeddings")
            self._norms[start:start + 8192] = np.linalg.norm(block, axis=1)
        self._cache, self._cache_size = OrderedDict(), cache_size
        # Evaluation-only reuse of tiny route outputs for one current query.
        # Eight entries cover two semantic routes x four declared windows.
        self._route_query_key, self._query_routes = None, OrderedDict()

    @classmethod
    def from_paths(cls, processed_dir, artifact_dir, temporal_store, *, encoder_path=None, config=None):
        processed, artifacts = Path(processed_dir), Path(artifact_dir)
        encoder_path = Path(encoder_path or artifacts / "v2" / "user_encoder.npz")
        encoder = NumpyMeanEncoder.load(encoder_path)
        paths = {"features": processed / "news_features.npy", "first_seen": processed / "news_first_seen.npy",
                 "train_clicks": processed / "news_train_clicks.npy", "train_exposures": processed / "news_train_exposures.npy",
                 "embeddings": artifacts / "article_embeddings.npy", "retriever": artifacts / "retriever.pt",
                 "encoder": encoder_path, "encoder_manifest": encoder_path.with_suffix(encoder_path.suffix + ".json"),
                 "data_manifest": processed / "manifest.json",
                 "news_ids": processed / "news_ids.json", "temporal_manifest": Path(temporal_store.path) / "manifest.json"}
        bindings = {name: sha256_file(path) for name, path in paths.items()}
        if encoder.metadata.get("checkpoint_sha256") != bindings["retriever"]:
            raise ValueError("NumPy encoder/retriever binding mismatch")
        if encoder.metadata.get("feature_sha256") != bindings["features"]:
            raise ValueError("NumPy encoder/features binding mismatch")
        if encoder.metadata.get("data_manifest_sha256") != bindings["data_manifest"]:
            raise ValueError("NumPy encoder/data manifest binding mismatch")
        if encoder.metadata.get("article_embeddings_sha256") != bindings["embeddings"]:
            raise ValueError("NumPy encoder/article embeddings binding mismatch")
        return cls(np.load(paths["features"], mmap_mode="r", allow_pickle=False),
                   np.load(paths["embeddings"], mmap_mode="r", allow_pickle=False),
                   np.load(paths["first_seen"], allow_pickle=False),
                   np.load(paths["train_clicks"], allow_pickle=False),
                   np.load(paths["train_exposures"], allow_pickle=False), temporal_store,
                   encoder, config=config, bindings=bindings)

    @property
    def version(self):
        return canonical_hash({"bindings": self.bindings, "config": asdict(self.config),
                               "source_sha256": sha256_file(__file__)})

    def _catalogue(self, snapshot, config):
        key = (int(snapshot.index), int(snapshot.cutoff), config.recent_hours, config.freeze_feedback)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        cutoff = int(snapshot.cutoff)
        visible = np.flatnonzero((np.arange(len(self.first_seen)) > 0)
                                 & (self.first_seen >= 0) & (self.first_seen < cutoff))
        recent = visible if config.recent_hours is None else visible[
            self.first_seen[visible] >= int(snapshot.as_of_start) - config.recent_hours * 3600]
        clicks, exposures = np.asarray(snapshot.clicks), np.asarray(snapshot.exposures)
        if (clicks.shape != self.first_seen.shape or exposures.shape != self.first_seen.shape
                or not np.isfinite(clicks).all() or not np.isfinite(exposures).all()
                or np.any(clicks < 0) or np.any(exposures < 0)):
            raise ValueError("invalid temporal popularity statistics")
        def popular(ids):
            return ids[np.lexsort((ids, -self.first_seen[ids], -exposures[ids], -clicks[ids]))]
        value = (visible, recent, popular(visible), popular(recent))
        self._cache[key] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return value

    def _dot(self, matrix, ids, query, *, normalize_features=False):
        scores = np.empty(len(ids), dtype=np.float32)
        for start in range(0, len(ids), 8192):
            selected = ids[start:start + 8192]
            values = matrix[selected] @ query
            if normalize_features:
                values /= np.maximum(self._norms[selected], np.float32(1e-12))
            scores[start:start + len(selected)] = values
        return scores

    def retrieve(self, history_ids, as_of, *, config=None, reuse_query_routes=False):
        """Retrieve candidates; route reuse is opt-in for paired ablation sweeps.

        Reuse shares only semantic top IDs across recipes for the exact same
        request. It does not change popularity, fusion, fallback or metadata.
        The default independent-request path never consumes or fills this cache.
        """
        config = config or self.config
        if not isinstance(reuse_query_routes, bool):
            raise ValueError("reuse_query_routes must be a boolean")
        if not isinstance(as_of, (int, np.integer)) or isinstance(as_of, (bool, np.bool_)):
            raise ValueError("as_of must be an integer timestamp")
        as_of = int(as_of)
        training_cutoff = self.encoder.metadata.get("train_cutoff_timestamp")
        history = np.asarray(history_ids)
        if history.ndim != 1 or (len(history) and not np.issubdtype(history.dtype, np.integer)):
            raise ValueError("history IDs must be a vector of integer catalogue indices")
        history = history.astype(np.int64)
        if np.any(history < 0) or np.any(history >= len(self.first_seen)):
            raise ValueError("history contains an OOV index")
        if reuse_query_routes:
            route_query_key = (as_of, tuple(history.tolist()))
            if route_query_key != self._route_query_key:
                self._query_routes.clear()
                self._route_query_key = route_query_key
        else:
            self._query_routes.clear()
            self._route_query_key = None
        history = history[history > 0]
        if np.any(self.first_seen[history] > as_of):
            raise ValueError("history contains an article unavailable at request time")
        snapshot = (self.temporal.frozen_snapshot(as_of) if config.freeze_feedback
                    else self.temporal.snapshot(as_of))
        if training_cutoff is not None and snapshot.cutoff <= int(training_cutoff):
            raise ValueError("snapshot cutoff must be strictly after the encoder training cutoff")
        visible, recent, global_popular, recent_popular = self._catalogue(snapshot, config)
        def unseen(ids):
            return ids[~np.isin(ids, history)] if len(history) else ids
        recent, popular_ids = unseen(recent), unseen(recent_popular)
        hf = np.asarray(self.features[history[-20:]], dtype=np.float32)
        mask = np.any(hf != 0, axis=-1)
        pooled = hf.sum(axis=0, dtype=np.float32) / np.float32(max(int(mask.sum()), 1))
        content_norm = float(np.linalg.norm(pooled))
        query = self.encoder.encode_user(hf[None], mask[None])[0]
        tower_norm = float(np.linalg.norm(query))
        sources, zero_channels = {}, []
        for channel in config.channels:
            if channel == "popular":
                ids = popular_ids[:config.route_size]
            else:
                norm = content_norm if channel == "content" else tower_norm
                if norm <= 1e-12:
                    zero_channels.append(channel)
                    continue
                route_key = (channel, config.recent_hours, config.route_size)
                if reuse_query_routes and route_key in self._query_routes:
                    self._query_routes.move_to_end(route_key)
                    ids = self._query_routes[route_key]
                else:
                    matrix = self.features if channel == "content" else self.article_embeddings
                    vector = pooled / np.float32(norm) if channel == "content" else query
                    scores = self._dot(matrix, recent, vector, normalize_features=channel == "content")
                    ids = _top_ids(recent, scores, config.route_size)
                    if reuse_query_routes:
                        self._query_routes[route_key] = ids.copy()
                        if len(self._query_routes) > 8:
                            self._query_routes.popitem(last=False)
            sources[channel] = ids.astype(int).tolist()
        fused, masks = {}, {}
        for channel, ids in sources.items():
            for rank, item in enumerate(ids, 1):
                fused[item] = fused.get(item, 0.) + 1. / (config.rank_constant + rank)
                masks[item] = masks.get(item, 0) | CHANNEL_BITS[channel]
        union = np.asarray(list(fused), dtype=np.int64)
        union_scores = np.asarray([fused[int(item)] for item in union], dtype=np.float64)
        selected = union[rerank_order(union, union_scores)[:config.pool_size]].tolist()
        fallback = []
        if len(selected) < config.pool_size:
            blocked = set(selected) | set(history.tolist())
            for item in global_popular:
                item = int(item)
                if item not in blocked:
                    selected.append(item)
                    fallback.append(item)
                    blocked.add(item)
                    if len(selected) == config.pool_size:
                        break
        if fallback:
            sources["fallback"] = fallback
            for rank, item in enumerate(fallback, 1):
                masks[item] = CHANNEL_BITS["fallback"]
                # Backfilled items follow every fused candidate; this score is
                # deliberately not another RRF vote from a disabled component.
                fused[item] = 0.
        ids = np.asarray(selected, dtype=np.int64)
        metadata = {"config": asdict(config), "config_version": config.version,
                    "strict_catalogue_rule": "first_seen < snapshot.cutoff; exclude every supplied history ID",
                    "recency_reference": "snapshot.as_of_start minus first observed time; not publication age",
                    "visible_catalogue_count": len(visible), "recent_unseen_catalogue_count": len(recent),
                    "route_counts": {name: len(values) for name, values in sources.items()},
                    "union_count_before_truncation": len(union), "fallback_count": len(fallback),
                    "zero_query_channels": zero_channels, "empty_usable_history": not bool(mask.any()),
                    "popularity_order": POPULARITY_POLICY,
                    "feedback_source": "frozen_training" if config.freeze_feedback else "lagged_hourly_EMA",
                    "history_length_used": int(mask.sum()),
                    "history_events_encoded": min(len(history), 20),
                    "snapshot_as_of_start": int(snapshot.as_of_start)}
        return CandidateBatch(ids, np.asarray([fused[int(item)] for item in ids], dtype=np.float64),
                              np.asarray([masks[int(item)] for item in ids], dtype=np.uint8), sources,
                              int(snapshot.index), int(snapshot.cutoff), as_of,
                              max(0, config.pool_size - len(ids)), metadata)

    def close(self):
        self._cache.clear()
        self._query_routes.clear()
        self._route_query_key = None
        for array in (self.features, self.article_embeddings):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()


def export_encoder(checkpoint_path, output, processed_dir, article_embeddings_path, *, validation_count=512):
    """Export and validate on fixed internal requests; never read label arrays."""
    import torch
    from threadpoolctl import threadpool_limits
    from newsforge.models import TwoTower
    torch.set_num_threads(1)
    threadpool_limits(1)
    processed_dir, output = Path(processed_dir), Path(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"]["user_pooling"] != "mean":
        raise ValueError("NumPy export supports the frozen mean user tower only")
    data_hash = sha256_file(processed_dir / "manifest.json")
    feature_hash = sha256_file(processed_dir / "news_features.npy")
    if (checkpoint.get("data_manifest_sha256") != data_hash
            or checkpoint.get("training_cache_signature", {}).get("features_sha256") != feature_hash):
        raise ValueError("frozen checkpoint/data/features provenance mismatch")
    model = TwoTower(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    arrays = {name: checkpoint["state_dict"][key].cpu().numpy() for name, key in
              (("w0", "user_encoder.0.weight"), ("b0", "user_encoder.0.bias"),
               ("w2", "user_encoder.2.weight"), ("b2", "user_encoder.2.bias"))}
    encoder = NumpyMeanEncoder(arrays)
    split = np.load(processed_dir / "all_split.npy", mmap_mode="r", allow_pickle=False)
    eligible = np.flatnonzero(split == 1)
    if validation_count < 1 or len(eligible) < validation_count:
        raise ValueError("insufficient internal validation rows for export check")
    rows = np.sort(np.random.default_rng(2026).choice(eligible, validation_count, replace=False))
    features = np.load(processed_dir / "news_features.npy", mmap_mode="r", allow_pickle=False)
    history = np.load(processed_dir / "all_history.npy", mmap_mode="r", allow_pickle=False)
    max_error, all_passed = 0., True
    for start in range(0, len(rows), 64):
        values = np.asarray(features[history[rows[start:start + 64]]], dtype=np.float32)
        mask = np.any(values != 0, axis=-1)
        with torch.inference_mode():
            reference = model.encode_user(torch.from_numpy(values), torch.from_numpy(mask)).numpy()
        actual = encoder.encode_user(values, mask)
        max_error = max(max_error, float(np.max(np.abs(actual - reference), initial=0)))
        all_passed &= bool(np.allclose(actual, reference, atol=1e-6, rtol=1e-5))
    if not all_passed:
        raise ValueError(f"NumPy user-encoder equivalence failed: max_abs_error={max_error}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    metadata = {"format_version": 1, "checkpoint_sha256": sha256_file(checkpoint_path),
                "npz_sha256": sha256_file(output), "feature_sha256": feature_hash,
                "data_manifest_sha256": data_hash,
                "article_embeddings_sha256": sha256_file(article_embeddings_path),
                "run_id": checkpoint["run_id"], "train_cutoff_timestamp": checkpoint["train_cutoff_timestamp"],
                "source_sha256": sha256_file(__file__), "config": checkpoint["config"],
                "validation": {"split": "internal_validation", "seed": 2026, "rows": len(rows),
                               "row_indices_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
                               "max_absolute_error": max_error, "atol": 1e-6, "rtol": 1e-5,
                               "passed": all_passed, "labels_read": False, "threads": 1}}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export-encoder"])
    parser.add_argument("--checkpoint", default="artifacts/retriever.pt")
    parser.add_argument("--output", default="artifacts/v2/user_encoder.npz")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--article-embeddings", default="artifacts/article_embeddings.npy")
    parser.add_argument("--validation-count", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(export_encoder(args.checkpoint, args.output, args.processed_dir,
                                   args.article_embeddings, validation_count=args.validation_count), indent=2))


if __name__ == "__main__":
    main()
