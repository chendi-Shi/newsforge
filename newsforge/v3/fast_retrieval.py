"""Exact popular-only serving specialization of the frozen v2 generator.

The selected v2 recipe is a single popularity route. Its order depends on an
immutable hourly snapshot, not on the user encoder. This implementation keeps
that order in a bounded cache and filters only a provably sufficient prefix.
Every CandidateBatch field keeps the v2 contract. Other recipes delegate to
v2; no model, data, cutoff, ranking policy or quality parameter is changed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import threading

import numpy as np

from newsforge.v2.retrieval import (
    CHANNEL_BITS, POPULARITY_POLICY, CandidateBatch, CandidateGenerator,
    canonical_hash, rerank_order, sha256_file,
)


@dataclass(frozen=True)
class _PopularCatalogue:
    visible_count: int
    recent_count: int
    global_order: np.ndarray
    recent_order: np.ndarray
    recent_membership: np.ndarray

    @property
    def nbytes(self):
        return sum(array.nbytes for array in
                   (self.global_order, self.recent_order, self.recent_membership))


class FastCandidateGenerator(CandidateGenerator):
    """API-compatible, thread-safe generator with a popular-only fast path.

    Inputs have the same immutable-artifact lifetime as CandidateGenerator.
    The instance owns its feature/embedding mappings, and close is idempotent;
    a caller must not share those mappings with another owning generator.
    TemporalStore is borrowed and remains the engine's responsibility.

    Hourly entries contain only read-only ranking/membership arrays. History,
    selected IDs and scores are never cached. A reentrant lock protects both
    cache state and mapping lifetime, including calls to the v2 fallback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fast_lock = threading.RLock()
        self._fast_closed = False
        self._popular_cache = OrderedDict()
        self._popular_hits = 0
        self._popular_misses = 0

    @property
    def version(self):
        return canonical_hash({
            "reference_generator_version": super().version,
            "optimization_source_sha256": sha256_file(__file__),
            "implementation": "exact_popular_prefix_v1",
        })

    def cache_info(self):
        """Return diagnostics without exposing mutable cached arrays."""
        with self._fast_lock:
            return {"entries": len(self._popular_cache), "capacity": self._cache_size,
                    "array_bytes": sum(value.nbytes for value in self._popular_cache.values()),
                    "hits": self._popular_hits, "misses": self._popular_misses,
                    "closed": self._fast_closed}

    def clear_cache(self):
        """Discard derived state, for controlled cold/warm measurements."""
        with self._fast_lock:
            self._popular_cache.clear()
            self._cache.clear()
            self._query_routes.clear()
            self._route_query_key = None

    def _popular_catalogue(self, snapshot, config):
        key = (int(snapshot.index), int(snapshot.cutoff), int(snapshot.as_of_start),
               config.recent_hours, config.freeze_feedback)
        if key in self._popular_cache:
            self._popular_hits += 1
            self._popular_cache.move_to_end(key)
            return self._popular_cache[key]

        clicks, exposures = np.asarray(snapshot.clicks), np.asarray(snapshot.exposures)
        if (clicks.shape != self.first_seen.shape or exposures.shape != self.first_seen.shape
                or not np.isfinite(clicks).all() or not np.isfinite(exposures).all()
                or np.any(clicks < 0) or np.any(exposures < 0)):
            raise ValueError("invalid temporal popularity statistics")
        visible_mask = (self.first_seen >= 0) & (self.first_seen < int(snapshot.cutoff))
        if len(visible_mask):
            visible_mask[0] = False
        visible = np.flatnonzero(visible_mask)
        recent_membership = visible_mask.copy()
        if config.recent_hours is not None:
            recent_membership &= (
                self.first_seen >= int(snapshot.as_of_start) - config.recent_hours * 3600)
        recent = np.flatnonzero(recent_membership)

        def ordered(ids):
            return ids[np.lexsort((ids, -self.first_seen[ids], -exposures[ids], -clicks[ids]))]

        global_order, recent_order = ordered(visible), ordered(recent)
        for array in (global_order, recent_order, recent_membership):
            array.setflags(write=False)
        value = _PopularCatalogue(len(visible), len(recent), global_order,
                                  recent_order, recent_membership)
        self._popular_misses += 1
        self._popular_cache[key] = value
        if len(self._popular_cache) > self._cache_size:
            self._popular_cache.popitem(last=False)
        return value

    @staticmethod
    def _unseen_prefix(order, blocked, count):
        # At most len(blocked) entries of a unique order can be excluded.
        # Therefore this prefix always contains the same first `count` unseen
        # entries as filtering the entire catalogue, even for long histories.
        prefix = order[:count + len(blocked)]
        if len(blocked):
            prefix = prefix[~np.isin(prefix, blocked, assume_unique=True)]
        return prefix[:count]

    def retrieve(self, history_ids, as_of, *, config=None, reuse_query_routes=False):
        with self._fast_lock:
            if self._fast_closed:
                raise RuntimeError("candidate generator is closed")
            config = config or self.config
            if not isinstance(reuse_query_routes, bool):
                raise ValueError("reuse_query_routes must be a boolean")
            if config.channels != ("popular",):
                # Serialize the original evaluator's single-query route cache
                # as well. Its semantics and opt-in behavior stay unchanged.
                return super().retrieve(history_ids, as_of, config=config,
                                        reuse_query_routes=reuse_query_routes)
            self._query_routes.clear()
            self._route_query_key = None
            return self._retrieve_popular(history_ids, as_of, config)

    def _retrieve_popular(self, history_ids, as_of, config):
        if not isinstance(as_of, (int, np.integer)) or isinstance(as_of, (bool, np.bool_)):
            raise ValueError("as_of must be an integer timestamp")
        as_of = int(as_of)
        history = np.asarray(history_ids)
        if history.ndim != 1 or (len(history) and not np.issubdtype(history.dtype, np.integer)):
            raise ValueError("history IDs must be a vector of integer catalogue indices")
        history = history.astype(np.int64)
        if np.any(history < 0) or np.any(history >= len(self.first_seen)):
            raise ValueError("history contains an OOV index")
        history = history[history > 0]
        if np.any(self.first_seen[history] > as_of):
            raise ValueError("history contains an article unavailable at request time")
        snapshot = (self.temporal.frozen_snapshot(as_of) if config.freeze_feedback
                    else self.temporal.snapshot(as_of))
        training_cutoff = self.encoder.metadata.get("train_cutoff_timestamp")
        if training_cutoff is not None and snapshot.cutoff <= int(training_cutoff):
            raise ValueError("snapshot cutoff must be strictly after the encoder training cutoff")
        catalogue = self._popular_catalogue(snapshot, config)
        seen = np.unique(history)
        popular = self._unseen_prefix(catalogue.recent_order, seen, config.route_size)
        # Use Python float arithmetic as in v2. Very large valid rank constants
        # can round adjacent votes to the same float, requiring v2's ID tie
        # break rather than assuming strictly decreasing reciprocal scores.
        route_scores = np.asarray([1. / (config.rank_constant + rank)
                                   for rank in range(1, len(popular) + 1)], dtype=np.float64)
        if np.all(route_scores[:-1] > route_scores[1:]):
            positions = np.arange(min(len(popular), config.pool_size))
        else:
            positions = rerank_order(popular, route_scores)[:config.pool_size]
        selected = popular[positions].copy()
        fused_count = len(selected)
        fallback = np.empty(0, dtype=np.int64)
        if len(selected) < config.pool_size:
            blocked = np.union1d(seen, selected)
            fallback = self._unseen_prefix(catalogue.global_order, blocked,
                                           config.pool_size - len(selected)).copy()
            selected = np.concatenate((selected, fallback))
        scores = np.zeros(len(selected), dtype=np.float64)
        scores[:fused_count] = route_scores[positions]
        source_mask = np.full(len(selected), CHANNEL_BITS["popular"], dtype=np.uint8)
        source_mask[fused_count:] = CHANNEL_BITS["fallback"]
        sources = {"popular": popular.astype(int).tolist()}
        if len(fallback):
            sources["fallback"] = fallback.astype(int).tolist()
        # Preserve v2's public history metadata without evaluating the unused
        # content vector or neural user encoder.
        hf = np.asarray(self.features[history[-20:]], dtype=np.float32)
        mask = np.any(hf != 0, axis=-1)
        metadata = {
            "config": asdict(config), "config_version": config.version,
            "strict_catalogue_rule": "first_seen < snapshot.cutoff; exclude every supplied history ID",
            "recency_reference": "snapshot.as_of_start minus first observed time; not publication age",
            "visible_catalogue_count": catalogue.visible_count,
            "recent_unseen_catalogue_count": catalogue.recent_count - int(catalogue.recent_membership[seen].sum()),
            "route_counts": {name: len(values) for name, values in sources.items()},
            "union_count_before_truncation": len(popular), "fallback_count": len(fallback),
            "zero_query_channels": [], "empty_usable_history": not bool(mask.any()),
            "popularity_order": POPULARITY_POLICY,
            "feedback_source": "frozen_training" if config.freeze_feedback else "lagged_hourly_EMA",
            "history_length_used": int(mask.sum()), "history_events_encoded": min(len(history), 20),
            "snapshot_as_of_start": int(snapshot.as_of_start),
        }
        return CandidateBatch(selected, scores, source_mask, sources, int(snapshot.index),
                              int(snapshot.cutoff), as_of,
                              max(0, config.pool_size - len(selected)), metadata)

    def retrieve_batch(self, requests, *, config=None):
        """Evaluate 1–32 (history IDs, timestamp) pairs with shared hourly state."""
        if not 1 <= len(requests) <= 32:
            raise ValueError("batch size must be between 1 and 32")
        with self._fast_lock:
            return [self.retrieve(history, as_of, config=config) for history, as_of in requests]

    def close(self):
        with self._fast_lock:
            if not self._fast_closed:
                self._fast_closed = True
                self._popular_cache.clear()
                super().close()
