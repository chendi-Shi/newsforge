"""Frozen MIND residual inference and independent causal replay audit."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from newsforge.v2.ranker import content_features
from newsforge.v2.temporal import _MemoryMonitor
from newsforge.v3.residual import PairwiseResidual

from .common import (DECAY, HOUR, METHODS, START, check_files, check_previous_bindings,
                     file_records, read_json, sha256, stage, write_new_json)
from .prepare import connect_new, connect_readonly


def load_models(repo):
    return {variant: [PairwiseResidual.load(Path(repo) / f"artifacts/v3/models/{variant}-seed{seed}.npz")
                      for seed in (17, 29, 43)]
            for variant in ("full13", "no_category", "no_recent")}


def user_history(connection, user, timestamp):
    return np.fromiter((row[0] for row in connection.execute("""SELECT item FROM events
        WHERE user=? AND timestamp<? ORDER BY timestamp,member COLLATE BINARY,line""", (int(user), int(timestamp)))),
                       dtype=np.int64)


class HourlyEMA:
    """One monotonic state cursor shared by every query and method."""
    def __init__(self, connection, n_items, log=None):
        self.state = np.zeros(n_items, dtype=np.float64)
        self.cutoff = START
        began, last_log = time.perf_counter(), [time.perf_counter()]
        def heartbeat():
            now = time.perf_counter()
            if log is not None and now - last_log[0] >= 15:
                log("optimized_grouping_progress", elapsed_seconds=now - began)
                last_log[0] = now
            return 0
        connection.set_progress_handler(heartbeat, 1000000)
        try:
            self.cursor = iter(connection.execute("""SELECT timestamp/3600,item,COUNT(*) FROM events
                GROUP BY timestamp/3600,item ORDER BY timestamp/3600,item"""))
        finally:
            connection.set_progress_handler(None, 0)
        self.next_row = next(self.cursor, None)

    def advance(self, cutoff):
        if cutoff < self.cutoff or cutoff % HOUR:
            raise ValueError("EMA cutoff must be hourly and monotonic")
        while self.cutoff < cutoff:
            bucket = self.cutoff // HOUR
            self.state *= DECAY
            while self.next_row is not None and self.next_row[0] == bucket:
                _, item, count = self.next_row
                self.state[item] += count
                self.next_row = next(self.cursor, None)
            if self.next_row is not None and self.next_row[0] < bucket:
                raise ValueError("EMA stream contained an event before its cursor")
            self.cutoff += HOUR
        return self.state


def catalogue_orders(state, first_seen, timestamp):
    top_hour = int(timestamp) // HOUR * HOUR
    cutoff = top_hour - HOUR
    visible = np.flatnonzero((first_seen >= 0) & (first_seen < cutoff))
    visible = visible[visible > 0]
    order = visible[np.lexsort((visible, -first_seen[visible], -state[visible]))]
    recent = order[first_seen[order] >= top_hour - 24 * HOUR]
    return order, recent


def select_pool(global_order, recent_order, history):
    seen = np.unique(history)
    recent = recent_order[~np.isin(recent_order, seen, assume_unique=True)][:200]
    selected = recent[:100]
    recent_selected = len(selected)
    if len(selected) < 100:
        blocked = np.union1d(seen, selected)
        fallback = global_order[~np.isin(global_order, blocked, assume_unique=True)][:100 - len(selected)]
        selected = np.concatenate((selected, fallback))
    return selected, len(selected) - recent_selected


def frozen_features(features, history, pool):
    recent = history[-20:]
    return content_features(features[recent][None, :, :], features[pool][None, :, :],
                            (recent > 0)[None, :], text_dimensions=96)[0, :, :13]


def rank_all(features, history, pool, models):
    m = len(pool)
    original = np.arange(m, dtype=np.int64)
    scores = np.empty((len(METHODS), m), dtype=np.float64)
    scores[0] = -original
    if m == 0:
        return np.empty((len(METHODS), 0), dtype=np.int16), scores
    n = min(len(history), 20)
    matrix = frozen_features(features, history, pool)
    residuals = {variant: np.mean(np.stack([np.clip(model.predict_raw(matrix), -1., 1.)
                                          for model in seed_models]), axis=0)
                 for variant, seed_models in models.items()}
    for index, (variant, gate) in enumerate((
            ("full13", True), ("no_category", True), ("no_recent", True), ("full13", False)), 1):
        multiplier = n / (n + 5.) if gate else 1.
        if not n:
            multiplier = 0.
        scores[index] = -original.astype(np.float64) + 10. * multiplier * residuals[variant]
    orders = np.stack([np.lexsort((original, -value)) for value in scores]).astype(np.int16)
    return orders, scores


def query_diagnostics(first_seen, coverage, history, pool, target, timestamp, fallback):
    cutoff = int(timestamp) // HOUR * HOUR - HOUR
    return {
        "empty_history": int(len(history) == 0),
        "unavailable_target": int(first_seen[target] >= cutoff),
        "already_read_target": int(target in history),
        "shortfall": 100 - len(pool), "queries_with_shortfall": int(len(pool) < 100),
        "fallback_items": int(fallback), "queries_with_fallback": int(fallback > 0),
        "history_items": min(len(history), 20), "candidate_items": len(pool),
        **{f"history_{name}": int(coverage[history[-20:], i].sum()) for i, name in enumerate(
            ("empty_title", "zero_tfidf", "zero_svd", "unknown_category"))},
        **{f"candidate_{name}": int(coverage[pool, i].sum()) for i, name in enumerate(
            ("empty_title", "zero_tfidf", "zero_svd", "unknown_category"))},
    }


class ReferenceReplay:
    """Separate disk aggregation and from-zero replay, caching one cutoff only.

    No optimized state, catalogue or user-history output is reused. Group raw
    events once instead of repeating a disk sort for every audit query. At most
    5000 grouped rows and one article-state array are resident in Python.
    """
    def __init__(self, source, path, n_items, log):
        self.connection = connect_new(path)
        self.connection.execute("CREATE TABLE counts(hour INTEGER,item INTEGER,n INTEGER,PRIMARY KEY(hour,item)) WITHOUT ROWID")
        began = time.perf_counter()
        last_log = [began]
        def heartbeat():
            now = time.perf_counter()
            if now - last_log[0] >= 15:
                log("reference_grouping_progress", elapsed_seconds=now - began)
                last_log[0] = now
            return 0
        source.set_progress_handler(heartbeat, 1000000)
        try:
            rows = source.execute("""SELECT timestamp/3600,item,COUNT(*) FROM events
                GROUP BY timestamp/3600,item ORDER BY timestamp/3600,item""")
            count = 0
            while True:
                batch = rows.fetchmany(5000)
                if not batch:
                    break
                self.connection.executemany("INSERT INTO counts VALUES(?,?,?)", batch)
                self.connection.commit()
                count += len(batch)
                heartbeat()
        finally:
            source.set_progress_handler(None, 0)
        self.n_items, self.last_cutoff, self.last_state = n_items, None, None
        log("reference_grouping_complete", grouped_rows=count, elapsed_seconds=time.perf_counter() - began)

    def state(self, cutoff):
        if cutoff == self.last_cutoff:
            return self.last_state
        state = np.zeros(self.n_items, dtype=np.float64)
        rows = iter(self.connection.execute("SELECT hour,item,n FROM counts WHERE hour<? ORDER BY hour,item", (cutoff // HOUR,)))
        current = next(rows, None)
        for hour in range(START, cutoff, HOUR):
            state *= DECAY
            while current is not None and current[0] == hour // HOUR:
                state[current[1]] += current[2]
                current = next(rows, None)
        if current is not None:
            raise ValueError("reference replay failed to consume prefix")
        self.last_cutoff, self.last_state = cutoff, state
        return state

    def close(self):
        self.connection.close()


def reference_query(connection, first_seen, features, models, user, timestamp, replay=None):
    """Rebuild one request from the DB, without the optimized EMA/catalogue cursor.

    The frozen feature definition is shared intentionally; the reference does not
    refit or substitute a mathematically different feature implementation.
    It independently normalizes/scales model arrays and constructs every order.
    """
    timestamp = int(timestamp)
    top_hour = timestamp // HOUR * HOUR
    cutoff = top_hour - HOUR
    # Separate query forms verify all-history filtering versus last-20 features,
    # without materializing a potentially ten-million-row user history in Python.
    seen = {row[0] for row in connection.execute("SELECT DISTINCT item FROM events WHERE user=? AND timestamp<?", (int(user), timestamp))}
    history_tail = list(connection.execute("""SELECT item FROM events WHERE user=? AND timestamp<?
        ORDER BY timestamp DESC,member COLLATE BINARY DESC,line DESC LIMIT 20""", (int(user), timestamp)))
    history = np.asarray([row[0] for row in reversed(history_tail)], dtype=np.int64)
    if replay is not None:
        state = replay.state(cutoff)
    else:
        # Tiny synthetic tests may use this uncached slow reference directly.
        counts = iter(connection.execute("""SELECT timestamp/3600,item,COUNT(*) FROM events
            WHERE timestamp<? GROUP BY timestamp/3600,item ORDER BY timestamp/3600,item""", (cutoff,)))
        current = next(counts, None)
        state = np.zeros(len(first_seen), dtype=np.float64)
        for hour in range(START, cutoff, HOUR):
            state *= DECAY
            while current is not None and current[0] == hour // HOUR:
                state[current[1]] += current[2]
                current = next(counts, None)
        if current is not None:
            raise ValueError("reference replay failed to consume prefix")
    visible = [index for index in range(1, len(first_seen))
               if 0 <= first_seen[index] < cutoff and index not in seen]
    visible.sort(key=lambda index: (-float(state[index]), -int(first_seen[index]), index))
    recent = [index for index in visible if first_seen[index] >= top_hour - 24 * HOUR][:200]
    pool = recent[:100]
    blocked = set(pool)
    for index in visible:
        if len(pool) == 100:
            break
        if index not in blocked:
            pool.append(index)
            blocked.add(index)
    pool = np.asarray(pool, dtype=np.int64)
    m = len(pool)
    if m == 0:
        return pool, np.empty((5, 0), dtype=np.int16), np.empty((5, 0), dtype=np.float64)
    recent_history = history[-20:]
    matrix = content_features(features[recent_history][None], features[pool][None],
                              (recent_history > 0)[None], text_dimensions=96)[0, :, :13]
    residuals = {}
    for variant, seed_models in models.items():
        seed_values = []
        for model in seed_models:
            values = np.asarray(matrix, dtype=np.float64)[:, list(model.feature_indices)]
            values = ((values - model.mean) / model.scale) @ model.weights
            seed_values.append(np.maximum(-1., np.minimum(1., values)))
        residuals[variant] = (seed_values[0] + seed_values[1] + seed_values[2]) / 3.
    all_scores, all_orders = [], []
    for method in METHODS:
        if method == "baseline" or not len(recent_history):
            score = -np.arange(m, dtype=np.float64)
        else:
            variant, gate_name = method.split("+")
            gate = len(recent_history) / (len(recent_history) + 5.) if gate_name == "gate" else 1.
            score = -np.arange(m, dtype=np.float64) + 10. * gate * residuals[variant]
        all_scores.append(score)
        all_orders.append(sorted(range(m), key=lambda index: (-float(score[index]), index)))
    return pool, np.asarray(all_orders, dtype=np.int16), np.asarray(all_scores, dtype=np.float64)


def evaluate(repo, run_dir):
    run_dir = Path(run_dir)
    preparation = read_json(run_dir / "prepare/receipt.json")
    with stage(run_dir, "evaluate") as (out, log), _MemoryMonitor() as memory:
        start = time.perf_counter()
        bindings = check_previous_bindings(repo, preparation["bindings"])
        prepared = run_dir / "prepare"
        check_files(prepared, preparation["files"])
        features = np.load(prepared / "features.npy", mmap_mode="r", allow_pickle=False)
        coverage = np.load(prepared / "coverage.npy", mmap_mode="r", allow_pickle=False)
        first_seen = np.load(prepared / "first_seen.npy", mmap_mode="r", allow_pickle=False)
        models = load_models(repo)
        connection = connect_readonly(prepared / "events.sqlite")
        reference = None
        try:
            query_rows = list(connection.execute("""SELECT e.id,e.timestamp,e.user,e.item FROM sampled s
                JOIN events e ON e.id=s.event_id ORDER BY e.timestamp,e.member COLLATE BINARY,e.line"""))
            queries = np.asarray(query_rows, dtype=np.int64)
            if queries.ndim != 2 or not 128 <= len(queries) <= 15000:
                raise ValueError("invalid formal query count")
            np.save(out / "queries.npy", queries, allow_pickle=False)
            n = len(queries)
            pools = np.lib.format.open_memmap(out / "pools.npy", mode="w+", dtype=np.int32, shape=(n, 100))
            sizes = np.lib.format.open_memmap(out / "sizes.npy", mode="w+", dtype=np.int16, shape=(n,))
            orders = np.lib.format.open_memmap(out / "orders.npy", mode="w+", dtype=np.int16, shape=(n, 5, 100))
            scores = np.lib.format.open_memmap(out / "scores.npy", mode="w+", dtype=np.float64, shape=(n, 5, 100))
            hits = np.lib.format.open_memmap(out / "hits10.npy", mode="w+", dtype=np.uint8, shape=(n, 5))
            hits100 = np.lib.format.open_memmap(out / "hits100.npy", mode="w+", dtype=np.uint8, shape=(n, 5))
            reciprocal = np.lib.format.open_memmap(out / "reciprocal10.npy", mode="w+", dtype=np.float64, shape=(n, 5))
            pools[:] = 0; orders[:] = -1; scores[:] = 0; hits[:] = 0; hits100[:] = 0; reciprocal[:] = 0
            diagnostics = Counter()
            ema, previous_hour = HourlyEMA(connection, len(first_seen), log), None
            last_log = time.perf_counter()
            with threadpool_limits(limits=1):
                for index, (_, stamp, user, target) in enumerate(queries):
                    hour = int(stamp) // HOUR * HOUR
                    if hour != previous_hour:
                        state = ema.advance(hour - HOUR)
                        global_order, recent_order = catalogue_orders(state, first_seen, stamp)
                        previous_hour = hour
                    history = user_history(connection, user, stamp)
                    pool, fallback = select_pool(global_order, recent_order, history)
                    method_orders, method_scores = rank_all(features, history, pool, models)
                    m = len(pool)
                    if len(np.unique(pool)) != m:
                        raise ValueError("duplicate candidate in pool")
                    sizes[index] = m
                    pools[index, :m] = pool
                    orders[index, :, :m] = method_orders
                    scores[index, :, :m] = method_scores
                    for method_index in range(5):
                        ordered = pool[method_orders[method_index]]
                        if not np.array_equal(np.sort(ordered), np.sort(pool)):
                            raise ValueError("reranking changed the candidate set")
                        locations = np.flatnonzero(ordered == target)
                        hits100[index, method_index] = bool(len(locations))
                        if len(locations) and locations[0] < 10:
                            hits[index, method_index] = 1
                            reciprocal[index, method_index] = 1. / (int(locations[0]) + 1)
                    diagnostics.update(query_diagnostics(first_seen, coverage, history, pool, int(target), int(stamp), fallback))
                    now = time.perf_counter()
                    if now - last_log >= 15:
                        log("prediction_progress", completed=index + 1, total=n)
                        last_log = now
                audit_indices = np.linspace(0, n - 1, 128, dtype=np.int64)
                max_error = 0.
                reference = ReferenceReplay(connection, out / "reference_counts.sqlite", len(first_seen), log)
                for audit_number, index in enumerate(audit_indices):
                    _, stamp, user, target = queries[index]
                    pool, reference_orders, reference_scores = reference_query(
                        connection, first_seen, features, models, user, stamp, replay=reference)
                    m = int(sizes[index])
                    if not np.array_equal(pool, pools[index, :m]):
                        raise ValueError("independent audit candidate mismatch")
                    if not np.array_equal(reference_orders, orders[index, :, :m]):
                        raise ValueError("independent audit ranking mismatch")
                    error = float(np.max(np.abs(reference_scores - scores[index, :, :m]), initial=0.))
                    max_error = max(max_error, error)
                    if error > 1e-8:
                        raise ValueError("independent audit score mismatch")
                    for method_index in range(5):
                        expected_hit = int(target in pool[reference_orders[method_index]][:10])
                        if expected_hit != int(hits[index, method_index]):
                            raise ValueError("independent audit target count mismatch")
                    if (audit_number + 1) % 8 == 0:
                        log("reference_audit_progress", completed=audit_number + 1, total=128)
            if not np.all(hits100 == hits100[:, :1]):
                raise ValueError("per-query HitRate@100 invariant failed")
            for array in (pools, sizes, orders, scores, hits, hits100, reciprocal):
                array.flush()
            del pools, sizes, orders, scores, hits, hits100, reciprocal
        finally:
            if reference is not None:
                reference.close()
            connection.close()
        check_previous_bindings(repo, bindings)
        check_files(prepared, preparation["files"])
        audit = {"schema_version": 1, "status": "passed", "queries": 128,
            "selection": "linspace(0,n-1,128,dtype=int64); includes first and last fixed query",
            "indices_sha256": __import__("hashlib").sha256(audit_indices.tobytes()).hexdigest(),
            "candidate_id_equality": "exact", "ranking_equality": "exact", "hit_equality": "exact",
            "score_atol": 1e-8, "score_rtol": 0., "score_max_absolute_error": max_error,
            "independently_rebuilt": ["historical event prefix", "hourly EMA", "full history filter",
                                       "candidate fallback", "normalization and model scores", "sorting", "target hits"],
            "shared_frozen_definition": "newsforge.v2.ranker.content_features",
            "reference_aggregation": "separate raw-event SQL aggregation on disk; EMA rebuilt from zero per distinct cutoff; one-cutoff cache",
            "all_method_per_query_candidate_set_invariant": True,
            "all_method_per_query_hit100_invariant": True}
        write_new_json(out / "audit.json", audit)
        record = {"schema_version": 1, "status": "passed", "bindings": bindings,
            "prepare_receipt_sha256": sha256(prepared / "receipt.json"),
            "sample_identity_sha256": preparation["sample_identity_sha256"],
            "queries": n, "methods": list(METHODS), "diagnostics": dict(diagnostics),
            "elapsed_seconds": time.perf_counter() - start, "memory": memory.report(),
            "files": file_records(out, ("queries.npy", "pools.npy", "sizes.npy", "orders.npy", "scores.npy",
                                        "hits10.npy", "hits100.npy", "reciprocal10.npy", "audit.json", "reference_counts.sqlite"))}
        write_new_json(out / "receipt.json", record)
        return record
