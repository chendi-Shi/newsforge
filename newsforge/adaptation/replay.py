"""Read-only causal replay for the separate Adressa adaptation experiment.

The caller binds the existing input files and chooses the phase boundaries.
This module never changes the frozen external experiment or its databases.
"""
from __future__ import annotations

import hashlib
from operator import index

import numpy as np

from newsforge.external.common import DECAY, END, HOUR, START
from newsforge.external.evaluate import catalogue_orders, select_pool, user_history


def _sample_key(phase, member, line):
    payload = f"newsforge-adressa-adapt-v1|{phase}|{member}|{line}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


def select_phase_queries(connection, phase, start, end, limit):
    """Select label-independent hash minima, then return chronological queries.

    Output columns are event ID, timestamp, user and target item. SQLite's
    ORDER BY/LIMIT retains only the bounded top-k selection; the Python list
    is bounded by ``limit``. Registering a scalar function changes no database
    bytes and also works on query-only, mode=ro connections.
    """
    start, end, limit = index(start), index(end), index(limit)
    if not isinstance(phase, str) or not phase or "|" in phase:
        raise ValueError("phase must be a nonempty namespace component")
    if start >= end or limit < 0:
        raise ValueError("invalid phase interval or query limit")
    connection.create_function("adaptation_sample_key", 3, _sample_key, deterministic=True)
    rows = list(connection.execute("""SELECT id,timestamp,user,item,member,line FROM events
        WHERE timestamp>=? AND timestamp<?
        ORDER BY adaptation_sample_key(?,member,line),member COLLATE BINARY,line
        LIMIT ?""", (start, end, phase, limit)))
    rows.sort(key=lambda row: (row[1], row[4].encode("utf-8"), row[5]))
    return np.asarray([row[:4] for row in rows], dtype=np.int64).reshape(-1, 4)


def load_original_queries(connection):
    """Load exactly the saved original sample, without selecting it again."""
    rows = list(connection.execute("""SELECT e.id,e.timestamp,e.user,e.item
        FROM sampled s JOIN events e ON e.id=s.event_id
        ORDER BY e.timestamp,e.member COLLATE BINARY,e.line"""))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 4)


def build_ema_snapshots(counts_connection, n_items):
    """Return states at every cutoff START..END, including both endpoints.

    Row ``(C-START)//HOUR`` includes only events strictly before C. Input is
    the already bound counts(hour,item,n) table, with integer Unix-hour keys.
    Padding item zero is never updated. No user events or raw archive are read.
    """
    n_items = index(n_items)
    if n_items < 1:
        raise ValueError("n_items must include a padding row")
    hours = (END - START) // HOUR
    snapshots = np.zeros((hours + 1, n_items), dtype=np.float64)
    rows = iter(counts_connection.execute("SELECT hour,item,n FROM counts ORDER BY hour,item"))
    row = next(rows, None)
    for offset in range(hours):
        hour = START // HOUR + offset
        snapshots[offset + 1] = snapshots[offset] * DECAY
        while row is not None and row[0] <= hour:
            bucket, item, count = row
            if bucket != hour or not 0 < item < n_items or count < 0:
                raise ValueError("invalid hourly count row")
            snapshots[offset + 1, item] += count
            row = next(rows, None)
    if row is not None:
        raise ValueError("hourly counts extend past the bound week")
    return snapshots


def _query_state(snapshots, first_seen, timestamp):
    cutoff = index(timestamp) // HOUR * HOUR - HOUR
    if snapshots.shape != ((END - START) // HOUR + 1, len(first_seen)):
        raise ValueError("unexpected EMA snapshot shape")
    if cutoff > END:
        raise ValueError("query cutoff extends past the bound week")
    # Jan 1's first-hour requests have a cutoff before the observed week.
    return snapshots[max(0, (cutoff - START) // HOUR)]


def replay_query(connection, first_seen, snapshots, user, timestamp):
    """Return full strictly prior history and the frozen popularity pool.

    Queries can arrive in any order because the public state is a snapshot.
    The complete history excludes read items; downstream content models may
    separately use its last twenty entries.
    """
    state = _query_state(snapshots, first_seen, timestamp)
    history = user_history(connection, user, timestamp)
    global_order, recent_order = catalogue_orders(state, first_seen, timestamp)
    pool, _ = select_pool(global_order, recent_order, history)
    return history, pool


def load_reference_hourly_counts(connection):
    """Independently aggregate normalized events for repeated audit queries.

    These rows must originate from this query, not the optimized counts-table
    snapshots. The compact result can be passed repeatedly to the slow path.
    """
    return list(connection.execute("""SELECT timestamp/3600,item,COUNT(*) FROM events
        GROUP BY timestamp/3600,item ORDER BY timestamp/3600,item"""))


def slow_reference_pool(connection, first_seen, user, timestamp, *, hourly_counts=None):
    """Independently rebuild a pool from raw SQL and explicit Python ordering.

    Return the same ``(full_history, pool)`` contract as ``replay_query``.
    ``hourly_counts`` may cache ``load_reference_hourly_counts`` output, but
    each query rebuilds its state from zero and never uses optimized candidate
    functions, snapshots, or saved pools. Rows at/after the public cutoff are
    ignored even when the supplied raw aggregation spans the complete week.
    """
    timestamp = index(timestamp)
    cutoff = timestamp // HOUR * HOUR - HOUR
    if cutoff > END:
        raise ValueError("query cutoff extends past the bound week")
    history = np.asarray([row[0] for row in connection.execute("""SELECT item FROM events
        WHERE user=? AND timestamp<? ORDER BY timestamp,member COLLATE BINARY,line""",
        (index(user), timestamp))], dtype=np.int64)
    state = np.zeros(len(first_seen), dtype=np.float64)
    if cutoff > START:
        if hourly_counts is None:
            hourly_counts = connection.execute("""SELECT timestamp/3600,item,COUNT(*) FROM events
                WHERE timestamp<? GROUP BY timestamp/3600,item ORDER BY timestamp/3600,item""",
                (cutoff,))
        rows = iter(hourly_counts)
        row = next(rows, None)
        for hour in range(START // HOUR, cutoff // HOUR):
            state *= DECAY
            while row is not None and row[0] <= hour:
                bucket, item, count = row
                if bucket != hour or not 0 < item < len(first_seen) or count < 0:
                    raise ValueError("invalid independent hourly count row")
                state[item] += count
                row = next(rows, None)
    seen = set(history.tolist())
    visible = [item for item in range(1, len(first_seen))
               if 0 <= int(first_seen[item]) < cutoff and item not in seen]
    visible.sort(key=lambda item: (-float(state[item]), -int(first_seen[item]), item))
    recent_start = timestamp // HOUR * HOUR - 24 * HOUR
    recent = [item for item in visible if int(first_seen[item]) >= recent_start][:200]
    pool = recent[:100]
    blocked = set(pool)
    for item in visible:
        if len(pool) >= 100:
            break
        if item not in blocked:
            pool.append(item)
            blocked.add(item)
    return history, np.asarray(pool, dtype=np.int64)
