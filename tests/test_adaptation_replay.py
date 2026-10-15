"""Invented events only; no actual Adressa records or frozen results are read."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import newsforge

_staged = Path(__file__).resolve().parents[1] / "newsforge"
if str(_staged) not in newsforge.__path__:
    newsforge.__path__.insert(0, str(_staged))

import numpy as np

from newsforge.adaptation.replay import (build_ema_snapshots, load_original_queries,
    load_reference_hourly_counts, replay_query, select_phase_queries, slow_reference_pool)
from newsforge.external.common import DECAY, END, HOUR, START
from newsforge.external.evaluate import HourlyEMA, catalogue_orders, select_pool, user_history
from newsforge.external.prepare import connect_readonly


def event_database(events, path=":memory:"):
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE events(id INTEGER PRIMARY KEY,timestamp INTEGER,user INTEGER,
                            item INTEGER,member TEXT,line INTEGER);
        CREATE INDEX events_time ON events(timestamp,item);
        CREATE INDEX events_user_time ON events(user,timestamp,member COLLATE BINARY,line);
        CREATE TABLE sampled(event_id INTEGER PRIMARY KEY,day TEXT,sample_key BLOB);
        CREATE TABLE counts(hour INTEGER,item INTEGER,n INTEGER,PRIMARY KEY(hour,item));
    """)
    connection.executemany("INSERT INTO events VALUES(?,?,?,?,?,?)", events)
    connection.execute("""INSERT INTO counts
        SELECT timestamp/3600,item,COUNT(*) FROM events GROUP BY timestamp/3600,item""")
    connection.commit()
    return connection


def first_observation(events, n_items):
    values = np.full(n_items, -1, dtype=np.int64)
    for _, stamp, _, item, _, _ in events:
        if values[item] < 0 or stamp < values[item]:
            values[item] = stamp
    return values


class AdaptationSamplingTests(unittest.TestCase):
    def setUp(self):
        self.events = [(i + 1, START + (i * 17) % 90, i % 4 + 1,
                        i % 7 + 1, "week/" + ("é" if i % 2 else "a"), i + 1)
                       for i in range(30)]
        self.connection = event_database(self.events)

    def tearDown(self):
        self.connection.close()

    def test_exact_hash_namespace_boundaries_and_label_independence(self):
        eligible = [row for row in self.events if START + 10 <= row[1] < START + 80]
        def expected_key(row):
            payload = f"newsforge-adressa-adapt-v1|fit|{row[4]}|{row[5]}"
            return hashlib.sha256(payload.encode()).digest(), row[4].encode(), row[5]
        expected = sorted(sorted(eligible, key=expected_key)[:9],
                          key=lambda row: (row[1], row[4].encode(), row[5]))
        selected = select_phase_queries(self.connection, "fit", START + 10, START + 80, 9)
        np.testing.assert_array_equal(selected, np.asarray([row[:4] for row in expected]))
        self.connection.execute("UPDATE events SET item=item+100,user=user+200")
        changed = select_phase_queries(self.connection, "fit", START + 10, START + 80, 9)
        np.testing.assert_array_equal(changed[:, :2], selected[:, :2])
        self.assertEqual(selected.dtype, np.dtype(np.int64))
        self.assertEqual(select_phase_queries(self.connection, "fit", START, END, 0).shape, (0, 4))
        self.assertEqual(select_phase_queries(self.connection, "fit", END, END + HOUR, 9).shape, (0, 4))

    def test_hash_tie_uses_physical_identity_then_returns_chronological(self):
        with patch("newsforge.adaptation.replay._sample_key", return_value=b"same"):
            selected = select_phase_queries(self.connection, "tune", START, END, 8)
        expected = sorted(sorted(self.events, key=lambda row: (row[4].encode(), row[5]))[:8],
                          key=lambda row: (row[1], row[4].encode(), row[5]))
        np.testing.assert_array_equal(selected, [row[:4] for row in expected])

    def test_original_sample_is_preserved_exactly(self):
        identifiers = [18, 2, 29, 5]
        self.connection.executemany("INSERT INTO sampled VALUES(?,?,?)",
                                   [(i, "invented", b"unused") for i in identifiers])
        expected = sorted([row for row in self.events if row[0] in identifiers],
                          key=lambda row: (row[1], row[4].encode(), row[5]))
        np.testing.assert_array_equal(load_original_queries(self.connection),
                                      [row[:4] for row in expected])


class AdaptationReplayTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            (1, START, 9, 1, "a", 1),
            (2, START + HOUR - 1, 9, 2, "a", 2),
            (3, START + HOUR, 9, 3, "a", 3),
            (4, START + HOUR + 1, 9, 1, "a", 4),
            (5, START + 2 * HOUR - 1, 7, 2, "a", 5),
            (6, START + 3 * HOUR, 7, 3, "a", 6),
            (7, START + 3 * HOUR, 7, 1, "b", 1),
            (8, START + 3 * HOUR + 1, 7, 4, "b", 2),
        ]
        self.connection = event_database(self.events)
        self.first_seen = first_observation(self.events, 6)
        self.snapshots = build_ema_snapshots(self.connection, len(self.first_seen))

    def tearDown(self):
        self.connection.close()

    def test_snapshot_strict_cutoff_and_hourly_decay(self):
        self.assertEqual(self.snapshots.shape, (169, 6))
        self.assertEqual(self.snapshots.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(self.snapshots[0], np.zeros(6))
        np.testing.assert_array_equal(self.snapshots[1], [0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(self.snapshots[2], [0, DECAY + 1, DECAY + 1, 1, 0, 0])
        self.assertTrue(np.all(self.snapshots[:, 0] == 0))
        ema = HourlyEMA(self.connection, len(self.first_seen))
        for offset in range(169):
            np.testing.assert_array_equal(self.snapshots[offset], ema.advance(START + offset * HOUR))

    def test_same_second_history_isolation_and_early_empty_pool(self):
        history, pool = replay_query(self.connection, self.first_seen, self.snapshots, 7, START + 3 * HOUR)
        np.testing.assert_array_equal(history, [2])
        self.assertNotIn(2, pool)
        self.assertIn(1, pool)
        self.assertIn(3, pool)
        self.assertNotIn(4, pool)
        self.assertNotIn(5, pool)
        history_after, _ = replay_query(self.connection, self.first_seen, self.snapshots, 7,
                                        START + 3 * HOUR + 1)
        np.testing.assert_array_equal(history_after, [2, 3, 1])
        for stamp in (START, START + HOUR - 1, START + HOUR):
            _, early = replay_query(self.connection, self.first_seen, self.snapshots, 7, stamp)
            self.assertEqual(len(early), 0)

    def test_arbitrary_order_matches_frozen_candidates_and_independent_raw_replay(self):
        raw_counts = load_reference_hourly_counts(self.connection)
        stamps = [START + 28 * HOUR, START + 3 * HOUR, START + 2 * HOUR,
                  START + 3 * HOUR + 1, START, START + HOUR]
        for stamp in stamps:
            history, pool = replay_query(self.connection, self.first_seen, self.snapshots, 7, stamp)
            reference_history, reference_pool = slow_reference_pool(
                self.connection, self.first_seen, 7, stamp, hourly_counts=raw_counts)
            np.testing.assert_array_equal(history, reference_history)
            np.testing.assert_array_equal(pool, reference_pool)
            raw_history, raw_pool = slow_reference_pool(self.connection, self.first_seen, 7, stamp)
            np.testing.assert_array_equal(history, raw_history)
            np.testing.assert_array_equal(pool, raw_pool)
            cutoff = stamp // HOUR * HOUR - HOUR
            if cutoff >= START:
                ema = HourlyEMA(self.connection, len(self.first_seen))
                global_order, recent_order = catalogue_orders(ema.advance(cutoff), self.first_seen, stamp)
                old_pool, _ = select_pool(global_order, recent_order, user_history(self.connection, 7, stamp))
                np.testing.assert_array_equal(pool, old_pool)

    def test_future_events_and_future_article_observations_do_not_change_past(self):
        stamp = START + 3 * HOUR
        before = replay_query(self.connection, self.first_seen, self.snapshots, 7, stamp)
        self.connection.executemany("INSERT INTO events VALUES(?,?,?,?,?,?)", [
            (9, START + 4 * HOUR, 7, 5, "z", 1),
            (10, START + 4 * HOUR, 9, 1, "z", 2)])
        self.connection.execute("DELETE FROM counts")
        self.connection.execute("INSERT INTO counts SELECT timestamp/3600,item,COUNT(*) FROM events GROUP BY timestamp/3600,item")
        later_seen = self.first_seen.copy()
        later_seen[5] = START + 4 * HOUR
        later = build_ema_snapshots(self.connection, len(later_seen))
        after = replay_query(self.connection, later_seen, later, 7, stamp)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)

    def test_full_history_filter_recent_fallback_and_stable_ties(self):
        events = [(i, START + 1, 9, i, "a", i) for i in range(1, 126)]
        events += [(125 + i, START + 25 * HOUR, 9, i, "b", i) for i in range(126, 141)]
        events += [(300 + i, START + 26 * HOUR + i, 7, i, "c", i) for i in range(1, 26)]
        connection = event_database(events)
        try:
            first_seen = first_observation(events, 141)
            snapshots = build_ema_snapshots(connection, 141)
            history, pool = replay_query(connection, first_seen, snapshots, 7, START + 28 * HOUR)
            self.assertEqual(len(history), 25)
            self.assertEqual(len(pool), 100)
            self.assertFalse(set(range(1, 26)) & set(pool))
            np.testing.assert_array_equal(pool[:15], np.arange(126, 141))
            np.testing.assert_array_equal(pool[15:], np.arange(26, 111))
            _, reference = slow_reference_pool(connection, first_seen, 7, START + 28 * HOUR)
            np.testing.assert_array_equal(pool, reference)
        finally:
            connection.close()

    def test_readonly_database_bytes_unchanged_by_all_interfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invented.sqlite"
            writer = event_database(self.events, path)
            writer.execute("INSERT INTO sampled VALUES(1,'invented',X'00')")
            writer.commit()
            writer.close()
            digest_before = hashlib.sha256(path.read_bytes()).hexdigest()
            connection = connect_readonly(path)
            try:
                selected = select_phase_queries(connection, "fit", START, END, 3)
                self.assertEqual(len(selected), 3)
                self.assertEqual(len(load_original_queries(connection)), 1)
                snapshots = build_ema_snapshots(connection, len(self.first_seen))
                replay_query(connection, self.first_seen, snapshots, 7, START + 3 * HOUR)
                slow_reference_pool(connection, self.first_seen, 7, START + 3 * HOUR,
                                    hourly_counts=load_reference_hourly_counts(connection))
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute("DELETE FROM events")
            finally:
                connection.close()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest_before)
            self.assertEqual({file.name for file in path.parent.iterdir()}, {path.name})


if __name__ == "__main__":
    unittest.main()
