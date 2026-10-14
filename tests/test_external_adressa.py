"""Only invented events: these tests never download or read the target dataset."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import newsforge

# Support review in a separate staging tree without changing the frozen package.
_staged = Path(__file__).resolve().parents[1] / "newsforge"
if str(_staged) not in newsforge.__path__:
    newsforge.__path__.insert(0, str(_staged))

import numpy as np
from scipy.sparse import csr_matrix

from newsforge.external import acquire as acquisition_module
from newsforge.external import prepare as preparation_module
from newsforge.external import common as common_module
from newsforge.external import evaluate as evaluation_module
from newsforge.external.__main__ import main as external_main
from newsforge.external.aggregate import paired_interval, summarize
from newsforge.external.common import DECAY, EVAL_START, HOUR, METHODS, START, sample_key, stage
from newsforge.external.evaluate import (HourlyEMA, ReferenceReplay, catalogue_orders, frozen_features, rank_all,
                                         reference_query, select_pool, user_history)
from newsforge.external.prepare import (connect_new, ingest_archive, normalize_database,
                                        article_batches, parse_event, safe_member, transform_articles)
from newsforge.v3.residual import PairwiseResidual, VARIANTS


def quiet(*args, **kwargs):
    pass


def archive(path, files):
    with tarfile.open(path, "w:gz") as stream:
        for name, body in files:
            if not isinstance(body, bytes):
                body = ("\n".join(json.dumps(value) for value in body) + "\n").encode()
            info = tarfile.TarInfo(name)
            info.size = len(body)
            stream.addfile(info, io.BytesIO(body))


def models():
    result = {}
    for variant in VARIANTS:
        result[variant] = []
        for seed in (17, 29, 43):
            model = PairwiseResidual(variant)
            n = len(model.feature_indices)
            model.mean = np.linspace(-.1, .1, n)
            model.scale = np.linspace(.8, 1.2, n)
            model.weights = np.random.default_rng(seed).normal(0, .2, n)
            result[variant].append(model)
    return result


def event_db(events):
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE events(id INTEGER PRIMARY KEY,timestamp INTEGER,user INTEGER,item INTEGER,member TEXT,line INTEGER)")
    connection.executemany("INSERT INTO events VALUES(?,?,?,?,?,?)", events)
    connection.execute("CREATE INDEX et ON events(timestamp,item)")
    connection.execute("CREATE INDEX eu ON events(user,timestamp,member,line)")
    return connection


class SchemaAndArchiveTests(unittest.TestCase):
    def test_malformed_primary_id_cannot_fallback_and_valid_alias_can(self):
        value = {"userId": "u", "time": START, "id": 123, "documentId": "a"}
        self.assertEqual(parse_event(value), (None, "invalid_article"))
        self.assertEqual(parse_event({**value, "id": None})[0][2], "a")
        self.assertEqual(parse_event({**value, "id": "a", "documentId": 123}), (None, "invalid_article"))

    def test_schema_and_alias_conflict(self):
        valid = {"userId": "user", "time": START, "id": "article", "title": None}
        self.assertEqual(parse_event(valid), (("user", START, "article", "", ""), None))
        for value, reason in (({**valid, "time": True}, "invalid_timestamp"),
                              ({**valid, "time": START - 1}, "out_of_range"),
                              ({**valid, "userId": " "}, "invalid_user"),
                              ({**valid, "id": None}, "invalid_article")):
            self.assertEqual(parse_event(value)[1], reason)
        self.assertEqual(parse_event({**valid, "id": None, "documentId": "fallback"})[0][2], "fallback")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            parse_event({**valid, "documentId": "different"})

    def test_unsafe_and_duplicate_members(self):
        for value in ("../x", "/absolute", "C:/x", "a\\b", "a//b", "a/./b", "",
                      "./../x", "./a/../x", "./C:/x", ".//absolute", "./a/./b"):
            with self.assertRaises(ValueError):
                safe_member(value, set())
        seen = set()
        safe_member("week/20170101", seen)
        with self.assertRaises(ValueError):
            safe_member("week/20170101", seen)

    def test_safe_leading_dot_and_root_directory_require_actual_directory(self):
        seen = set()
        for name in (".", "./", "./one_week/", "./one_week/20170101"):
            directory = name in (".", "./", "./one_week/")
            safe_member(name, seen, is_directory=directory)
            self.assertIn(name, seen)
        for name in (".", "./"):
            with self.assertRaises(ValueError):
                safe_member(name, set(), is_directory=False)
        with self.assertRaises(ValueError):
            safe_member("./one_week/20170101", seen)
        # Distinct raw identities are never collapsed, even if paths look alike.
        safe_member("one_week/20170101", seen)
        self.assertEqual(len(seen), 5)

    def test_streamed_leading_dot_tar_preserves_raw_member_sampling_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.tar.gz"
            body = (json.dumps({"userId": "u", "time": EVAL_START, "documentId": "article"}) + "\n").encode()
            with tarfile.open(path, "w:gz") as bundle:
                for name in ("./", "./one_week/"):
                    directory = tarfile.TarInfo(name)
                    directory.type = tarfile.DIRTYPE
                    bundle.addfile(directory)
                member = tarfile.TarInfo("./one_week/20170105")
                member.size = len(body)
                bundle.addfile(member, io.BytesIO(body))
            connection = connect_new(Path(folder) / "events.sqlite")
            try:
                counts, members = ingest_archive(path, connection, quiet)
                self.assertEqual(counts["deduplicated_events"], 1)
                self.assertEqual(members[0]["name"], "./one_week/20170105")
                normalize_database(connection, quiet)
                member, key = connection.execute("SELECT e.member,s.sample_key FROM events e JOIN sampled s ON s.event_id=e.id").fetchone()
                self.assertEqual(member, "./one_week/20170105")
                self.assertEqual(key, sample_key(EVAL_START, "./one_week/20170105", 1))
                self.assertNotEqual(key, sample_key(EVAL_START, "one_week/20170105", 1))
            finally:
                connection.close()

    def test_link_member_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.tar.gz"
            with tarfile.open(path, "w:gz") as stream:
                item = tarfile.TarInfo("link")
                item.type = tarfile.SYMTYPE
                item.linkname = "../outside"
                stream.addfile(item)
            connection = connect_new(Path(folder) / "events.sqlite")
            try:
                with self.assertRaisesRegex(ValueError, "non-regular"):
                    ingest_archive(path, connection, quiet)
            finally:
                connection.close()

    def test_dedup_and_first_metadata_are_physical_and_causal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.tar.gz"
            base = {"userId": "u", "time": START + 10, "id": "z", "category": "cat"}
            archive(path, [("week/b", [{**base, "title": "wrong-later-member"},
                         {**base, "time": START + 11, "title": "later-revised-title"}]),
                         ("week/a", [{**base, "title": ""}])])
            connection = connect_new(Path(folder) / "events.sqlite")
            try:
                counts, member_records = ingest_archive(path, connection, quiet)
                self.assertEqual(counts["deduplicated_events"], 2)
                self.assertEqual(counts["duplicate_events_removed"], 1)
                self.assertEqual(sum(row["lines"] for row in member_records), 3)
                normalize_database(connection, quiet)
                self.assertEqual(connection.execute("SELECT title,first_member,first_line FROM articles").fetchone(), ("", "week/a", 1))
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 2)
            finally:
                connection.close()

    def test_invalid_records_are_counted_and_row_budget_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.tar.gz"
            valid = json.dumps({"userId": "u", "time": START, "id": "a"}).encode()
            archive(path, [("week/day", b"broken\n[]\n" + valid + b"\n")])
            connection = connect_new(Path(folder) / "one.sqlite")
            try:
                counts, _ = ingest_archive(path, connection, quiet)
                self.assertEqual(counts["invalid_json"], 1)
                self.assertEqual(counts["not_object"], 1)
            finally:
                connection.close()
            connection = connect_new(Path(folder) / "two.sqlite")
            try:
                with patch.object(preparation_module, "MAX_LINES", 1), self.assertRaisesRegex(ValueError, "line budget"):
                    ingest_archive(path, connection, quiet)
            finally:
                connection.close()

    def test_sampling_is_target_independent_and_uses_lowest_hashes(self):
        first = sample_key(EVAL_START, "week/day", 1)
        self.assertEqual(first, sample_key(EVAL_START, "week/day", 1))
        self.assertNotEqual(first, sample_key(EVAL_START, "week/day", 2))
        self.assertNotEqual(first, sample_key(EVAL_START + 86400, "week/day", 1))
        selections = []
        with tempfile.TemporaryDirectory() as folder:
            for version in range(2):
                path = Path(folder) / f"{version}.tar.gz"
                rows = [{"userId": f"u{i}", "time": EVAL_START + i, "id": f"item-{version}-{i}"}
                        for i in range(10)]
                archive(path, [("week/day", rows)])
                connection = connect_new(Path(folder) / f"{version}.sqlite")
                try:
                    ingest_archive(path, connection, quiet)
                    with patch.object(preparation_module, "MAX_PER_DAY", 3):
                        normalize_database(connection, quiet)
                    selected = sorted(row[0] for row in connection.execute(
                        "SELECT e.line FROM sampled s JOIN events e ON e.id=s.event_id"))
                    expected = sorted(sorted(range(1, 11), key=lambda i: sample_key(EVAL_START, "week/day", i))[:3])
                    self.assertEqual(selected, expected)
                    selections.append(selected)
                finally:
                    connection.close()
        self.assertEqual(selections[0], selections[1])


class FrozenTransformTests(unittest.TestCase):
    def test_article_batch_limits_rows_and_utf8_payload_without_changing_order(self):
        rows = [(index, START, "å" * 3, "") for index in range(10)]
        groups = list(article_batches(iter(rows), max_rows=4, max_bytes=10))
        self.assertEqual([len(group) for group in groups], [2] * 5)
        self.assertEqual([row for group in groups for row in group], rows)
        groups = list(article_batches(iter(rows), max_rows=3, max_bytes=1000))
        self.assertEqual([len(group) for group in groups], [3, 3, 3, 1])

    def test_only_transform_is_called_and_unknowns_are_real_history(self):
        class Vectorizer:
            def fit(self, *args):
                raise AssertionError("external fit forbidden")
            def fit_transform(self, *args):
                raise AssertionError("external fit_transform forbidden")
            def transform(self, titles):
                return csr_matrix(np.asarray([[1., 0.] if title else [0., 0.] for title in titles], dtype=np.float32))
        class SVD:
            def fit(self, *args):
                raise AssertionError("external fit forbidden")
            def transform(self, matrix):
                return matrix.toarray()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = root / "data/processed"
            processed.mkdir(parents=True)
            (processed / "category_vocabulary.json").write_text('["[UNKNOWN]", "sports"]')
            (processed / "manifest.json").write_text(json.dumps({"features": {"text_dimensions": 96, "dimensions": 98}}))
            connection = sqlite3.connect(":memory:")
            connection.execute("CREATE TABLE articles(id INTEGER,first_seen INTEGER,title TEXT,category TEXT)")
            connection.executemany("INSERT INTO articles VALUES (?,?,?,?)", [(1, START, "Norwegian title", "nyheter"), (2, START, "", "sports")])
            try:
                with patch.object(preparation_module.joblib, "load", side_effect=[Vectorizer(), SVD()]):
                    result = transform_articles(connection, root, root, 2, quiet)
                f = np.load(root / "features.npy")
                self.assertTrue(np.all(f[0] == 0))
                self.assertEqual(f[1, 96], 1)
                self.assertEqual(f[2, 97], 1)
                self.assertTrue(np.all(f[2, :96] == 0))
                self.assertEqual(result["coverage"], {"empty_title": 1, "zero_tfidf": 1, "zero_svd": 1, "unknown_category": 1, "articles": 2})
                feat = frozen_features(f, np.array([2]), np.array([1]))
                self.assertEqual(feat.shape, (1, 13))
                self.assertTrue(np.isfinite(feat).all())
            finally:
                connection.close()


class CausalReplayTests(unittest.TestCase):
    def test_independent_disk_counts_cache_matches_slow_prefix_rebuild(self):
        features = np.zeros((4, 115), dtype=np.float32)
        features[1:, 96] = 1.
        first = np.array([-1, START, START + HOUR, EVAL_START - HOUR])
        events = [(1, START, 1, 1, "a", 1), (2, START + HOUR, 2, 2, "a", 2),
                  (3, EVAL_START - HOUR, 1, 3, "a", 3), (4, EVAL_START, 1, 2, "a", 4)]
        db = event_db(events)
        with tempfile.TemporaryDirectory() as folder:
            replay = ReferenceReplay(db, Path(folder) / "counts.sqlite", len(first), quiet)
            try:
                for stamp in (EVAL_START, EVAL_START + HOUR, EVAL_START):
                    fast = reference_query(db, first, features, models(), 1, stamp, replay=replay)
                    slow = reference_query(db, first, features, models(), 1, stamp)
                    for actual, expected in zip(fast, slow):
                        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
                before = replay.state(EVAL_START - HOUR)
                self.assertIs(before, replay.state(EVAL_START - HOUR))
            finally:
                replay.close()
                db.close()

    def test_cutoff_equality_and_future_events_do_not_enter_ema_or_catalogue(self):
        cutoff = EVAL_START - HOUR
        events = [(1, cutoff - 1, 1, 1, "day", 1), (2, cutoff, 2, 2, "day", 2),
                  (3, EVAL_START, 3, 1, "day", 3)]
        connection = event_db(events)
        try:
            state = HourlyEMA(connection, 3).advance(cutoff)
            np.testing.assert_array_equal(state, [0., 1., 0.])
            first = np.array([-1, cutoff - 1, cutoff])
            global_order, recent = catalogue_orders(state, first, EVAL_START)
            np.testing.assert_array_equal(global_order, [1])
            np.testing.assert_array_equal(recent, [1])
            connection.execute("UPDATE events SET item=2 WHERE timestamp>=?", (cutoff,))
            np.testing.assert_array_equal(HourlyEMA(connection, 3).advance(cutoff), state)
        finally:
            connection.close()

    def test_same_second_user_events_are_mutually_invisible(self):
        connection = event_db([(1, EVAL_START - 1, 1, 1, "x", 1),
                               (2, EVAL_START, 1, 2, "x", 2),
                               (3, EVAL_START, 1, 3, "x", 3)])
        try:
            np.testing.assert_array_equal(user_history(connection, 1, EVAL_START), [1])
            np.testing.assert_array_equal(user_history(connection, 1, EVAL_START + 1), [1, 2, 3])
        finally:
            connection.close()

    def test_ema_decay_and_out_of_order_request_rejected(self):
        connection = event_db([(1, START, 1, 1, "x", 1)])
        try:
            state = HourlyEMA(connection, 2)
            self.assertEqual(state.advance(START + HOUR)[1], 1.)
            self.assertEqual(state.advance(START + 2 * HOUR)[1], DECAY)
            with self.assertRaises(ValueError):
                state.advance(START + HOUR)
        finally:
            connection.close()

    def test_all_history_excluded_recent_budget_and_fallback(self):
        global_order = np.arange(1, 301)
        recent = np.arange(101, 131)
        history = np.arange(101, 126)
        pool, fallback = select_pool(global_order, recent, history)
        np.testing.assert_array_equal(pool[:5], [126, 127, 128, 129, 130])
        self.assertEqual(fallback, 95)
        self.assertEqual(len(pool), 100)
        self.assertFalse(np.isin(pool, history).any())
        self.assertFalse(101 in pool)  # the first history event is outside the last 20

    def test_stable_ties_first_seen_then_raw_id_ordinal(self):
        first = np.array([-1, START, START + 1, START + 1])
        order, _ = catalogue_orders(np.array([0., 2., 2., 2.]), first, EVAL_START)
        np.testing.assert_array_equal(order, [2, 3, 1])

    def test_empty_history_and_fixed_candidates(self):
        feature = np.zeros((9, 115), dtype=np.float32)
        feature[1:, 96] = 1
        pool = np.arange(1, 9)
        order, scores = rank_all(feature, np.array([], dtype=np.int64), pool, models())
        for index in range(5):
            np.testing.assert_array_equal(order[index], np.arange(8))
            np.testing.assert_array_equal(scores[index], -np.arange(8))
            np.testing.assert_array_equal(np.sort(pool[order[index]]), pool)

    def test_clip_each_seed_before_average_and_no_gate_empty_fallback(self):
        class Constant:
            def __init__(self, value): self.value = value
            def predict_raw(self, features): return np.full(len(features), self.value)
        frozen = {variant: [Constant(3), Constant(-1), Constant(-1)] for variant in VARIANTS}
        feature = np.zeros((4, 115), dtype=np.float32)
        feature[1:, 96] = 1
        _, score = rank_all(feature, np.array([1]), np.array([2, 3]), frozen)
        np.testing.assert_allclose(score[1], -np.arange(2) - 10. / 18., atol=1e-14, rtol=0)
        np.testing.assert_allclose(score[4], -np.arange(2) - 10. / 3., atol=1e-14, rtol=0)

    def test_bounded_inversion_and_delta_zero_model_contract(self):
        model = PairwiseResidual("full13")
        model.mean = np.zeros(13); model.scale = np.ones(13); model.weights = np.r_[1., np.zeros(12)]
        matrix = np.zeros((60, 13))
        matrix[:, 0] = np.linspace(-2, 2, 60)
        order, _ = model.rank(matrix, history_length=20, delta=10, gate=False)
        positions = np.empty(len(order), dtype=int); positions[order] = np.arange(len(order))
        for early in range(60):
            for late in range(early + 20, 60):
                self.assertLess(positions[early], positions[late])
        np.testing.assert_array_equal(model.rank(matrix, history_length=20, delta=0)[0], np.arange(60))

    def test_optimized_and_slow_reference_match(self):
        rng = np.random.default_rng(7)
        features = np.zeros((141, 115), dtype=np.float32)
        features[1:, :96] = rng.normal(size=(140, 96)).astype(np.float32)
        features[1:, :96] /= np.linalg.norm(features[1:, :96], axis=1, keepdims=True)
        features[np.arange(1, 141), 96 + np.arange(140) % 19] = 1
        first_seen = np.full(141, START, dtype=np.int64); first_seen[0] = -1
        events = [(i + 1, START + i * 30, i % 7 + 1, i % 140 + 1, "early", i + 1) for i in range(500)]
        events += [(501 + i, EVAL_START - 3 * HOUR + i * 10, i % 7 + 1, i % 140 + 1, "late", i + 1) for i in range(400)]
        connection = event_db(events)
        try:
            frozen = models()
            ema = HourlyEMA(connection, len(first_seen))
            for offset in (0, HOUR, 2 * HOUR):
                timestamp = EVAL_START + offset
                state = ema.advance(timestamp - HOUR)
                whole, recent = catalogue_orders(state, first_seen, timestamp)
                for user in (1, 2, 99):
                    history = user_history(connection, user, timestamp)
                    pool, _ = select_pool(whole, recent, history)
                    order, score = rank_all(features, history, pool, frozen)
                    ref_pool, ref_order, ref_score = reference_query(connection, first_seen, features, frozen, user, timestamp)
                    np.testing.assert_array_equal(pool, ref_pool)
                    np.testing.assert_array_equal(order, ref_order)
                    np.testing.assert_allclose(score, ref_score, atol=1e-8, rtol=0)
        finally:
            connection.close()


class ReceiptsAndMetricsTests(unittest.TestCase):
    def test_resource_amendment_is_actually_bound_and_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs").mkdir()
            (root / "reports/external").mkdir(parents=True)
            (root / "artifacts/v3").mkdir(parents=True)
            protocol = root / common_module.PROTOCOL
            protocol.write_text("Invented fixed scientific protocol.\n", encoding="utf-8")
            amendment = root / common_module.RESOURCE_AMENDMENT
            amendment.write_text("Invented resource-only amendment: raw rows 20000000.\n", encoding="utf-8")
            source = {"frozen_mind_and_protocol_files": {common_module.PROTOCOL: {
                "bytes": protocol.stat().st_size, "sha256": common_module.sha256(protocol)}}}
            (root / common_module.SOURCE_REPORT).write_text(json.dumps(source), encoding="utf-8")
            (root / "artifacts/v3/selection.json").write_text('{"delta":10,"primary_recipe":"full13+gate"}', encoding="utf-8")
            before = common_module.experiment_bindings(root)
            self.assertEqual(before["resource_amendment_sha256"], common_module.sha256(amendment))
            self.assertEqual(common_module.check_previous_bindings(root, before), before)
            amendment.write_text("Invented resource amendment was changed.\n", encoding="utf-8")
            after = common_module.experiment_bindings(root)
            self.assertEqual([key for key in before if before[key] != after[key]], ["resource_amendment_sha256"])
            with self.assertRaisesRegex(ValueError, "changed between phases"):
                common_module.check_previous_bindings(root, before)

    def test_resource_amendment_missing_is_not_silently_unbound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "reports/external").mkdir(parents=True)
            (root / common_module.SOURCE_REPORT).write_text("{}", encoding="utf-8")
            with patch.object(common_module, "frozen_bindings", return_value={}):
                with self.assertRaises(FileNotFoundError):
                    common_module.experiment_bindings(root)

    def test_four_phase_cli_synthetic_end_to_end_with_128_query_audit(self):
        class Vectorizer:
            def transform(self, titles):
                return csr_matrix(np.array([[1., len(title) % 7 / 7.] if title else [0., 0.] for title in titles], dtype=np.float32))
            def fit(self, *args):
                raise AssertionError("external fitting forbidden")
        class SVD:
            def transform(self, matrix):
                return matrix.toarray()
            def fit(self, *args):
                raise AssertionError("external fitting forbidden")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = root / "data/processed"
            processed.mkdir(parents=True)
            amendment = root / common_module.RESOURCE_AMENDMENT
            amendment.parent.mkdir(parents=True)
            amendment.write_text("Invented resource-only amendment.\n", encoding="utf-8")
            (processed / "category_vocabulary.json").write_text(json.dumps(["[UNKNOWN]"] + [f"category{i}" for i in range(18)]))
            (processed / "manifest.json").write_text(json.dumps({"features": {"text_dimensions": 96, "dimensions": 115}}))
            records = [{"userId": "warmup", "time": START + i, "id": f"article{i:03d}",
                        "title": f"invented title {i}", "category": "norsk"} for i in range(150)]
            records += [{"userId": f"user{i}", "time": EVAL_START - 2 * HOUR, "id": f"article{i:03d}"} for i in range(15)]
            for day in range(3):
                records += [{"userId": f"user{i % 15}", "time": EVAL_START + day * 86400 + i * 50,
                             "id": f"article{(i * 3 + day) % 150:03d}"} for i in range(50)]
            archive(root / "invented.tar.gz", [("invented/one", records)])
            payload = (root / "invented.tar.gz").read_bytes()
            class Response(io.BytesIO):
                status = 200
                headers = {"Content-Length": str(len(payload)), "Last-Modified": "invented"}
                def geturl(self):
                    return acquisition_module.DATA_URL
            legacy_bindings = {"implementation": {"invented.py": "synthetic"}, "frozen_files": {}}
            bindings = {**legacy_bindings, "resource_amendment_sha256": common_module.sha256(amendment)}
            run, target, reports = root / "run", root / "target.tar.gz", root / "public"
            with ExitStack() as stack:
                for module in (common_module, preparation_module):
                    stack.enter_context(patch.object(module, "experiment_bindings", return_value=bindings))
                # The old acquisition receipt remains unchanged and predates
                # the resource amendment; preparation adds the new binding.
                stack.enter_context(patch.object(acquisition_module, "experiment_bindings", return_value=legacy_bindings))
                stack.enter_context(patch.object(acquisition_module, "ARCHIVE_BYTES", len(payload)))
                stack.enter_context(patch.object(acquisition_module.urllib.request, "urlopen", return_value=Response(payload)))
                stack.enter_context(patch.object(preparation_module.joblib, "load", side_effect=[Vectorizer(), SVD()]))
                stack.enter_context(patch.object(evaluation_module, "load_models", return_value=models()))
                for phase in ("acquisition", "prepare", "evaluate", "aggregate"):
                    with patch.object(sys, "argv", ["external", phase, "--repo", str(root), "--run-dir", str(run),
                                                   "--archive", str(target), "--reports", str(reports)]):
                        external_main()
            report = json.loads((reports / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["overall"]["queries"], 150)
            self.assertEqual(report["audit"]["queries"], 128)
            self.assertLessEqual(report["audit"]["score_max_absolute_error"], 1e-8)
            self.assertFalse(report["external_model_fitting"])
            self.assertFalse(report["prospective_future_generalization"])
            self.assertTrue(report["audit"]["all_method_per_query_hit100_invariant"])
            self.assertIn("preparation_memory", report)
            self.assertEqual(set(report["by_day"]), {"2017-01-05", "2017-01-06", "2017-01-07"})
            for name in ("acquisition", "prepare", "evaluate", "aggregate"):
                self.assertTrue((run / name / "receipt.json").exists())
            acquired = json.loads((run / "acquisition/receipt.json").read_text(encoding="utf-8"))
            prepared = json.loads((run / "prepare/receipt.json").read_text(encoding="utf-8"))
            self.assertNotIn("resource_amendment_sha256", acquired["bindings"])
            self.assertEqual(prepared["bindings"]["resource_amendment_sha256"], common_module.sha256(amendment))

    def test_append_only_failure_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                with stage(folder, "test"):
                    raise RuntimeError("synthetic failure")
            failure = json.loads((Path(folder) / "test/failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            with self.assertRaises(FileExistsError):
                with stage(folder, "test"):
                    pass

    def test_acquisition_streams_bytes_without_tar_or_label_read(self):
        payload = b"invented compressed bytes are never parsed"
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Length": str(len(payload)), "Last-Modified": "synthetic"}
            def geturl(self): return acquisition_module.DATA_URL
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(acquisition_module, "ARCHIVE_BYTES", len(payload)), \
                 patch.object(acquisition_module, "experiment_bindings", return_value={}), \
                 patch.object(acquisition_module.urllib.request, "urlopen", return_value=Response(payload)), \
                 patch.object(tarfile, "open", side_effect=AssertionError("must not inspect archive")):
                result = acquisition_module.acquire(root, root / "run", root / "raw/data.tar.gz")
            self.assertFalse(result["labels_inspected"])
            self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual((root / "raw/data.tar.gz").read_bytes(), payload)
            self.assertTrue((root / "run/acquisition/receipt.json").exists())

    def test_bootstrap_uses_user_clusters_and_request_denominator(self):
        result = paired_interval(np.array([1, 1, 2]), np.array([1, 1, -1]))
        self.assertEqual(result["difference"], 1. / 3.)
        self.assertEqual(result["user_clusters"], 2)
        self.assertEqual(result["ci95"], [-1., 1.])
        self.assertEqual(result, paired_interval(np.array([1, 1, 2]), np.array([1, 1, -1])))

    def test_aggregate_keeps_unreachable_targets_in_denominator(self):
        hit10 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.uint8)
        hit100 = np.array([[1] * 5, [0] * 5, [1] * 5])
        rr = hit10.astype(float)
        result = summarize([1, 2, 3], hit10, hit100, rr)
        self.assertEqual(result["methods"]["baseline"]["denominator"], 3)
        self.assertEqual(result["methods"]["baseline"]["hit_rate10"], 1. / 3.)
        self.assertEqual(result["methods"]["full13+gate"]["hit_rate10"], 2. / 3.)
        self.assertAlmostEqual(result["comparisons"]["full13+gate_minus_baseline"]["difference"], 1. / 3.)
        hit100[0, 1] = 0
        with self.assertRaises(ValueError):
            summarize([1, 2, 3], hit10, hit100, rr)


if __name__ == "__main__":
    unittest.main()
