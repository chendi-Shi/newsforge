"""Bounded-memory archive parsing, causal identities and frozen text transform."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
import time

import joblib
import numpy as np
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from newsforge.v2.temporal import _MemoryMonitor

from .common import (START, EVAL_START, END, DAYS, MAX_PER_DAY, MAX_LINES,
                     MAX_ARTICLES, MAX_USERS, MAX_EXPANDED_BYTES, MAX_LINE_BYTES,
                     check_previous_bindings, experiment_bindings, file_records, read_json, sample_key,
                     sha256, stage, write_new_json)


def safe_member(name, seen, *, is_directory=False):
    # Permit the conventional POSIX tar './' prefix without changing the raw
    # physical member identity used in deduplication, sampling, and receipts.
    checked = name
    while checked.startswith("./"):
        checked = checked[2:]
    root_directory = checked in ("", ".") and is_directory
    if (not name or "\\" in name or ":" in name or name.startswith("/")
            or PurePosixPath(name).is_absolute()
            or (not root_directory and any(part in ("", ".", "..")
                for part in checked.rstrip("/").split("/")))):
        raise ValueError("unsafe archive member name")
    name.encode("utf-8")
    if name in seen:
        raise ValueError("duplicate archive member name")
    seen.add(name)


def parse_event(value):
    """Return (user, time, article, title, category), or a counted schema reason."""
    if not isinstance(value, dict):
        return None, "not_object"
    user, stamp = value.get("userId"), value.get("time")
    if not isinstance(user, str) or not user.strip():
        return None, "invalid_user"
    if not isinstance(stamp, int) or isinstance(stamp, bool):
        return None, "invalid_timestamp"
    article, alternative = value.get("id"), value.get("documentId")
    if article is not None and not isinstance(article, str):
        return None, "invalid_article"
    if alternative is not None and not isinstance(alternative, str):
        return None, "invalid_article"
    has_article = isinstance(article, str) and bool(article.strip())
    has_alternative = isinstance(alternative, str) and bool(alternative.strip())
    if has_article and has_alternative and article != alternative:
        raise ValueError("conflicting id and documentId schema")
    article = article if has_article else alternative
    if not isinstance(article, str) or not article.strip():
        return None, "invalid_article"
    if not START <= stamp < END:
        return None, "out_of_range"
    title, category = value.get("title"), value.get("category")
    title = title if isinstance(title, str) else ""
    category = category if isinstance(category, str) else ""
    # Escaped unpaired surrogates are not valid UTF-8 identities or article strings.
    try:
        for text in (user, article, title, category):
            text.encode("utf-8")
    except UnicodeEncodeError:
        return None, "invalid_unicode"
    return (user, stamp, article, title, category), None


def connect_new(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("database already exists")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    return connection


def connect_readonly(path):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-32768")
    return connection


def ingest_archive(archive, connection, log):
    connection.execute("""CREATE TABLE raw_events (
        raw_user TEXT NOT NULL COLLATE BINARY, timestamp INTEGER NOT NULL,
        raw_article TEXT NOT NULL COLLATE BINARY, title TEXT NOT NULL, category TEXT NOT NULL,
        member TEXT NOT NULL COLLATE BINARY, line INTEGER NOT NULL,
        PRIMARY KEY (raw_user, timestamp, raw_article)) WITHOUT ROWID""")
    statement = """INSERT INTO raw_events VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(raw_user,timestamp,raw_article) DO UPDATE SET
        title=excluded.title, category=excluded.category, member=excluded.member, line=excluded.line
        WHERE excluded.member < raw_events.member COLLATE BINARY
        OR (excluded.member = raw_events.member AND excluded.line < raw_events.line)"""
    counts = Counter()
    members, seen, pending = [], set(), []
    total_bytes = 0
    pending_bytes = 0
    last_log = time.perf_counter()

    def flush():
        nonlocal pending_bytes
        if pending:
            connection.executemany(statement, pending)
            connection.commit()
            pending.clear()
            pending_bytes = 0

    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            safe_member(member.name, seen, is_directory=member.isdir())
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("archive contains a non-regular member")
            if member.size < 0:
                raise ValueError("archive member has negative size")
            total_bytes += member.size
            if total_bytes > MAX_EXPANDED_BYTES:
                raise ValueError("expanded archive budget exceeded")
            digest, actual_size, line_number = hashlib.sha256(), 0, 0
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("regular archive member has no stream")
            with stream:
                while True:
                    line = stream.readline(MAX_LINE_BYTES + 1)
                    if not line:
                        break
                    if len(line) > MAX_LINE_BYTES:
                        raise ValueError("single line exceeds bounded-parser safety limit")
                    actual_size += len(line)
                    digest.update(line)
                    line_number += 1
                    counts["raw_lines"] += 1
                    if counts["raw_lines"] > MAX_LINES:
                        raise ValueError("raw event line budget exceeded")
                    now = time.perf_counter()
                    if now - last_log >= 15:
                        log("parse_progress", raw_lines=counts["raw_lines"],
                            valid_before_dedup=counts["valid_before_dedup"])
                        last_log = now
                    try:
                        value = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        counts["invalid_json"] += 1
                        continue
                    event, reason = parse_event(value)
                    if reason is not None:
                        counts[reason] += 1
                        continue
                    counts["valid_before_dedup"] += 1
                    counts["empty_title_events"] += not bool(event[3])
                    counts["empty_category_events"] += not bool(event[4])
                    pending.append((*event, member.name, line_number))
                    pending_bytes += len(line)
                    if len(pending) >= 5000 or pending_bytes >= 8 * (1 << 20):
                        flush()
            if actual_size != member.size:
                raise ValueError("archive member size mismatch")
            members.append({"name": member.name, "bytes": actual_size,
                            "sha256": digest.hexdigest(), "lines": line_number})
            log("member_parsed", member=member.name, bytes=actual_size, lines=line_number)
    flush()
    counts["deduplicated_events"] = connection.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
    counts["duplicate_events_removed"] = counts["valid_before_dedup"] - counts["deduplicated_events"]
    counts["expanded_bytes"] = total_bytes
    if not counts["deduplicated_events"]:
        raise ValueError("no valid in-range Adressa events")
    return dict(counts), members


def normalize_database(connection, log):
    started = time.perf_counter()
    last_log = [started]
    def heartbeat():
        now = time.perf_counter()
        if now - last_log[0] >= 15:
            log("normalization_sql_progress", elapsed_seconds=now - started)
            last_log[0] = now
        return 0
    connection.set_progress_handler(heartbeat, 1000000)
    try:
        return _normalize_database(connection, log)
    finally:
        connection.set_progress_handler(None, 0)


def _normalize_database(connection, log):
    users = connection.execute("SELECT COUNT(DISTINCT raw_user) FROM raw_events").fetchone()[0]
    articles = connection.execute("SELECT COUNT(DISTINCT raw_article) FROM raw_events").fetchone()[0]
    if users > MAX_USERS or articles > MAX_ARTICLES:
        raise ValueError("user or article budget exceeded")
    log("normalizing_identities", users=users, articles=articles)
    connection.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY, raw_id TEXT UNIQUE NOT NULL COLLATE BINARY);
        INSERT INTO users(raw_id) SELECT DISTINCT raw_user FROM raw_events ORDER BY raw_user COLLATE BINARY;
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY, raw_id TEXT UNIQUE NOT NULL COLLATE BINARY,
            first_seen INTEGER NOT NULL, title TEXT NOT NULL, category TEXT NOT NULL,
            first_member TEXT NOT NULL, first_line INTEGER NOT NULL);
        INSERT INTO articles(raw_id,first_seen,title,category,first_member,first_line)
            SELECT raw_article,timestamp,title,category,member,line FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY raw_article ORDER BY timestamp,member COLLATE BINARY,line) AS ordinal
                FROM raw_events)
            WHERE ordinal=1 ORDER BY raw_article COLLATE BINARY;
        CREATE TABLE events (
            id INTEGER PRIMARY KEY, timestamp INTEGER NOT NULL, user INTEGER NOT NULL,
            item INTEGER NOT NULL, member TEXT NOT NULL COLLATE BINARY, line INTEGER NOT NULL);
        INSERT INTO events(timestamp,user,item,member,line)
            SELECT r.timestamp,u.id,a.id,r.member,r.line FROM raw_events r
            JOIN users u ON u.raw_id=r.raw_user JOIN articles a ON a.raw_id=r.raw_article
            ORDER BY r.timestamp,r.member COLLATE BINARY,r.line;
        CREATE INDEX events_time ON events(timestamp,item);
        CREATE INDEX events_user_time ON events(user,timestamp,member COLLATE BINARY,line);
        CREATE TABLE sampled (
            event_id INTEGER PRIMARY KEY, day TEXT NOT NULL, sample_key BLOB NOT NULL);
    """)
    connection.create_function("sample_key", 3, sample_key, deterministic=True)
    connection.create_function("utc_day", 1, lambda t: DAYS[(t - EVAL_START) // 86400], deterministic=True)
    populations = {}
    for index, day in enumerate(DAYS):
        lo, hi = EVAL_START + index * 86400, EVAL_START + (index + 1) * 86400
        populations[day] = connection.execute(
            "SELECT COUNT(*) FROM events WHERE timestamp>=? AND timestamp<?", (lo, hi)).fetchone()[0]
        connection.execute("""INSERT INTO sampled
            SELECT id,utc_day(timestamp),sample_key(timestamp,member,line) FROM events
            WHERE timestamp>=? AND timestamp<?
            ORDER BY sample_key(timestamp,member,line),member COLLATE BINARY,line LIMIT ?""",
                           (lo, hi, MAX_PER_DAY))
    connection.execute("DROP TABLE raw_events")
    connection.commit()
    log("database_normalized", users=users, articles=articles, daily_populations=populations)
    return {"users": users, "articles": articles, "daily_populations": populations}


def article_batches(cursor, max_rows=4096, max_bytes=8 * (1 << 20)):
    """Bound both row count and string payload; a single parser-bounded row fits."""
    batch, size = [], 0
    for row in cursor:
        batch.append(row)
        size += len(row[2].encode("utf-8")) + len(row[3].encode("utf-8"))
        if len(batch) >= max_rows or size >= max_bytes:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def transform_articles(connection, repo, out, count, log):
    repo, out = Path(repo), Path(out)
    vectorizer = joblib.load(repo / "data/processed/tfidf.joblib")
    svd = joblib.load(repo / "data/processed/svd.joblib")
    categories = read_json(repo / "data/processed/category_vocabulary.json")
    manifest = read_json(repo / "data/processed/manifest.json")
    text_dim = manifest["features"]["text_dimensions"]
    dimension = manifest["features"]["dimensions"]
    if text_dim != 96 or dimension != text_dim + len(categories) or categories[0] != "[UNKNOWN]":
        raise ValueError("unexpected frozen MIND feature layout")
    category_lookup = {value: index for index, value in enumerate(categories)}
    features = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float32,
                                        shape=(count + 1, dimension))
    flags = np.lib.format.open_memmap(out / "coverage.npy", mode="w+", dtype=np.uint8, shape=(count + 1, 4))
    first_seen = np.lib.format.open_memmap(out / "first_seen.npy", mode="w+", dtype=np.int64, shape=(count + 1,))
    features[:] = 0
    flags[:] = 0
    first_seen[:] = -1
    cursor = connection.execute("SELECT id,first_seen,title,category FROM articles ORDER BY id")
    with threadpool_limits(limits=1):
        for records in article_batches(cursor):
            ids = np.asarray([row[0] for row in records], dtype=np.int64)
            titles = [row[2] for row in records]
            matrix = vectorizer.transform(titles)
            text = normalize(svd.transform(matrix), norm="l2")
            if text.shape[1] > text_dim or not np.isfinite(text).all():
                raise ValueError("invalid external frozen text transform")
            features[ids, :text.shape[1]] = text
            cats = np.asarray([category_lookup.get(row[3], 0) for row in records])
            features[ids, text_dim + cats] = 1
            flags[ids] = np.column_stack((np.asarray([not bool(t) for t in titles]),
                np.asarray(matrix.getnnz(axis=1) == 0), np.linalg.norm(text, axis=1) == 0, cats == 0))
            first_seen[ids] = np.asarray([row[1] for row in records], dtype=np.int64)
            log("text_transform_progress", articles_complete=int(ids[-1]), total_articles=count)
    features.flush(); flags.flush(); first_seen.flush()
    coverage = {name: int(flags[1:, index].sum()) for index, name in enumerate(
        ("empty_title", "zero_tfidf", "zero_svd", "unknown_category"))}
    coverage["articles"] = count
    del features, flags, first_seen
    return {"feature_dimensions": dimension, "text_dimensions": text_dim, "coverage": coverage}


def prepare(repo, run_dir, archive):
    run_dir = Path(run_dir)
    acquisition = read_json(run_dir / "acquisition/receipt.json")
    with stage(run_dir, "prepare") as (out, log), _MemoryMonitor() as memory:
        started = time.perf_counter()
        # Downloading compressed bytes does not inspect event labels. The full
        # implementation may be completed/tested after that download, but must
        # be frozen before this first label-reading phase. All research inputs
        # and the numeric environment are required to remain unchanged.
        bindings = experiment_bindings(repo)
        for key, value in acquisition["bindings"].items():
            if key != "implementation" and bindings.get(key) != value:
                raise ValueError("frozen inputs changed since label-blind acquisition")
        log("implementation_frozen_before_event_read", files=len(bindings["implementation"]))
        if sha256(archive) != acquisition["sha256"] or Path(archive).stat().st_size != acquisition["bytes"]:
            raise ValueError("official archive does not match acquisition receipt")
        connection = connect_new(out / "events.sqlite")
        try:
            counts, members = ingest_archive(archive, connection, log)
            identity = normalize_database(connection, log)
            transformed = transform_articles(connection, repo, out, identity["articles"], log)
            digest = hashlib.sha256()
            daily_samples = Counter()
            for member, line, day in connection.execute("""SELECT e.member,e.line,s.day FROM sampled s
                    JOIN events e ON e.id=s.event_id ORDER BY e.timestamp,e.member COLLATE BINARY,e.line"""):
                digest.update(json.dumps([member, line], ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
                daily_samples[day] += 1
            if sum(daily_samples.values()) < 128:
                raise ValueError("fewer than 128 target-independent evaluation requests")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ValueError("SQLite integrity check failed")
        finally:
            connection.close()
        check_previous_bindings(repo, bindings)
        if sha256(archive) != acquisition["sha256"]:
            raise ValueError("archive changed during preparation")
        receipt = {"schema_version": 1, "status": "passed", "bindings": bindings,
            "archive_sha256": acquisition["sha256"], "acquisition_receipt_sha256": sha256(run_dir / "acquisition/receipt.json"),
            "counts": counts, "identity": identity, "transformed": transformed,
            "archive_members": members, "daily_samples": dict(daily_samples),
            "sample_identity_sha256": digest.hexdigest(),
            "elapsed_seconds": time.perf_counter() - started, "memory": memory.report(),
            "files": file_records(out, ("events.sqlite", "features.npy", "coverage.npy", "first_seen.npy")),
            "target_independent_sampling": True, "external_model_fitting": False}
        write_new_json(out / "receipt.json", receipt)
        return receipt
