"""Bounded parallel HTTP ranges of the same official opaque archive bytes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from queue import Empty, Queue
import threading
import time
import urllib.error
import urllib.request

from .common import (ARCHIVE_BYTES, DATA_URL, experiment_bindings, sha256, stage,
                     utc_now, write_new_json)


def byte_ranges(total, chunk_bytes):
    return [(start, min(start + chunk_bytes, total) - 1) for start in range(0, total, chunk_bytes)]


def validate_response(response, start, end, total, etag, modified):
    if response.status != 206 or response.geturl() != DATA_URL:
        raise ValueError("range request must return 206 from the exact official URL")
    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
        raise ValueError("range response offset or total mismatch")
    if int(response.headers.get("Content-Length", -1)) != end - start + 1:
        raise ValueError("range response Content-Length mismatch")
    if etag is not None and response.headers.get("ETag") != etag:
        raise ValueError("official archive ETag changed")
    if modified is not None and response.headers.get("Last-Modified") != modified:
        raise ValueError("official archive Last-Modified changed")


def download_part(parts, start, end, total, etag, modified, progress, stop, *, attempts=3):
    basename = f"{start:012d}-{end:012d}"
    expected = end - start + 1
    for attempt in range(1, attempts + 1):
        if stop.is_set():
            raise RuntimeError("parallel acquisition cancelled after another part failed")
        partial = Path(parts) / f"{basename}.attempt{attempt}.partial"
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": "NewsForge-research/1.0"}
        if etag is not None and not etag.startswith("W/"):
            headers["If-Match"] = etag
        elif modified is not None:
            headers["If-Unmodified-Since"] = modified
        received, digest = 0, hashlib.sha256()
        try:
            request = urllib.request.Request(DATA_URL, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
                validate_response(response, start, end, total, etag, modified)
                while received < expected:
                    if stop.is_set():
                        raise RuntimeError("parallel acquisition cancelled")
                    block = response.read(min(64 * 1024, expected - received))
                    if not block:
                        raise IOError("truncated official range response")
                    received += len(block)
                    stream.write(block)
                    digest.update(block)
                    progress.put(("bytes", len(block)))
                if response.read(1):
                    raise ValueError("range response exceeded its advertised interval")
            if received != expected or sha256(partial) != digest.hexdigest():
                raise ValueError("range file verification failed")
            destination = Path(parts) / (basename + ".bin")
            if destination.exists():
                raise FileExistsError("completed part would be overwritten")
            partial.rename(destination)
            record = {"start": start, "end_inclusive": end, "bytes": received,
                      "filename": destination.name, "sha256": digest.hexdigest(), "attempts": attempt}
            progress.put(("part", record))
            return record
        except (urllib.error.URLError, TimeoutError, IOError) as exc:
            # Malformed ranges and validator changes are not retried. Failed
            # network attempts remain under unique names for forensic evidence.
            if isinstance(exc, FileExistsError):
                raise
            if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                raise
            progress.put(("retry", {"start": start, "attempt": attempt, "error_type": type(exc).__name__}))
            if attempt == attempts:
                raise
            if stop.wait(min(2 ** attempt, 8)):
                raise RuntimeError("parallel acquisition cancelled") from exc
    raise AssertionError("unreachable range attempt loop")


def assemble(parts, records, target, total):
    target = Path(target)
    if target.exists():
        raise FileExistsError("archive destination exists")
    partial = target.with_name(target.name + ".parallel.partial")
    digest, offset = hashlib.sha256(), 0
    ordered = sorted(records, key=lambda record: record["start"])
    with partial.open("xb") as stream:
        for record in ordered:
            path = Path(parts) / record["filename"]
            if record["start"] != offset or record["end_inclusive"] != offset + record["bytes"] - 1:
                raise ValueError("range manifest has a gap, overlap or invalid length")
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError("completed range corrupted before assembly")
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    stream.write(block)
                    digest.update(block)
            offset += record["bytes"]
    if offset != total or partial.stat().st_size != total or sha256(partial) != digest.hexdigest():
        raise ValueError("assembled archive length or second SHA256 verification failed")
    if target.exists():
        raise FileExistsError("archive destination appeared during assembly")
    partial.rename(target)
    return digest.hexdigest()


def acquire_parallel(repo, run_dir, archive, *, workers=8, chunk_bytes=4 * (1 << 20)):
    if not 1 <= workers <= 16 or not 64 * 1024 <= chunk_bytes <= 16 * (1 << 20):
        raise ValueError("workers or chunk size outside bounded acquisition limits")
    with stage(run_dir, "acquisition") as (out, log):
        bindings = experiment_bindings(repo)
        target = Path(archive)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError("archive destination already exists")
        started = time.perf_counter()
        request = urllib.request.Request(DATA_URL, method="HEAD", headers={"User-Agent": "NewsForge-research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200 or response.geturl() != DATA_URL:
                raise ValueError("unexpected official archive HEAD response")
            total = int(response.headers.get("Content-Length", -1))
            etag, modified = response.headers.get("ETag"), response.headers.get("Last-Modified")
            if total != ARCHIVE_BYTES or not (etag or modified):
                raise ValueError("official archive size changed or has no consistency validator")
        parts = out / "parts"
        parts.mkdir()
        progress, stop = Queue(), threading.Event()
        results = []
        traffic = 0
        ranges = byte_ranges(total, chunk_bytes)
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="adressa-range")
        futures = [executor.submit(download_part, parts, start, end, total, etag, modified,
                                   progress, stop) for start, end in ranges]
        try:
            while len(results) < len(ranges):
                until = time.monotonic() + 15
                while time.monotonic() < until and len(results) < len(ranges):
                    for future in futures:
                        if future.done() and future.exception() is not None:
                            raise future.exception()
                    try:
                        kind, value = progress.get(timeout=min(1., max(.01, until - time.monotonic())))
                    except Empty:
                        continue
                    if kind == "bytes":
                        traffic += value
                    elif kind == "part":
                        results.append(value)
                    else:
                        log("range_retry", **value)
                log("parallel_download_progress", verified_parts=len(results), total_parts=len(ranges),
                    verified_bytes=sum(record["bytes"] for record in results), network_bytes=traffic,
                    expected_bytes=total, workers=workers, elapsed_seconds=time.perf_counter() - started)
            # Collect every worker exception even if progress delivered the last part first.
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        write_new_json(out / "ranges.json", {"source_url": DATA_URL, "total_bytes": total,
            "etag": etag, "last_modified": modified, "parts": sorted(results, key=lambda r: r["start"])})
        archive_hash = assemble(parts, results, target, total)
        record = {"schema_version": 1, "status": "passed", "dataset": "Adressa light one week",
            "downloaded_utc": utc_now(), "source_url": DATA_URL, "archive_name": target.name,
            "bytes": total, "sha256": archive_hash, "etag": etag, "last_modified": modified,
            "labels_inspected": False, "transport": "bounded verified HTTP byte ranges",
            "workers": workers, "chunk_bytes": chunk_bytes, "range_count": len(results),
            "ranges_manifest_sha256": sha256(out / "ranges.json"),
            "elapsed_seconds": time.perf_counter() - started, "bindings": bindings}
        write_new_json(out / "receipt.json", record)
        log("archive_verified", bytes=total, sha256=archive_hash)
        return record

