"""Stream the single official archive without extraction or label inspection."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
import urllib.request

from .common import (ARCHIVE_BYTES, DATA_URL, experiment_bindings, sha256, stage,
                     utc_now, write_new_json)


def acquire(repo, run_dir, archive):
    with stage(run_dir, "acquisition") as (out, log):
        bindings = experiment_bindings(repo)
        target = Path(archive)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError("archive already exists; use its original acquisition receipt")
        partial = target.with_name(target.name + ".partial")
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "NewsForge-research/1.0"})
        digest = hashlib.sha256()
        received = 0
        start = time.perf_counter()
        last_log = start
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
            if response.status != 200 or response.geturl() != DATA_URL:
                raise ValueError("unexpected official archive response or redirect")
            declared = int(response.headers.get("Content-Length", -1))
            if declared != ARCHIVE_BYTES:
                raise ValueError("archive size changed since preregistration; retain failure and investigate")
            modified = response.headers.get("Last-Modified")
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                received += len(block)
                if received > ARCHIVE_BYTES:
                    raise ValueError("archive exceeds declared size")
                stream.write(block)
                digest.update(block)
                now = time.perf_counter()
                if now - last_log >= 15:
                    log("download_progress", bytes=received, expected_bytes=ARCHIVE_BYTES,
                        elapsed_seconds=now - start)
                    last_log = now
        if received != ARCHIVE_BYTES:
            raise ValueError("incomplete archive download")
        if sha256(partial) != digest.hexdigest():
            raise ValueError("download write verification failed")
        # Both paths are explicit siblings in the requested raw-data directory.
        # rename refuses existing targets on Windows; no replacement is authorized.
        if target.exists():
            raise FileExistsError("archive destination was created during download")
        partial.rename(target)
        record = {"schema_version": 1, "status": "passed", "dataset": "Adressa light one week",
                  "downloaded_utc": utc_now(), "source_url": DATA_URL,
                  "archive_name": target.name, "bytes": received, "sha256": digest.hexdigest(),
                  "last_modified": modified, "labels_inspected": False,
                  "elapsed_seconds": time.perf_counter() - start, "bindings": bindings}
        write_new_json(out / "receipt.json", record)
        log("archive_verified", bytes=received, sha256=digest.hexdigest())
        return record
