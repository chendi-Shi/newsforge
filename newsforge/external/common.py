"""File bindings and append-only receipts for the Adressa experiment."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

HOUR = 3600
DAY = 24 * HOUR
START = 1483228800  # 2017-01-01 00:00:00 UTC
EVAL_START = START + 4 * DAY
END = START + 7 * DAY
DAYS = ("2017-01-05", "2017-01-06", "2017-01-07")
MAX_PER_DAY = 5000
MAX_LINES = 10_000_000
MAX_ARTICLES = 200_000
MAX_USERS = 2_000_000
MAX_EXPANDED_BYTES = 25 * (1 << 30)
MAX_LINE_BYTES = 16 * (1 << 20)
ARCHIVE_BYTES = 1_414_905_649
DATA_URL = "https://reclab.idi.ntnu.no/dataset/one_week.tar.gz"
DECAY = 2.0 ** (-1.0 / 12.0)
METHODS = ("baseline", "full13+gate", "no_category+gate", "no_recent+gate", "full13+no_gate")
SOURCE_REPORT = "reports/external/adressa_source.json"
PROTOCOL = "docs/EXTERNAL_ADRESSA_PROTOCOL.md"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
            "numeric_threads": 1}


def frozen_bindings(repo):
    """Check actual bytes before loading trusted local serialized models."""
    repo = Path(repo)
    source = read_json(repo / SOURCE_REPORT)
    bindings = source["frozen_mind_and_protocol_files"]
    for relative, record in bindings.items():
        path = repo / relative
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("frozen MIND or protocol binding mismatch: " + relative)
    selection = read_json(repo / "artifacts/v3/selection.json")
    if selection["delta"] != 10 or selection["primary_recipe"] != "full13+gate":
        raise ValueError("unexpected frozen ranking selection")
    return {relative: record["sha256"] for relative, record in bindings.items()}


def implementation_bindings():
    folder = Path(__file__).resolve().parent
    return {"newsforge/external/" + path.name: sha256(path)
            for path in sorted(folder.glob("*.py"))}


def experiment_bindings(repo):
    result = {"frozen_files": frozen_bindings(repo), "implementation": implementation_bindings(),
              "source_report_sha256": sha256(Path(repo) / SOURCE_REPORT), "environment": environment()}
    lock = Path(repo) / "requirements-delivery-windows.lock"
    result["dependency_lock_sha256"] = sha256(lock) if lock.exists() else None
    return result


def check_previous_bindings(repo, previous):
    current = experiment_bindings(repo)
    if current != previous:
        raise ValueError("experiment inputs, implementation or environment changed between phases")
    return current


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def day_string(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")


def sample_key(timestamp, member, line):
    payload = f"newsforge-adressa-v1|{day_string(timestamp)}|{member}|{line}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


class Log:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8", newline="\n")

    def __call__(self, message, **fields):
        record = {"utc": utc_now(), "message": message, **fields}
        line = json.dumps(record, ensure_ascii=True, allow_nan=False)
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def close(self):
        self.stream.close()


@contextmanager
def stage(run_dir, name):
    """Existing stage directories are evidence and must never be overwritten."""
    directory = Path(run_dir) / name
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    log = Log(directory / "progress.jsonl")
    began = time.perf_counter()
    log("stage_started", stage=name)
    try:
        yield directory, log
    except BaseException as exc:
        # Avoid source records, absolute host paths and exception messages in public receipts.
        write_new_json(directory / "failure.json", {
            "status": "failed", "stage": name, "error_type": type(exc).__name__,
            "elapsed_seconds": time.perf_counter() - began,
            "note": "Retain this directory; inspect the private process log for details."
        })
        log("stage_failed", error_type=type(exc).__name__)
        raise
    else:
        log("stage_completed", elapsed_seconds=time.perf_counter() - began)
    finally:
        log.close()


def file_records(folder, names):
    return {name: {"sha256": sha256(Path(folder) / name),
                   "bytes": (Path(folder) / name).stat().st_size} for name in names}


def check_files(folder, records):
    for name, expected in records.items():
        path = Path(folder) / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("phase artifact binding mismatch: " + name)

