"""Install a verified raw-pipeline wheel lock in an existing isolated venv.

Every invocation owns new log and receipt paths. Failed attempts are retained;
choose new paths after correcting an error. This command actually reinstalls
packages in its current interpreter and never retroactively certifies an older
installation as having been executed by this command.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import site
import subprocess
import sys
import time
import traceback

from newsforge.delivery import pipeline as p
from newsforge.delivery.acceptance import audit_environment, parse_lock, validate_venv
from newsforge.delivery.raw_acceptance import validate_environment_receipt
from newsforge.delivery.wheels import wheel_records


INSTALL_OPTIONS = ("--isolated", "--no-compile", "--force-reinstall", "--no-index",
                   "--find-links", "--require-hashes")


def check_interpreter(root):
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise ValueError("clear Python path overrides before isolated installation")
    if sys.version_info[:2] != (3, 10) or sys.platform != "win32":
        raise ValueError("this wheel lock requires Windows CPython 3.10")
    return validate_venv(sys.prefix, sys.base_prefix, sys.executable, root, sys.path,
                         user_site_enabled=site.ENABLE_USER_SITE)


def verify_inputs(wheelhouse, lock, wheel_receipt):
    """Verify real wheel bytes and metadata before allowing pip to modify a venv."""
    expected = parse_lock(lock)
    recorded = p.read(wheel_receipt)
    lock_sha256 = p.digest(lock)
    if recorded.get("schema") != 1 or recorded.get("lock_sha256") != lock_sha256:
        raise ValueError("wheel receipt schema or locked requirements hash differs")
    actual = wheel_records(wheelhouse)
    rows = recorded.get("wheels", [])
    if (len(rows) != len(expected) or len(actual) != len(expected)
            or {row.get("name") for row in rows} != set(expected)
            or any(row.get("version") != expected[row["name"]] for row in rows)
            or sorted(rows, key=lambda row: row["name"]) != actual):
        raise ValueError("wheelhouse bytes, distribution metadata or inventory differ from the receipt")
    # parse_lock enforces pin/hash syntax; every actual wheel must be an allowed
    # hash for its own named requirement, not merely present in a coherent receipt.
    import re
    logical = re.sub(r"\\\r?\n[ \t]*", " ", Path(lock).read_text(encoding="utf-8"))
    hashes = {}
    for line in logical.splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name = re.sub(r"[-_.]+", "-", line.split("==", 1)[0].strip()).lower()
            hashes[name] = {value.lower() for value in re.findall(r"--hash=sha256:([0-9a-fA-F]{64})", line)}
    if any(row["sha256"] not in hashes[row["name"]] for row in actual):
        raise ValueError("wheel bytes are not allowed by their locked requirement")
    return {"lock_sha256": lock_sha256, "wheel_receipt_sha256": p.digest(wheel_receipt),
            "verified_wheels": len(actual)}


def run(*, wheelhouse, lock, wheel_receipt, output, log, root=None, timeout=1200):
    root = Path.cwd().resolve() if root is None else Path(root).resolve()
    wheelhouse, lock, wheel_receipt, output, log = (
        Path(value).resolve() for value in (wheelhouse, lock, wheel_receipt, output, log))
    started_path, failed_path = output.with_suffix(".started.json"), output.with_suffix(".failed.json")
    outputs = (output, log, started_path, failed_path)
    if len(set(outputs)) != len(outputs) or set(outputs) & {lock, wheel_receipt}:
        raise ValueError("log and all receipt paths must be distinct from one another and from inputs")
    if any(path.exists() for path in outputs):
        raise FileExistsError("retain previous installation evidence; choose new output and log paths")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive installation timeout required")
    p.write_new(started_path, {"status": "running", "created_utc": p.utc(),
                              "installer_source_sha256": p.digest(__file__),
                              "requested_action": "hash_checked_reinstall",
                              "install_options": list(INSTALL_OPTIONS)})
    log.parent.mkdir(parents=True, exist_ok=True)
    phase, install_returncode, check_returncode = "preflight", None, None
    started = time.perf_counter()
    try:
        with log.open("xb") as stream:
            try:
                check_interpreter(root)
                inputs = verify_inputs(wheelhouse, lock, wheel_receipt)
                command = [sys.executable, "-m", "pip", "--isolated", "install", "--no-compile",
                           "--force-reinstall", "--no-index", "--find-links", str(wheelhouse),
                           "--require-hashes", "-r", str(lock)]
                child = {"cwd": str(root), "check": False,
                         "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
                phase = "pip-install"
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=timeout, **child)
                install_returncode = result.returncode
                stream.flush()
                if result.returncode:
                    raise RuntimeError("pip install failed; inspect the retained private log")
                phase = "environment-audit"
                importlib.invalidate_caches()
                environment = audit_environment(root, lock)
                phase = "pip-check"
                check = subprocess.run([sys.executable, "-m", "pip", "check"],
                                       capture_output=True, text=True, timeout=120, **child)
                check_returncode = check.returncode
                if check.returncode:
                    stream.write((check.stdout + check.stderr).encode("utf-8", errors="replace"))
                    raise RuntimeError("pip check failed; inspect the retained private log")
                phase = "final-verification"
                if verify_inputs(wheelhouse, lock, wheel_receipt) != inputs:
                    raise ValueError("installation inputs changed during execution")
                receipt = {"passed": True, "created_utc": p.utc(),
                           "scope": "Fresh hash-checked reinstall in an isolated Windows CPython 3.10 venv for raw-pipeline reproduction",
                           "installation_executed_by_this_cli": True,
                           "installer_source_sha256": p.digest(__file__),
                           "started_receipt_sha256": p.digest(started_path),
                           "environment": environment,
                           "lock_sha256": inputs["lock_sha256"],
                           "wheel_receipt_sha256": inputs["wheel_receipt_sha256"],
                           "install_log_sha256": p.digest(log), "install_options": list(INSTALL_OPTIONS),
                           "pip_install": {"returncode": install_returncode},
                           "pip_check": {"returncode": check.returncode, "stdout": check.stdout, "stderr": check.stderr},
                           "elapsed_seconds": time.perf_counter() - started}
                validate_environment_receipt(receipt, lock, wheel_receipt_path=wheel_receipt, install_log_path=log)
                p.write_new(output, receipt)
                return receipt
            except BaseException:
                stream.write(traceback.format_exc().encode("utf-8", errors="replace"))
                stream.flush()
                raise
    except BaseException as exc:
        p.write_new(failed_path, {"passed": False, "status": "failed", "created_utc": p.utc(),
                                 "phase": phase, "error_type": type(exc).__name__,
                                 "error": "Installation did not complete; inspect the retained private log.",
                                 "pip_install_returncode": install_returncode,
                                 "pip_check_returncode": check_returncode,
                                 "install_log_sha256": p.digest(log) if log.is_file() else None,
                                 "started_receipt_sha256": p.digest(started_path),
                                 "elapsed_seconds": time.perf_counter() - started,
                                 "partial_environment_and_logs_retained": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--wheel-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New installation JSON; existing attempts cannot be reused")
    parser.add_argument("--log", type=Path, required=True, help="New private pip stdout/stderr log")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Checkout root for import-origin verification")
    parser.add_argument("--timeout", type=float, default=1200, help="pip install timeout in seconds")
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"passed": True, "locked_distributions": result["environment"]["locked_distributions"],
                      "lock_sha256": result["lock_sha256"]}))


if __name__ == "__main__":
    main()
