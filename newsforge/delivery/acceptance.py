"""Accept an isolated clean clone against the frozen v3 local service release.

This validates installation and service delivery, not retraining or production
deployment. All subprocess logs and request-level evidence stay under artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


IMPORT_PROBES = ("numpy", "scipy", "sklearn", "torch", "faiss", "fastapi",
                 "pydantic", "uvicorn", "psutil", "httpx", "joblib", "threadpoolctl")
HTTP_CONFIGURATIONS = {(1, 1), (8, 1), (1, 4), (8, 4)}
CONTRACT_NAMES = {"oov_history", "pad_history", "negative_timestamp", "before_coverage",
                  "after_coverage", "invalid_top_k", "empty_batch", "oversized_batch",
                  "empty_history", "coverage_minimum", "coverage_maximum",
                  "hour_boundary_minus_one", "hour_boundary_exact"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def public_path(path, *, prefix, base, root):
    """Classify actual resolved locations before producing a portable receipt."""
    path = Path(path).resolve()
    if inside(path, prefix):
        return "<venv>/" + path.relative_to(Path(prefix).resolve()).as_posix()
    if any(part.lower() in {"site-packages", "dist-packages"} for part in path.parts):
        raise ValueError("third-party path is outside the isolated virtual environment")
    if inside(path, root):
        return "<clone>/" + path.relative_to(Path(root).resolve()).as_posix()
    if inside(path, base):
        return "<base>/" + path.relative_to(Path(base).resolve()).as_posix()
    raise ValueError("import location is outside the virtual environment, clone and standard library")


def validate_venv(prefix, base, executable, root, search_path, *, user_site_enabled):
    prefix, base, root = Path(prefix).resolve(), Path(base).resolve(), Path(root).resolve()
    if prefix == base or not inside(executable, prefix):
        raise ValueError("acceptance requires an independent virtual environment interpreter")
    config = prefix / "pyvenv.cfg"
    settings = {}
    for line in config.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip().lower()] = value.strip().lower()
    if settings.get("include-system-site-packages") != "false":
        raise ValueError("include-system-site-packages must be explicitly false")
    if user_site_enabled is not False:
        raise ValueError("the Python user site must be disabled")
    checked_paths = [public_path(value or root, prefix=prefix, base=base, root=root)
                     for value in search_path]
    return {"isolated_prefix": True, "include_system_site_packages": False,
            "user_site_enabled": False,
            "executable": public_path(executable, prefix=prefix, base=base, root=root),
            "search_path": checked_paths, "pyvenv_config_sha256": sha256(config)}


def parse_lock(path):
    """Require an exact pin and complete SHA-256 for every logical requirement."""
    text = Path(path).read_text(encoding="utf-8")
    logical_lines = re.sub(r"\\\r?\n[ \t]*", " ", text).splitlines()
    requirements = {}
    for line in logical_lines:
        line = line.strip()
        if (not line or line.startswith("#") or line == "--only-binary=:all:"
                or line.startswith(("--index-url ", "--extra-index-url "))):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\;]+)"
                             r"((?:\s+--hash=sha256:[0-9a-fA-F]{64})+)", line)
        if not match:
            raise ValueError("every delivery wheel-hash lock requirement needs an exact pin and SHA-256")
        name, version = match.group(1, 2)
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in requirements:
            raise ValueError("duplicate requirement in delivery lock")
        requirements[normalized] = version
    if not requirements:
        raise ValueError("delivery acceptance requires a wheel-hash lock")
    return requirements


def module_record(name, module, *, prefix, base, root):
    location = vars(module).get("__file__")
    if location is None:
        return None
    path = Path(location)
    if not path.exists():
        # Torch registers these two module objects dynamically. Their nominal
        # relative __file__ values are labels, not paths used by its loader.
        dynamic = {"torch.ops": ("torch._ops", "_Ops"),
                   "torch.classes": ("torch._classes", "_Classes")}
        if name in dynamic:
            provider_name, class_name = dynamic[name]
            provider = sys.modules.get(provider_name)
            if provider is None or type(module) is not vars(provider).get(class_name):
                raise ValueError("dynamic Torch module does not match its registered provider")
            provider_path = vars(provider).get("__file__")
            if provider_path is None or not Path(provider_path).is_file() or not inside(provider_path, prefix):
                raise ValueError("dynamic Torch provider is outside the isolated virtual environment")
            return {"dynamic_module_type": provider_name + "." + class_name,
                    "provider_path": public_path(provider_path, prefix=prefix, base=base, root=root),
                    "provider_sha256": sha256(provider_path)}
        # Frozen stdlib code can retain a non-existent build-time filename.
        if getattr(vars(module).get("__spec__"), "origin", None) == "frozen":
            return None
        raise ValueError("loaded module has a non-existent source location")
    return {"path": public_path(path, prefix=prefix, base=base, root=root), "sha256": sha256(path)}


def audit_environment(root, lock_path):
    import site

    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise ValueError("clear PYTHONPATH and PYTHONHOME before clean-environment acceptance")
    root = Path(root).resolve()
    result = validate_venv(sys.prefix, sys.base_prefix, sys.executable, root, sys.path,
                           user_site_enabled=site.ENABLE_USER_SITE)
    result["python_version"] = sys.version.split()[0]
    result["implementation"] = sys.implementation.name
    result["platform"] = sys.platform
    if sys.version_info[:2] != (3, 10) or sys.platform != "win32":
        raise ValueError("this wheel lock targets Windows CPython 3.10")
    expected = parse_lock(lock_path)
    distributions = {}
    for dist in importlib.metadata.distributions():
        location = Path(dist.locate_file("")).resolve()
        if not inside(location, sys.prefix):
            raise ValueError("an installed distribution is inherited from outside the virtual environment")
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name in distributions:
            raise ValueError("duplicate installed distribution")
        distributions[name] = {"version": dist.version,
                               "location": public_path(location, prefix=sys.prefix,
                                                       base=sys.base_prefix, root=root)}
    for name, version in expected.items():
        if distributions.get(name, {}).get("version") != version:
            raise ValueError("installed dependency does not match the delivery lock")
    if set(distributions) - set(expected) - {"pip"}:
        raise ValueError("installed environment contains an unlocked dependency")
    for name in IMPORT_PROBES:
        module = importlib.import_module(name)
        if not getattr(module, "__file__", None) or not inside(module.__file__, sys.prefix):
            raise ValueError("a third-party import is not supplied by the isolated virtual environment")
    modules = {}
    for name, module in sorted(list(sys.modules.items())):
        if module is not None:
            record = module_record(name, module, prefix=sys.prefix, base=sys.base_prefix, root=root)
            if record is not None:
                modules[name] = record
    result.update({"wheel_lock_sha256": sha256(lock_path), "locked_distributions": len(expected),
                   "distributions": distributions, "import_probes": list(IMPORT_PROBES),
                   "loaded_modules": modules, "passed": True})
    return result


def git_output(git, root, *arguments):
    result = subprocess.run([git, "-C", str(root), *arguments], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False, timeout=30)
    if result.returncode:
        raise RuntimeError("Git verification failed")
    return result.stdout.decode("utf-8")


def check_git_status(status, allowed_outputs):
    allowed = set(allowed_outputs)
    for entry in status.split("\0"):
        if not entry:
            continue
        if entry[:3] != "?? " or entry[3:] not in allowed:
            raise ValueError("the clone has a source change or an unauthorized untracked file")


def audit_source(git, root, expected_commit, allowed_outputs):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit):
        raise ValueError("expected commit must be the complete Git SHA")
    actual = git_output(git, root, "rev-parse", "HEAD").strip()
    if actual != expected_commit.lower():
        raise ValueError("clone HEAD differs from the expected source commit")
    if Path(git_output(git, root, "rev-parse", "--show-toplevel").strip()).resolve() != Path(root).resolve():
        raise ValueError("acceptance must run in the clone's repository root")
    check_git_status(git_output(git, root, "status", "--porcelain=v1", "-z", "--untracked-files=all"),
                     allowed_outputs)
    files = {}
    for name in git_output(git, root, "ls-files", "-z").split("\0"):
        if name:
            path = Path(root, name)
            if not inside(path, root) or not path.is_file():
                raise ValueError("tracked source is missing or resolves outside the clone")
            files[name] = sha256(path)
    return {"commit": actual, "tracked_files": files, "source_clean": True}


def run_logged(command, path, *, timeout):
    # This is the same tested identity-scoped lifecycle used by the frozen
    # benchmark. It never searches for or terminates unrelated Python jobs.
    from newsforge.v2.benchmark import OwnedProcessTree, child_options, stop_owned_process

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **child_options())
        owned = OwnedProcessTree(process)
        try:
            code = owned.wait(timeout=timeout)
        finally:
            stop_owned_process(owned)
    result = {"exit_code": code, "elapsed_seconds": time.perf_counter() - started,
              "private_log_sha256": sha256(path)}
    if code:
        raise RuntimeError("acceptance subprocess failed; inspect its retained private log")
    return result


def validate_http_report(report, raw_dir):
    if report.get("passed") is not True or report.get("benchmark_version") != 3:
        raise ValueError("HTTP benchmark did not report v3 success")
    if report.get("sampling", {}).get("queries") != 64:
        raise ValueError("HTTP acceptance requires all 64 fixed queries")
    if report.get("health", {}).get("api_version") != "v3":
        raise ValueError("HTTP endpoint was not the v3 service")
    causal = report.get("offline_causal_output_check", {})
    if causal.get("passed") is not True or causal.get("queries") != 64:
        raise ValueError("offline causal output validation did not pass")
    contracts = report.get("api_contract_checks", [])
    if (len(contracts) != len(CONTRACT_NAMES) or {c.get("name") for c in contracts} != CONTRACT_NAMES
            or any(c.get("passed") is not True for c in contracts)):
        raise ValueError("the complete HTTP schema/boundary contract was not verified")
    resources = report.get("resources", {})
    if resources.get("sampling_error", "missing") is not None or resources.get("resource_samples", 0) <= 0:
        raise ValueError("HTTP resource sampling did not complete")
    raw_dir = Path(raw_dir)
    for name in ("requests", "offline", "http_records"):
        if sha256(raw_dir / (name + ".json")) != report["private_evidence"][name + "_sha256"]:
            raise ValueError("HTTP private evidence hash mismatch")
    raw = json.loads((raw_dir / "http_records.json").read_text(encoding="utf-8"))
    requests = json.loads((raw_dir / "requests.json").read_text(encoding="utf-8"))
    offline = json.loads((raw_dir / "offline.json").read_text(encoding="utf-8"))
    if len(requests.get("requests", [])) != 64 or len(offline.get("responses", [])) != 64:
        raise ValueError("HTTP request/reference evidence is incomplete")
    configurations = report.get("configurations", [])
    if (len(configurations) != 4 or
            {(c.get("batch_size"), c.get("concurrency")) for c in configurations} != HTTP_CONFIGURATIONS):
        raise ValueError("the complete four-case HTTP matrix was not measured")
    for config in configurations:
        batch, concurrency = config["batch_size"], config["concurrency"]
        if (config.get("rounds") != 3 or config.get("measured_queries") != 192 or
                config.get("measured_http_requests") != 192 // batch or
                any(config.get(key) != 0 for key in ("http_errors", "failed_queries", "offline_http_mismatches"))):
            raise ValueError("HTTP parity matrix contains errors or an incomplete workload")
        records = raw[f"batch{batch}_concurrency{concurrency}"]
        if len(records["warmup"]) != 8 or len(records["measured"]) != 192 // batch:
            raise ValueError("HTTP raw record cardinality differs from the declared workload")
        if any(r.get("status") != 200 or any(r.get(key) != 0 for key in
                                             ("http_error", "query_errors", "mismatches"))
               for r in records["warmup"] + records["measured"]):
            raise ValueError("HTTP raw evidence contains an error or mismatch")
    return {"passed": True, "measured_queries": 768, "configurations": 4,
            "api_contracts": len(contracts), "all_response_fields_exact": True,
            "private_evidence_sha256": report["private_evidence"]}


def relative_output(path, root, parent):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not inside(path, root / parent) or path == root / parent:
        raise ValueError("acceptance output must stay in its designated workspace directory")
    return path.relative_to(root).as_posix()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def parse_test_summary(text):
    """Separate discovered cases from skips and reject unexpected test outcomes."""
    counts = re.findall(r"^Ran (\d+) tests? in ", text, re.M)
    outcomes = re.findall(r"^OK(?: \(([^\n]*)\))?\s*$", text, re.M)
    if len(counts) != 1 or int(counts[0]) == 0 or len(outcomes) != 1:
        raise ValueError("test discovery did not report a unique successful test summary")
    details = outcomes[0]
    skipped = 0
    if details:
        match = re.fullmatch(r"skipped=(\d+)", details)
        if not match:
            raise ValueError("test summary contains outcomes other than passing cases and skips")
        skipped = int(match.group(1))
    count = int(counts[0])
    if skipped >= count:
        raise ValueError("test discovery did not execute any passing cases")
    return {"count": count, "skipped": skipped, "executed_count": count - skipped}


def run(args):
    root = Path.cwd().resolve()
    report_relative = relative_output(args.report, root, "reports/delivery")
    report_path = root / report_relative
    http_path = report_path.with_name(report_path.stem + "_http.json")
    http_relative = http_path.relative_to(root).as_posix()
    raw_dir, logs = root / "artifacts/delivery/acceptance", root / "artifacts/delivery/logs"
    if report_path.exists() or http_path.exists() or raw_dir.exists() or logs.exists():
        raise FileExistsError("acceptance evidence already exists; use a fresh clone")
    git = args.git or shutil.which("git")
    if not git:
        raise ValueError("Git is required; pass --git if it is not on PATH")
    lock_path = root / args.lock
    if not inside(lock_path, root):
        raise ValueError("wheel lock must be in the checked source tree")
    started = time.perf_counter()
    receipt = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
               "scope": "clean-clone installation and frozen-artifact v3 localhost service acceptance",
               "retraining_performed": False, "production_deployment_verified": False,
               "unit_tests_required": not args.skip_tests, "passed": False,
               "status": "failed"}
    stage = "source"
    try:
        source = audit_source(git, root, args.expected_commit, (report_relative, http_relative))
        receipt["source"] = source
        if lock_path.resolve().relative_to(root).as_posix() not in source["tracked_files"]:
            raise ValueError("wheel lock must be tracked in the expected source commit")
        stage = "environment"
        receipt["environment"] = audit_environment(root, lock_path)
        stage = "pip_check"
        receipt["pip_check"] = run_logged([sys.executable, "-m", "pip", "check"],
                                           logs / "pip_check.log", timeout=120)
        stage = "unit_tests"
        if args.skip_tests:
            receipt["unit_tests"] = {"executed": False, "passed": None,
                                      "reason": "explicit --skip-tests; no test completion is claimed"}
        else:
            result = run_logged([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                logs / "unit_tests.log", timeout=args.test_timeout)
            summary = parse_test_summary((logs / "unit_tests.log").read_text(encoding="utf-8"))
            receipt["unit_tests"] = {**result, "executed": True, "passed": True, **summary}
        stage = "http_service"
        receipt["http_subprocess"] = run_logged(
            [sys.executable, "-m", "newsforge.v3.benchmark_service", "--raw-dir", raw_dir.as_posix(),
             "--report", http_relative], logs / "http_service.log", timeout=args.http_timeout)
        stage = "http_evidence"
        receipt["http_acceptance"] = validate_http_report(
            json.loads(http_path.read_text(encoding="utf-8")), raw_dir)
        receipt["http_acceptance"].update({"report": http_relative, "report_sha256": sha256(http_path)})
        stage = "source_recheck"
        if audit_source(git, root, args.expected_commit, (report_relative, http_relative)) != source:
            raise ValueError("source changed during acceptance")
        receipt["passed"] = not args.skip_tests
        receipt["status"] = "partial" if args.skip_tests else "passed"
    except Exception as error:
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / "failure.log").open("x", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        receipt["failure"] = {"stage": stage, "type": type(error).__name__,
                              "private_log_sha256": sha256(logs / "failure.log")}
    receipt["elapsed_seconds"] = time.perf_counter() - started
    write_new(report_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--git", help="Git executable when it is not on PATH")
    parser.add_argument("--lock", default="requirements-delivery-windows.lock")
    parser.add_argument("--report", default="reports/delivery/acceptance.json")
    parser.add_argument("--skip-tests", action="store_true", help="explicit reduced scope; no unit-test success is claimed")
    parser.add_argument("--test-timeout", type=int, default=1800)
    parser.add_argument("--http-timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.test_timeout < 1 or args.http_timeout < 1:
        parser.error("timeouts must be positive seconds")
    receipt = run(args)
    if receipt["status"] == "failed":
        print("Acceptance failed during " + receipt["failure"]["stage"] + "; inspect the retained private log.")
        raise SystemExit(1)
    if receipt["status"] == "partial":
        print("Partial acceptance only: unit tests were skipped; wrote " + args.report)
        raise SystemExit(2)
    print("Acceptance passed; wrote " + args.report)


if __name__ == "__main__":
    main()
