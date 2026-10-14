"""Read-only acceptance of a completed raw MIND reproduction.

Run with the pipeline's own isolated interpreter. This checks retained execution
evidence and rehashes the frozen auditors' bindings; it does not train models,
rerun predictions, or make an already observed benchmark an untouched test set.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import site
import sys

from newsforge.delivery import pipeline as p
from newsforge.delivery.acceptance import IMPORT_PROBES, parse_lock, validate_http_report, validate_venv


RAW_NAMES = {f"{split}/{name}" for split in ("train", "dev")
             for name in ("news.tsv", "behaviors.tsv")}
LOCK = "requirements-raw-windows.lock"
TEST_LOGS = {"unit-tests": "reports/delivery/all_unit_tests.txt",
             "research-unit-tests": "reports/v3_release_unit_tests.txt"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "evidence timestamps require a timezone")
    return result


class Evidence:
    """Hash each immutable file once, recording its stat identity for final check."""

    def __init__(self, workspace):
        self.root = Path(workspace).resolve()
        self.hashes, self.stats = {}, {}

    def path(self, name):
        return p.inside(self.root, name)

    @staticmethod
    def identity(path):
        value = path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def digest(self, name, expected=None):
        name = Path(name).as_posix()
        path = self.path(name)
        stat = self.identity(path)
        if name not in self.hashes:
            self.hashes[name] = p.digest(path)
            self.stats[name] = stat
        require(self.stats[name] == stat, f"evidence changed during acceptance: {name}")
        require(expected is None or self.hashes[name] == expected, f"SHA-256 mismatch: {name}")
        return self.hashes[name]

    def read(self, name):
        self.digest(name)
        return p.read(self.path(name))

    def finish(self):
        for name, identity in self.stats.items():
            require(self.identity(self.path(name)) == identity,
                    f"evidence changed during acceptance: {name}")


def receipt_chain(evidence, manifest, *, definitions=None, target="v3-audit", python=None):
    """Validate complete retained execution, not merely a final `passed` flag."""
    definitions = p.ordered_stages(target, definitions)
    manifest_sha = evidence.digest(p.STATE / "workspace.json")
    python = str(Path(sys.executable if python is None else python).resolve())
    receipt_hashes, records, public = {}, {}, []
    previous_finish = timestamp(manifest["created_utc"])
    for stage in definitions:
        directory = p.STATE / stage.name
        require(not evidence.path(directory / "failed.json").exists(),
                f"failed stage retained in supposedly successful chain: {stage.name}")
        started = evidence.read(directory / "started.json")
        result = evidence.read(directory / "success.json")
        require(started.get("status") == "running", f"invalid stage start: {stage.name}")
        require(result.get("status") == "success" and result.get("returncode") == 0,
                f"stage did not succeed: {stage.name}")
        require(all(result.get(key) == value for key, value in started.items() if key != "status"),
                f"start/success identities differ: {stage.name}")
        expected_dependencies = {name: receipt_hashes[name] for name in stage.dependencies}
        expected_inputs = {}
        for name in stage.dependencies:
            for path, digest in records[name]["outputs"].items():
                require(path not in expected_inputs or expected_inputs[path] == digest,
                        "conflicting dependency output identities")
                expected_inputs[path] = digest
        argv = [python, *stage.argv]
        if stage.name == "data" and manifest["raw_mode"] == "download":
            argv.append("--download")
        require(result.get("stage") == stage.name and result.get("python") == python
                and result.get("argv") == argv, f"stage command/interpreter differs: {stage.name}")
        require(result.get("stage_definition_sha256") == p.canonical_hash(asdict(stage))
                and result.get("workspace_manifest_sha256") == manifest_sha,
                f"stage definition/workspace differs: {stage.name}")
        require(result.get("dependencies") == expected_dependencies
                and result.get("inputs") == expected_inputs, f"dependency chain differs: {stage.name}")
        expected_outputs = set(stage.outputs)
        if stage.name in TEST_LOGS:
            expected_outputs.add(TEST_LOGS[stage.name])
        require(set(result.get("outputs", {})) == expected_outputs,
                f"declared output inventory differs: {stage.name}")
        start, finish = timestamp(result["started_utc"]), timestamp(result["finished_utc"])
        elapsed = result.get("elapsed_seconds")
        require(previous_finish <= start <= finish, f"stage chronology differs: {stage.name}")
        require(isinstance(elapsed, (float, int)) and math.isfinite(elapsed) and elapsed >= 0,
                f"invalid stage duration: {stage.name}")
        previous_finish = finish
        evidence.digest(directory / "output.log", result["log_sha256"])
        for name, expected in result["outputs"].items():
            evidence.digest(name, expected)
        receipt_hashes[stage.name] = evidence.digest(directory / "success.json")
        records[stage.name] = result
        public.append({"name": stage.name, "receipt_sha256": receipt_hashes[stage.name],
                       "started_utc": result["started_utc"], "finished_utc": result["finished_utc"],
                       "elapsed_seconds": elapsed, "output_files": len(result["outputs"]),
                       "log_sha256": result["log_sha256"]})
    return public


def validate_environment_receipt(receipt, lock_path, *, wheel_receipt_path, install_log_path):
    expected = parse_lock(lock_path)
    require(receipt.get("passed") is True, "installation receipt did not confirm success")
    require(receipt.get("lock_sha256") == p.digest(lock_path), "installation used another lock")
    required = {"--isolated", "--no-index", "--require-hashes", "--force-reinstall"}
    require(required <= set(receipt.get("install_options", [])), "hash-checked installation options missing")
    require(receipt.get("pip_check", {}).get("returncode") == 0, "pip check did not succeed")
    environment = receipt.get("environment", {})
    require(environment.get("passed") is True and environment.get("isolated_prefix") is True
            and environment.get("include_system_site_packages") is False
            and environment.get("user_site_enabled") is False,
            "installed environment isolation audit missing")
    require(environment.get("wheel_lock_sha256") == p.digest(lock_path), "environment audit lock differs")
    actual = environment.get("distributions", {})
    require(set(actual) - set(expected) <= {"pip"}
            and all(actual.get(name, {}).get("version") == version for name, version in expected.items()),
            "installed distribution versions differ from the full hash lock")
    lock_text = re.sub(r"\\\r?\n[ \t]*", " ", Path(lock_path).read_text(encoding="utf-8"))
    hashes = {}
    for line in lock_text.splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name = re.sub(r"[-_.]+", "-", line.split("==", 1)[0].strip()).lower()
            hashes[name] = set(re.findall(r"--hash=sha256:([0-9a-fA-F]{64})", line))
    # Consume the actual retained installation format, not reconstructed fields
    # inserted into a historical receipt after the installation has completed.
    require(p.digest(wheel_receipt_path) == receipt.get("wheel_receipt_sha256"),
            "installation wheel receipt bytes changed")
    require(p.digest(install_log_path) == receipt.get("install_log_sha256"),
            "installation log bytes changed")
    wheel_receipt = p.read(wheel_receipt_path)
    require(wheel_receipt.get("schema") == 1 and wheel_receipt.get("lock_sha256") == p.digest(lock_path),
            "wheel receipt schema or lock differs")
    wheels = wheel_receipt.get("wheels", [])
    require(len(wheels) == len(expected) and {x.get("name") for x in wheels} == set(expected),
            "installation wheel inventory is incomplete or duplicated")
    require(all(x.get("version") == expected[x["name"]]
                and x.get("sha256") in hashes[x["name"]]
                and isinstance(x.get("bytes"), int) and x["bytes"] > 0
                and isinstance(x.get("filename"), str) and x["filename"].endswith(".whl")
                and "/" not in x["filename"] and "\\" not in x["filename"] for x in wheels),
            "installation wheel bytes do not match the lock")
    raw_log = Path(install_log_path).read_bytes()
    log = raw_log.decode("utf-16" if raw_log.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    installed_lines = [line[len("Successfully installed "):].split() for line in log.splitlines()
                       if line.startswith("Successfully installed ")]
    expected_installs = {f"{name}-{version}" for name, version in expected.items()}
    require(len(installed_lines) == 1 and len(installed_lines[0]) == len(expected)
            and set(installed_lines[0]) == expected_installs,
            "installation log does not show the complete locked reinstall")
    processed = [re.split(r"[\\/]", line[len("Processing "):])[-1].casefold()
                 for line in log.splitlines() if line.startswith("Processing ")]
    require(len(processed) == len(wheels)
            and set(processed) == {row["filename"].casefold() for row in wheels},
            "installation log wheel inputs differ from the wheel receipt")
    return {"passed": True, "locked_distributions": len(expected), "lock_sha256": p.digest(lock_path),
            "wheel_receipt_sha256": receipt["wheel_receipt_sha256"],
            "install_log_sha256": receipt["install_log_sha256"], "complete_locked_reinstall_recorded": True}


def installed_module_bindings(receipt, prefix):
    """Rehash the imported dependency code recorded by the installation audit."""
    modules = receipt.get("environment", {}).get("loaded_modules", {})
    require(set(IMPORT_PROBES) <= set(modules), "installation import-probe evidence is incomplete")
    evidence = Evidence(prefix)
    for name in IMPORT_PROBES:
        require(modules[name].get("path", "").startswith("<venv>/"),
                "installation probe imported code outside the isolated environment")
    for record in modules.values():
        for path_key, hash_key in (("path", "sha256"), ("provider_path", "provider_sha256")):
            location = record.get(path_key, "")
            if location.startswith("<venv>/"):
                expected = record.get(hash_key)
                require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                        "installation module hash is absent or malformed")
                evidence.digest(location[len("<venv>/"):], expected)
    evidence.finish()
    return {"verified_imported_dependency_files": len(evidence.hashes),
            "scope": "Files and dynamic providers imported during installation verification; not every installed package file"}


def live_environment(workspace, lock_path, receipt):
    require(not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME"),
            "clear Python path overrides before acceptance")
    # The verifier can be a later source-only addition in the publishing checkout.
    verifier_root = Path(p.__file__).resolve().parents[2]
    current = validate_venv(sys.prefix, sys.base_prefix, sys.executable, verifier_root, sys.path,
                            user_site_enabled=site.ENABLE_USER_SITE)
    require(sys.version_info[:2] == (3, 10) and sys.platform == "win32",
            "raw wheel lock requires Windows CPython 3.10")
    require(current["pyvenv_config_sha256"] == receipt["environment"]["pyvenv_config_sha256"],
            "virtual environment configuration changed since installation")
    expected, actual = parse_lock(lock_path), {}
    for dist in importlib.metadata.distributions():
        require(Path(dist.locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve()),
                "distribution outside the isolated environment")
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        require(name not in actual, "duplicate installed distribution")
        actual[name] = dist.version
    require(set(actual) - set(expected) <= {"pip"}
            and all(actual.get(name) == version for name, version in expected.items()),
            "live environment differs from the full hash lock")
    return {**current, "python_version": sys.version.split()[0], "locked_distributions": len(expected),
            "installed_code": installed_module_bindings(receipt, sys.prefix)}


def raw_provenance(evidence, manifest):
    require(manifest.get("raw_mode") == "provided" and set(manifest.get("raw_files", {})) == RAW_NAMES,
            "acceptance requires all four explicitly supplied original raw TSVs")
    result = {}
    for name, digest in manifest["raw_files"].items():
        path = "data/raw/" + name
        evidence.digest(path, digest)
        result[name] = {"sha256": digest, "bytes": evidence.path(path).stat().st_size}
    provenance = evidence.read("data/processed/provenance.json")
    rows = provenance.get("downloads", [])
    require(len(rows) == 2 and {x.get("source_split") for x in rows} == {"train", "dev"},
            "processed provenance does not identify both supplied raw splits")
    for row in rows:
        require(row.get("locally_provided") is True and set(row.get("files", {})) == {"news.tsv", "behaviors.tsv"},
                "processed provenance does not bind locally supplied files")
        for name, value in row["files"].items():
            require(value == result[row["source_split"] + "/" + name],
                    "processed provenance differs from imported raw bytes")
    require(evidence.read("reports/data_validation.json").get("status") == "passed", "data audit did not pass")
    return result


def release_bindings(evidence, version):
    name = f"reports/{version}/release_integrity.json"
    report = evidence.read(name)
    require(report.get("passed") is True, f"{version} release auditor did not pass")
    bindings = report.get("verified_sha256_files", {})
    require(bindings and report.get("checks"), f"{version} release audit has no detailed evidence")
    checks = {
        "v2": {"frozen_inputs", "mature_internal_ranking", "mature_selection", "internal_selection",
               "catalog", "dev_ranking_sample", "dev_ranking_full"},
        "v3": {"sources_and_inputs", "fit_pairs_and_normalization", "tune", "confirm", "dev",
               "offline_quality_cache_to_service", "publication_evidence"},
    }
    require(version in checks and set(report["checks"]) == checks[version],
            f"{version} release audit check inventory differs")
    required_bindings = {"data/processed/manifest.json", "data/processed/news_features.npy",
                         "data/temporal_v2/manifest.json", "artifacts/v2/selection.json",
                         "artifacts/v2/dev_catalog_counts.npz"}
    if version == "v2":
        required_bindings |= {"reports/ranking_experiment.json", "reports/v2/select.json",
                              "reports/v2/catalog.json", "reports/v2/ranking.json", "reports/v2/ranking-full.json",
                              "reports/v2/backbone_matured_selection.json", "reports/v2/ranker_matured_selection.json",
                              "reports/v2/query_cache_equivalence.json", "artifacts/retriever.pt",
                              "artifacts/article_embeddings.npy", "artifacts/v2/user_encoder.npz",
                              "artifacts/v2/user_encoder.npz.json"}
        required_bindings |= {f"artifacts/v2/ranker/{variant}-seed{seed}.joblib{suffix}"
                              for variant in ("content_only", "context_only", "full")
                              for seed in p.SEEDS for suffix in ("", ".manifest.json")}
        require("auditor_source_sha256" in report, "v2 auditor source binding is absent")
    else:
        required_bindings |= {"newsforge/v3/audit.py", "docs/V3_PROTOCOL.md", "artifacts/v3/selection.json",
                              "reports/v2/release_integrity.json", "reports/v3_release_unit_tests.txt",
                              "reports/v3/service_benchmark.json", "reports/v3/fast_retrieval_benchmark.json",
                              "reports/v3/v3_results.metadata.json", "reports/v3/v3_results.png",
                              "reports/v3/v3_results.svg"}
        required_bindings |= {f"reports/v3/{phase}.json" for phase in ("fit", "tune", "confirm", "dev")}
        required_bindings |= {f"artifacts/v3/{phase}/cache.json" for phase in ("tune", "confirm", "dev")}
        required_bindings |= {f"artifacts/v3/models/{variant}-seed{seed}.npz{suffix}"
                              for variant in ("full13", "no_category", "no_recent")
                              for seed in p.SEEDS for suffix in ("", ".json")}
        required_bindings |= {path.relative_to(evidence.root).as_posix()
                              for path in evidence.path("tests").glob("test_*.py")}
    require(required_bindings <= set(bindings), f"{version} release audit is missing mandatory artifact bindings")
    for path, expected in bindings.items():
        evidence.digest(path, expected)
    if "auditor_source_sha256" in report:
        evidence.digest(f"newsforge/{version}/audit.py", report["auditor_source_sha256"])
    return {"passed": True, "report_sha256": evidence.digest(name), "verified_files": len(bindings),
            "checks": report["checks"]}


def scientific_summary(report):
    return {key: report[key] for key in ("phase", "all_clicks", "eligible_clicks", "seen_clicks",
                                         "shortfall_requests", "delta", "metrics", "deployment_gate_passed")
            if key in report} | {"rows": report.get("bindings", {}).get("rows")}


def comparisons(evidence):
    result = {}
    for phase in ("confirm", "dev"):
        fresh_path, old_path = f"reports/v3/{phase}.json", f"reports-delivered/v3/{phase}.json"
        fresh, old = scientific_summary(evidence.read(fresh_path)), scientific_summary(evidence.read(old_path))
        result[phase] = {"fresh": fresh, "previous": old, "scientific_summary_exactly_equal": fresh == old,
                         "fresh_report_sha256": evidence.digest(fresh_path),
                         "reference_report_sha256": evidence.digest(old_path)}
    return {"scope": "Descriptive comparison with copied prior reports; equality is not an acceptance criterion and does not select any model.",
            "phases": result}


def audit(workspace, environment_receipt, *, wheel_receipt=None, install_log=None):
    evidence = Evidence(workspace)
    manifest = evidence.read(p.STATE / "workspace.json")
    require(manifest.get("schema_version") == p.SCHEMA, "unsupported workspace schema")
    require(manifest.get("plan_sha256") == p.canonical_hash(p.make_plan()), "pipeline plan differs")
    require(set(manifest["source_files"]) == set(p.source_files(evidence.root)),
            "source inventory differs from initialization")
    for name, expected in manifest["source_files"].items():
        require(not name.startswith(("data/", "artifacts/", "reports/")), "initialization imported generated data/models")
        evidence.digest(name, expected)
    # A later independent verifier need not be part of the initialized source,
    # but it must interpret the exact orchestrator and HTTP/environment helpers
    # that were frozen for the executed run.
    for module, name in ((p, "newsforge/delivery/pipeline.py"),
                         (sys.modules[validate_http_report.__module__], "newsforge/delivery/acceptance.py")):
        require(p.digest(module.__file__) == manifest["source_files"][name], "executed verifier source differs")
    environment_receipt = Path(environment_receipt).resolve()
    wheel_receipt = environment_receipt.with_name("wheels.json") if wheel_receipt is None else Path(wheel_receipt)
    install_log = (environment_receipt.parents[2] / "artifacts/raw_rebuild/install.log"
                   if install_log is None else Path(install_log))
    installation_paths = {"receipt": environment_receipt, "wheels": wheel_receipt, "log": install_log}
    installation_hashes = {key: p.digest(path) for key, path in installation_paths.items()}
    receipt = p.read(environment_receipt)
    environment = validate_environment_receipt(receipt, evidence.path(LOCK),
                                                wheel_receipt_path=wheel_receipt, install_log_path=install_log)
    environment["live"] = live_environment(evidence.root, evidence.path(LOCK), receipt)
    environment["installation_receipt_sha256"] = installation_hashes["receipt"]
    raw = raw_provenance(evidence, manifest)
    chain = receipt_chain(evidence, manifest)
    releases = {version: release_bindings(evidence, version) for version in ("v2", "v3")}
    http = validate_http_report(evidence.read("reports/v3/service_benchmark.json"),
                                evidence.path("artifacts/v3/service_benchmark"))
    comparison = comparisons(evidence)
    evidence.finish()
    require(all(p.digest(path) == installation_hashes[key] for key, path in installation_paths.items()),
            "installation evidence changed during acceptance")
    return {"schema_version": 1, "passed": True, "created_utc": p.utc(),
            "scope": "Completed raw TSV preprocessing, all frozen training/selection/evaluation dependencies, and local HTTP service acceptance",
            "workspace_manifest_sha256": evidence.digest(p.STATE / "workspace.json"),
            "verifier_source_sha256": p.digest(__file__),
            "source_files": len(manifest["source_files"]), "plan_sha256": manifest["plan_sha256"],
            "environment": environment, "raw_files": raw, "completed_stages": chain,
            "release_audits": releases, "http": http, "comparison": comparison,
            "verified_sha256_files": dict(sorted(evidence.hashes.items())),
            "limits": ["Local retained process receipts are evidence, not remote attestation or an operating-system sandbox.",
                       "No old processed files or models are imported by the verified initializer; stage preconditions reject existing training outputs. Identical output bytes can be valid deterministic retraining.",
                       "This verifier rehashes the independent auditors' complete file bindings; it does not repeat their model predictions or numerical calculations.",
                       "The observed MIND dev set remains observed. No unseen external validation, production deployment, or online A/B test is established by this receipt.",
                       "Recorded HTTP timings describe this run and host; equality with historical performance is not required."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--wheel-receipt", type=Path, help="Default: wheels.json next to installation receipt")
    parser.add_argument("--install-log", type=Path,
                        help="Default: artifacts/raw_rebuild/install.log in the installation evidence checkout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain previous acceptance evidence; choose a new output path")
    result = audit(args.workspace, args.environment_receipt,
                   wheel_receipt=args.wheel_receipt, install_log=args.install_log)
    p.write_new(args.output, result)
    print(json.dumps({"passed": True, "stages": len(result["completed_stages"]),
                      "verified_files": len(result["verified_sha256_files"])}))


if __name__ == "__main__":
    main()
