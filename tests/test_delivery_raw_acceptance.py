"""Fail-closed checks for retained raw-rebuild evidence; no ML workload."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from newsforge.delivery import pipeline as p
from newsforge.delivery import raw_acceptance as a


def put(root, name, value):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class ReceiptChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.python = str((self.root / "python.exe").resolve())
        self.definitions = (p.Stage("a", (), ("-m", "first"), ("reports/a.json",)),
                            p.Stage("b", ("a",), ("-m", "second"), ("reports/b.json",)))
        self.manifest = {"created_utc": "2026-01-01T00:00:00+00:00", "raw_mode": "provided"}
        put(self.root, p.STATE / "workspace.json", self.manifest)
        self.records = {}
        for index, stage in enumerate(self.definitions):
            result_file = put(self.root, stage.outputs[0], {"stage": stage.name})
            directory = self.root / p.STATE / stage.name
            directory.mkdir(parents=True)
            (directory / "output.log").write_bytes(b"stage output\n")
            dependencies = {name: p.digest(self.root / p.STATE / name / "success.json") for name in stage.dependencies}
            inputs = {path: digest for name in stage.dependencies for path, digest in self.records[name]["outputs"].items()}
            started = {"stage": stage.name, "status": "running", "python": self.python,
                       "argv": [self.python, *stage.argv],
                       "started_utc": f"2026-01-01T00:00:0{index * 2 + 1}+00:00",
                       "stage_definition_sha256": p.canonical_hash(asdict(stage)),
                       "workspace_manifest_sha256": p.digest(self.root / p.STATE / "workspace.json"),
                       "dependencies": dependencies, "inputs": inputs}
            put(self.root, p.STATE / stage.name / "started.json", started)
            self.records[stage.name] = {**started, "status": "success", "returncode": 0,
                                      "finished_utc": f"2026-01-01T00:00:0{index * 2 + 2}+00:00",
                                      "elapsed_seconds": 1.0,
                                      "outputs": {stage.outputs[0]: p.digest(result_file)},
                                      "log_sha256": p.digest(directory / "output.log")}
            put(self.root, p.STATE / stage.name / "success.json", self.records[stage.name])

    def check(self):
        return a.receipt_chain(a.Evidence(self.root), self.manifest,
                               definitions=self.definitions, target="b", python=self.python)

    def update_b(self, **values):
        started_path = self.root / p.STATE / "b/started.json"
        started, result = p.read(started_path), deepcopy(self.records["b"])
        result.update(values)
        for key, value in values.items():
            if key in started and key != "status":
                started[key] = value
        put(self.root, p.STATE / "b/started.json", started)
        put(self.root, p.STATE / "b/success.json", result)

    def test_complete_chain_produces_portable_summary(self):
        result = self.check()
        self.assertEqual([row["name"] for row in result], ["a", "b"])
        self.assertNotIn(self.python, json.dumps(result))

    def test_final_success_does_not_replace_missing_predecessor(self):
        (self.root / p.STATE / "a/success.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_retained_failure_rejects_even_success_file(self):
        put(self.root, p.STATE / "b/failed.json", {"status": "failed"})
        with self.assertRaisesRegex(ValueError, "failed stage"):
            self.check()

    def test_changed_model_or_log_bytes_rejected(self):
        for path in ("reports/a.json", str(p.STATE / "a/output.log")):
            with self.subTest(path=path):
                target = self.root / path
                old = target.read_bytes()
                target.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    self.check()
                target.write_bytes(old)

    def test_missing_output_binding_rejected(self):
        self.update_b(outputs={})
        with self.assertRaisesRegex(ValueError, "output inventory"):
            self.check()

    def test_commands_interpreter_inputs_and_dependencies_bound(self):
        for values, message in (({"argv": [self.python, "-c", "pass"]}, "command/interpreter"),
                                ({"python": "other.exe"}, "command/interpreter"),
                                ({"inputs": {}}, "dependency chain"),
                                ({"dependencies": {}}, "dependency chain"),
                                ({"stage_definition_sha256": "0" * 64}, "stage definition")):
            with self.subTest(values=values):
                original = (self.root / p.STATE / "b/started.json").read_bytes()
                self.update_b(**values)
                with self.assertRaisesRegex(ValueError, message):
                    self.check()
                (self.root / p.STATE / "b/started.json").write_bytes(original)
                put(self.root, p.STATE / "b/success.json", self.records["b"])

    def test_start_success_mismatch_rejected(self):
        result = deepcopy(self.records["b"])
        result["started_utc"] = "2026-01-01T00:00:00+00:00"
        put(self.root, p.STATE / "b/success.json", result)
        with self.assertRaisesRegex(ValueError, "start/success"):
            self.check()

    def test_reversed_chronology_and_nonfinite_duration_rejected(self):
        self.update_b(started_utc="2026-01-01T00:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "chronology"):
            self.check()
        self.update_b(started_utc=self.records["b"]["started_utc"], elapsed_seconds=float("nan"))
        with self.assertRaisesRegex(ValueError, "duration"):
            self.check()


class EnvironmentReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lock = Path(self.temp.name) / "requirements-raw-windows.lock"
        self.lock.write_text("--only-binary=:all:\nexample==1.2 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
        self.wheel_receipt = self.lock.with_name("wheels.json")
        self.install_log = self.lock.with_name("install.log")
        self.wheels = [{"name": "example", "version": "1.2", "sha256": "a" * 64,
                        "bytes": 10, "filename": "example-1.2-py3-none-any.whl"}]
        self.install_log.write_text("Processing C:\\private\\example-1.2-py3-none-any.whl\nSuccessfully installed example-1.2\n")
        self.receipt = {"passed": True,
                        "lock_sha256": p.digest(self.lock),
                        "install_options": ["--isolated", "--no-index", "--require-hashes", "--force-reinstall"],
                        "pip_check": {"returncode": 0},
                        "environment": {"passed": True, "isolated_prefix": True,
                                        "include_system_site_packages": False, "user_site_enabled": False,
                                        "wheel_lock_sha256": p.digest(self.lock),
                                        "distributions": {"example": {"version": "1.2"}, "pip": {"version": "1"}}},
                        "install_log_sha256": p.digest(self.install_log)}
        self.update_wheels(self.wheels)

    def update_wheels(self, wheels):
        put(self.wheel_receipt.parent, self.wheel_receipt.name,
            {"schema": 1, "lock_sha256": p.digest(self.lock), "wheels": wheels})
        self.receipt["wheel_receipt_sha256"] = p.digest(self.wheel_receipt)

    def check(self, receipt=None):
        return a.validate_environment_receipt(self.receipt if receipt is None else receipt, self.lock,
                                               wheel_receipt_path=self.wheel_receipt, install_log_path=self.install_log)

    def test_hash_locked_installation_closure(self):
        result = self.check()
        self.assertEqual(result["locked_distributions"], 1)
        self.assertTrue(result["complete_locked_reinstall_recorded"])
        self.assertNotIn("private", json.dumps(result))

    def test_wrong_wheel_bytes_and_missing_installation_rejected(self):
        for wheels in ([], [{**self.wheels[0], "sha256": "b" * 64}], self.wheels * 2):
            with self.subTest(wheels=wheels):
                self.update_wheels(wheels)
                with self.assertRaisesRegex(ValueError, "wheel"):
                    self.check()

    def test_installation_originals_cannot_be_replaced(self):
        for path in (self.wheel_receipt, self.install_log):
            original = path.read_bytes()
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                self.check()
            path.write_bytes(original)

    def test_log_must_record_all_real_wheel_inputs_and_reinstalled_versions(self):
        for text in ("Successfully installed example-1.2\n",
                     "Processing C:\\private\\other.whl\nSuccessfully installed example-1.2\n",
                     "Processing C:\\private\\example-1.2-py3-none-any.whl\nRequirement already satisfied: example\n",
                     "Processing C:\\private\\example-1.2-py3-none-any.whl\nSuccessfully installed example-2.0\n"):
            self.install_log.write_text(text)
            self.receipt["install_log_sha256"] = p.digest(self.install_log)
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "installation log"):
                self.check()

    def test_inherited_extra_wrong_versions_and_failed_pip_rejected(self):
        cases = []
        value = deepcopy(self.receipt)
        value["environment"]["include_system_site_packages"] = True
        cases.append(value)
        value = deepcopy(self.receipt)
        value["environment"]["distributions"]["unlocked"] = {"version": "1"}
        cases.append(value)
        value = deepcopy(self.receipt)
        value["environment"]["distributions"]["example"]["version"] = "2"
        cases.append(value)
        value = deepcopy(self.receipt)
        value["pip_check"]["returncode"] = 1
        cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(value)

    def test_unchanged_versions_do_not_hide_mutated_installed_code(self):
        prefix = self.lock.parent / "venv"
        modules = {}
        for name in a.IMPORT_PROBES:
            relative = f"Lib/site-packages/{name}/__init__.py"
            path = prefix / relative
            path.parent.mkdir(parents=True)
            path.write_text("# imported fixture " + name)
            modules[name] = {"path": "<venv>/" + relative, "sha256": p.digest(path)}
        receipt = {"environment": {"loaded_modules": modules}}
        self.assertEqual(a.installed_module_bindings(receipt, prefix)["verified_imported_dependency_files"], len(modules))
        (prefix / "Lib/site-packages/numpy/__init__.py").write_text("# modified while version metadata is unchanged")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            a.installed_module_bindings(receipt, prefix)


class SourceEvidenceTests(unittest.TestCase):
    @staticmethod
    def v3_release_fixture(root):
        files = {"data/processed/manifest.json", "data/processed/news_features.npy",
                 "data/temporal_v2/manifest.json", "artifacts/v2/selection.json",
                 "artifacts/v2/dev_catalog_counts.npz", "newsforge/v3/audit.py", "docs/V3_PROTOCOL.md",
                 "artifacts/v3/selection.json", "reports/v2/release_integrity.json",
                 "reports/v3_release_unit_tests.txt", "reports/v3/service_benchmark.json",
                 "reports/v3/fast_retrieval_benchmark.json", "reports/v3/v3_results.metadata.json",
                 "reports/v3/v3_results.png", "reports/v3/v3_results.svg", "tests/test_v3_example.py"}
        files |= {f"reports/v3/{phase}.json" for phase in ("fit", "tune", "confirm", "dev")}
        files |= {f"artifacts/v3/{phase}/cache.json" for phase in ("tune", "confirm", "dev")}
        files |= {f"artifacts/v3/models/{variant}-seed{seed}.npz{suffix}"
                  for variant in ("full13", "no_category", "no_recent")
                  for seed in (17, 29, 43) for suffix in ("", ".json")}
        report = {"passed": True,
                  "checks": {name: {"fixture": True} for name in ("sources_and_inputs", "fit_pairs_and_normalization",
                             "tune", "confirm", "dev", "offline_quality_cache_to_service", "publication_evidence")},
                  "verified_sha256_files": {name: p.digest(put(root, name, {"fixture": name})) for name in files}}
        put(root, "reports/v3/release_integrity.json", report)
        return report

    def test_raw_processed_provenance_must_match_all_four_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes, splits = {}, {}
            for name in a.RAW_NAMES:
                path = root / "data/raw" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                hashes[name] = p.digest(path)
                split, filename = name.split("/")
                splits.setdefault(split, {})[filename] = {"sha256": hashes[name], "bytes": path.stat().st_size}
            provenance = {"downloads": [{"source_split": split, "locally_provided": True, "files": files}
                                        for split, files in splits.items()]}
            put(root, "data/processed/provenance.json", provenance)
            put(root, "reports/data_validation.json", {"status": "passed"})
            manifest = {"raw_mode": "provided", "raw_files": hashes}
            self.assertEqual(set(a.raw_provenance(a.Evidence(root), manifest)), a.RAW_NAMES)
            provenance["downloads"][0]["files"]["news.tsv"]["sha256"] = "0" * 64
            put(root, "data/processed/provenance.json", provenance)
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                a.raw_provenance(a.Evidence(root), manifest)

    def test_report_passed_flag_requires_current_bound_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.v3_release_fixture(root)
            model = root / "artifacts/v3/models/full13-seed17.npz"
            self.assertTrue(a.release_bindings(a.Evidence(root), "v3")["passed"])
            model.write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                a.release_bindings(a.Evidence(root), "v3")

    def test_release_audit_cannot_omit_model_or_check_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.v3_release_fixture(root)
            for key, name, message in (("verified_sha256_files", "artifacts/v3/models/no_recent-seed43.npz", "mandatory"),
                                       ("checks", "fit_pairs_and_normalization", "check inventory")):
                value = deepcopy(original)
                del value[key][name]
                put(root, "reports/v3/release_integrity.json", value)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                    a.release_bindings(a.Evidence(root), "v3")

    def test_paths_cannot_escape_or_read_delivered_data_as_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = a.Evidence(directory)
            for name in ("../model.pt", "C:/private/model.pt", "artifacts/../../model.pt"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    evidence.digest(name)
            with self.assertRaises(FileNotFoundError):
                evidence.read("reports/v3/release_integrity.json")

    def test_comparison_records_a_real_regression_without_reselecting(self):
        with tempfile.TemporaryDirectory() as directory:
            for phase in ("confirm", "dev"):
                put(directory, f"reports/v3/{phase}.json", {"phase": phase, "metrics": {"full13": {"hits10": 2}}})
                put(directory, f"reports-delivered/v3/{phase}.json", {"phase": phase, "metrics": {"full13": {"hits10": 3}}})
            result = a.comparisons(a.Evidence(directory))
            self.assertFalse(result["phases"]["dev"]["scientific_summary_exactly_equal"])
            self.assertEqual(result["phases"]["dev"]["fresh"]["metrics"]["full13"]["hits10"], 2)


if __name__ == "__main__":
    unittest.main()
