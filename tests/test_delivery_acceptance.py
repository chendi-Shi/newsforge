"""Small fixtures for clean-environment and end-to-end delivery evidence checks."""

import copy
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from newsforge.delivery.acceptance import (
    CONTRACT_NAMES, HTTP_CONFIGURATIONS, audit_source, check_git_status,
    module_record, parse_lock, parse_test_summary, public_path, relative_output, run, run_logged, sha256,
    validate_http_report, validate_venv, write_new,
)
from tests.support import workspace_temporary_directory


def environment_fixture(directory, inherit="false"):
    root = Path(directory)
    prefix, base = root / "clone/.venv", root / "base"
    prefix.mkdir(parents=True)
    base.mkdir()
    (prefix / "pyvenv.cfg").write_text("include-system-site-packages = " + inherit + "\n", encoding="utf-8")
    executable = prefix / "Scripts/python.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"fixture interpreter")
    return prefix, base, executable, root / "clone"


def http_fixture(directory):
    root = Path(directory)
    records, configurations = {}, []
    for batch, concurrency in sorted(HTTP_CONFIGURATIONS):
        record = {"seconds": .01, "http_error": 0, "query_errors": 0,
                  "mismatches": 0, "status": 200}
        records[f"batch{batch}_concurrency{concurrency}"] = {
            "warmup": [record] * 8, "measured": [record] * (192 // batch)}
        configurations.append({"batch_size": batch, "concurrency": concurrency,
                               "rounds": 3, "measured_queries": 192,
                               "measured_http_requests": 192 // batch,
                               "http_errors": 0, "failed_queries": 0, "offline_http_mismatches": 0})
    write_new(root / "requests.json", {"requests": [{}] * 64})
    write_new(root / "offline.json", {"responses": [{}] * 64})
    write_new(root / "http_records.json", records)
    return {"passed": True, "benchmark_version": 3, "sampling": {"queries": 64},
            "health": {"api_version": "v3"},
            "offline_causal_output_check": {"passed": True, "queries": 64},
            "api_contract_checks": [{"name": name, "passed": True} for name in CONTRACT_NAMES],
            "resources": {"sampling_error": None, "resource_samples": 8},
            "private_evidence": {name + "_sha256": sha256(root / (name + ".json"))
                                 for name in ("requests", "offline", "http_records")},
            "configurations": configurations}


class DeliveryAcceptanceTests(unittest.TestCase):
    def test_isolated_venv_paths_are_portable(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            prefix, base, executable, root = environment_fixture(directory)
            result = validate_venv(prefix, base, executable, root,
                                   [str(root), str(base / "Lib"), str(prefix / "Lib/site-packages")],
                                   user_site_enabled=False)
            self.assertFalse(result["include_system_site_packages"])
            self.assertEqual(result["search_path"], ["<clone>/.", "<base>/Lib", "<venv>/Lib/site-packages"])
            self.assertNotIn(str(Path(directory).resolve()), json.dumps(result))

    def test_inherited_site_packages_are_rejected(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            prefix, base, executable, root = environment_fixture(directory, "true")
            with self.assertRaisesRegex(ValueError, "explicitly false"):
                validate_venv(prefix, base, executable, root, [], user_site_enabled=False)

    def test_site_package_escape_is_rejected_even_inside_clone(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            prefix, base, executable, root = environment_fixture(directory)
            for escaped in (base / "Lib/site-packages", root / "old_env/Lib/site-packages"):
                with self.subTest(escaped=escaped), self.assertRaisesRegex(ValueError, "third-party path"):
                    public_path(escaped, prefix=prefix, base=base, root=root)

    def test_base_interpreter_and_enabled_user_site_are_rejected(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            prefix, base, executable, root = environment_fixture(directory)
            with self.assertRaisesRegex(ValueError, "independent virtual"):
                validate_venv(base, base, executable, root, [], user_site_enabled=False)
            with self.assertRaisesRegex(ValueError, "user site"):
                validate_venv(prefix, base, executable, root, [], user_site_enabled=True)

    def test_dynamic_torch_modules_require_the_actual_isolated_provider_class(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            prefix, base, executable, root = environment_fixture(directory)
            provider = types.ModuleType("torch._ops")
            provider.__file__ = str(prefix / "Lib/site-packages/torch/_ops.py")
            path = Path(provider.__file__)
            path.parent.mkdir(parents=True)
            path.write_text("# provider fixture\n", encoding="utf-8")
            provider._Ops = type("_Ops", (types.ModuleType,), {"__module__": "torch._ops"})
            dynamic = provider._Ops("torch.ops")
            dynamic.__file__ = "_ops.py"
            with patch.dict(sys.modules, {"torch._ops": provider}):
                result = module_record("torch.ops", dynamic, prefix=prefix, base=base, root=root)
                self.assertEqual(result["dynamic_module_type"], "torch._ops._Ops")
                self.assertEqual(result["provider_sha256"], sha256(path))
                with self.assertRaisesRegex(ValueError, "non-existent"):
                    module_record("arbitrary.module", dynamic, prefix=prefix, base=base, root=root)
                provider.__file__ = str(base / "_ops.py")
                Path(provider.__file__).write_text("# wrong location\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "outside"):
                    module_record("torch.ops", dynamic, prefix=prefix, base=base, root=root)

    def test_git_status_allows_only_named_untracked_reports(self):
        allowed = ["reports/delivery/acceptance.json"]
        check_git_status("?? reports/delivery/acceptance.json\0", allowed)
        for status in (" M newsforge/v3/serve.py\0", "?? secret.txt\0",
                       "A  reports/delivery/acceptance.json\0", "?? reports/delivery/other.json\0"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                check_git_status(status, allowed)

    def test_source_commit_and_actual_tracked_bytes_are_bound(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            root = Path(directory).resolve()
            (root / "source.py").write_text("value = 1\n", encoding="utf-8")
            outputs = ["a" * 40 + "\n", str(root) + "\n", "", "source.py\0"]
            with patch("newsforge.delivery.acceptance.git_output", side_effect=outputs):
                report = audit_source("git", root, "a" * 40, [])
            self.assertEqual(report["tracked_files"], {"source.py": sha256(root / "source.py")})
            with patch("newsforge.delivery.acceptance.git_output", return_value="b" * 40 + "\n"):
                with self.assertRaisesRegex(ValueError, "differs"):
                    audit_source("git", root, "a" * 40, [])

    def test_lock_requires_hashes_and_unique_pins(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            path = Path(directory) / "requirements.lock"
            path.write_text("numpy==1.26.4 \\\n    --hash=sha256:" + "a" * 64 + "\nTorch==2.14.0+cpu --hash=sha256:" + "b" * 64 + "\n", encoding="utf-8")
            self.assertEqual(parse_lock(path), {"numpy": "1.26.4", "torch": "2.14.0+cpu"})
            path.write_text("numpy==1.26.4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wheel-hash"):
                parse_lock(path)
            path.write_text(("numpy==1.26.4 --hash=sha256:" + "a" * 64 + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                parse_lock(path)

    def test_each_requirement_needs_a_valid_hash(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            path = Path(directory) / "requirements.lock"
            good = "numpy==1.26.4 --hash=sha256:" + "a" * 64 + "\n"
            for invalid in ("torch==2.14.0+cpu\n", "torch==2.14.0+cpu --hash=sha256:bad\n",
                            "torch>=2 --hash=sha256:" + "b" * 64 + "\n",
                            "-r other.lock\n"):
                with self.subTest(invalid=invalid):
                    path.write_text(good + invalid, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "every delivery"):
                        parse_lock(path)

    def test_test_summary_distinguishes_skips_from_passing_cases(self):
        self.assertEqual(parse_test_summary("Ran 222 tests in 43.662s\n\nOK (skipped=1)\n"),
                         {"count": 222, "skipped": 1, "executed_count": 221})
        self.assertEqual(parse_test_summary("Ran 12 tests in 1.2s\n\nOK\n"),
                         {"count": 12, "skipped": 0, "executed_count": 12})
        for text in ("Ran 0 tests in 0s\nOK\n", "Ran 2 tests in 1s\nOK (skipped=2)\n",
                     "Ran 2 tests in 1s\nFAILED (errors=1)\n",
                     "Ran 2 tests in 1s\nOK (expected failures=1)\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_test_summary(text)

    def test_full_http_evidence_passes_but_boolean_alone_does_not(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            report = http_fixture(directory)
            accepted = validate_http_report(report, directory)
            self.assertEqual(accepted["measured_queries"], 768)
            incomplete = copy.deepcopy(report)
            incomplete["api_contract_checks"].pop()
            with self.assertRaisesRegex(ValueError, "complete HTTP"):
                validate_http_report(incomplete, directory)
            incomplete = copy.deepcopy(report)
            incomplete["configurations"][0]["offline_http_mismatches"] = 1
            with self.assertRaisesRegex(ValueError, "parity matrix"):
                validate_http_report(incomplete, directory)

    def test_http_raw_tampering_and_short_workload_are_rejected(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            report = http_fixture(directory)
            path = Path(directory) / "http_records.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["batch1_concurrency1"]["measured"].pop()
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_http_report(report, directory)
            report["private_evidence"]["http_records_sha256"] = sha256(path)
            with self.assertRaisesRegex(ValueError, "cardinality"):
                validate_http_report(report, directory)

    def test_outputs_cannot_escape_and_existing_evidence_is_preserved(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            root = Path(directory)
            target = root / "reports/delivery/acceptance.json"
            self.assertEqual(relative_output(target, root, "reports/delivery"), "reports/delivery/acceptance.json")
            with self.assertRaises(ValueError):
                relative_output(root / "elsewhere.json", root, "reports/delivery")
            write_new(target, {"passed": True})
            with self.assertRaises(FileExistsError):
                write_new(target, {"passed": False})
            self.assertTrue(json.loads(target.read_text(encoding="utf-8"))["passed"])

    def test_timeout_cleans_only_the_owned_process_object(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            process, owned = MagicMock(), MagicMock()
            owned.wait.side_effect = subprocess.TimeoutExpired("fixture", 1)
            with patch("newsforge.delivery.acceptance.subprocess.Popen", return_value=process), \
                    patch("newsforge.v2.benchmark.OwnedProcessTree", return_value=owned) as wrapper, \
                    patch("newsforge.v2.benchmark.stop_owned_process") as stop:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_logged(["fixture"], Path(directory) / "child.log", timeout=1)
            wrapper.assert_called_once_with(process)
            stop.assert_called_once_with(owned)

    def test_skipped_tests_can_only_produce_partial_acceptance(self):
        with workspace_temporary_directory("acceptance-test-") as directory:
            root = Path(directory).resolve()
            report = root / "reports/delivery/acceptance.json"
            lock = root / "requirements.lock"
            lock.write_text("fixture", encoding="utf-8")
            source = {"commit": "a" * 40, "tracked_files": {lock.name: sha256(lock)}, "source_clean": True}
            args = Namespace(report=report, git="git", lock=lock.name, expected_commit="a" * 40,
                             skip_tests=True, test_timeout=60, http_timeout=60)

            def subprocess_fixture(command, path, *, timeout):
                if "newsforge.v3.benchmark_service" in command:
                    write_new(report.with_name("acceptance_http.json"), {})
                return {"exit_code": 0}

            with patch("newsforge.delivery.acceptance.Path.cwd", return_value=root), \
                    patch("newsforge.delivery.acceptance.audit_source", return_value=source), \
                    patch("newsforge.delivery.acceptance.audit_environment", return_value={"passed": True}), \
                    patch("newsforge.delivery.acceptance.run_logged", side_effect=subprocess_fixture) as runner, \
                    patch("newsforge.delivery.acceptance.validate_http_report", return_value={"passed": True}):
                receipt = run(args)
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["status"], "partial")
            self.assertFalse(receipt["unit_tests"]["executed"])
            self.assertNotIn("unittest", str(runner.call_args_list))


if __name__ == "__main__":
    unittest.main()
