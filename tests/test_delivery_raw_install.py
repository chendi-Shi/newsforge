"""Synthetic wheel/mock-pip checks; never installs into the running interpreter."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from newsforge.delivery import pipeline as p
from newsforge.delivery import raw_install as installer
from newsforge.delivery.raw_acceptance import validate_environment_receipt
from newsforge.delivery.wheels import render_lock, wheel_records


class RawInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Windows runners may return an 8.3 alias for the same temp directory.
        # Match the installer's canonical paths while retaining exact assertions.
        self.root = Path(self.temp.name).resolve()
        self.wheelhouse = self.root / "wheelhouse"
        self.wheelhouse.mkdir()
        self.wheel = self.wheelhouse / "example-1.2-py3-none-any.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("example-1.2.dist-info/METADATA", "Name: example\nVersion: 1.2\n")
        self.lock = self.root / "requirements.lock"
        self.wheel_receipt = self.root / "wheels.json"
        self.record_inputs()
        self.output, self.log = self.root / "reports/install.json", self.root / "private/install.log"
        self.kwargs = {"root": self.root, "wheelhouse": self.wheelhouse, "lock": self.lock,
                       "wheel_receipt": self.wheel_receipt, "output": self.output, "log": self.log}
        self.environment = {"passed": True, "isolated_prefix": True, "include_system_site_packages": False,
                            "user_site_enabled": False, "wheel_lock_sha256": p.digest(self.lock),
                            "locked_distributions": 1,
                            "distributions": {"example": {"version": "1.2"}, "pip": {"version": "1"}}}
        self.events = []

    def record_inputs(self):
        records = wheel_records(self.wheelhouse)
        self.lock.write_text(render_lock(records), encoding="utf-8")
        self.wheel_receipt.write_text(json.dumps({"schema": 1, "lock_sha256": p.digest(self.lock),
                                                  "wheels": records}), encoding="utf-8")

    def fake_process(self, command, **kwargs):
        self.events.append(command)
        self.assertEqual(command[0], sys.executable)
        self.assertFalse(kwargs["check"])
        self.assertEqual(kwargs["cwd"], str(self.root))
        if "install" in command:
            self.assertEqual(command[:5], [sys.executable, "-m", "pip", "--isolated", "install"])
            for argument in installer.INSTALL_OPTIONS:
                self.assertIn(argument, command)
            self.assertEqual(command[command.index("--find-links") + 1], str(self.wheelhouse))
            self.assertEqual(command[command.index("-r") + 1], str(self.lock))
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            kwargs["stdout"].write((f"Processing {self.wheel}\nSuccessfully installed example-1.2\n").encode())
            return subprocess.CompletedProcess(command, 0)
        self.assertEqual(command, [sys.executable, "-m", "pip", "check"])
        return subprocess.CompletedProcess(command, 0, "No broken requirements found.\n", "")

    def run_installer(self, process=None, environment=None):
        with patch.object(installer, "check_interpreter", return_value={"isolated_prefix": True}), \
             patch.object(installer, "audit_environment", return_value=self.environment if environment is None else environment), \
             patch.object(installer.subprocess, "run", side_effect=self.fake_process if process is None else process):
            return installer.run(**self.kwargs)

    def failure(self):
        return p.read(self.output.with_suffix(".failed.json"))

    def test_real_wheel_metadata_and_bytes_verified_before_mock_install(self):
        result = self.run_installer()
        self.assertTrue(result["installation_executed_by_this_cli"])
        self.assertEqual(len(self.events), 2)
        self.assertEqual(result["install_log_sha256"], p.digest(self.log))
        self.assertEqual(result["started_receipt_sha256"], p.digest(self.output.with_suffix(".started.json")))
        self.assertEqual(p.read(self.output), result)
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertFalse(self.output.with_suffix(".failed.json").exists())
        self.assertTrue(validate_environment_receipt(result, self.lock,
            wheel_receipt_path=self.wheel_receipt, install_log_path=self.log)["complete_locked_reinstall_recorded"])

    def test_tampered_wheel_is_rejected_before_any_pip_process(self):
        with zipfile.ZipFile(self.wheel, "a") as archive:
            archive.writestr("example/changed.py", "changed = True\n")
        with self.assertRaisesRegex(ValueError, "wheelhouse bytes"):
            self.run_installer()
        self.assertEqual(self.events, [])
        self.assertEqual(self.failure()["phase"], "preflight")
        self.assertIsNone(self.failure()["pip_install_returncode"])
        self.assertFalse(self.output.exists())

    def test_equivalent_root_path_is_canonicalized_before_install(self):
        self.kwargs["root"] = self.wheelhouse / ".."
        result = self.run_installer()
        self.assertTrue(result["installation_executed_by_this_cli"])
        self.assertEqual(len(self.events), 2)

    def test_coherent_receipt_cannot_override_required_wheel_hash(self):
        content = self.lock.read_text()
        current = wheel_records(self.wheelhouse)[0]["sha256"]
        self.lock.write_text(content.replace(current, "0" * 64))
        receipt = p.read(self.wheel_receipt)
        receipt["lock_sha256"] = p.digest(self.lock)
        self.wheel_receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "not allowed"):
            self.run_installer()
        self.assertEqual(self.events, [])

    def test_extra_wheel_cannot_be_silently_ignored(self):
        with zipfile.ZipFile(self.wheelhouse / "other-1.0-py3-none-any.whl", "w") as archive:
            archive.writestr("other-1.0.dist-info/METADATA", "Name: other\nVersion: 1.0\n")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.run_installer()
        self.assertEqual(self.events, [])

    def test_failed_pip_keeps_returncode_log_and_refuses_same_attempt(self):
        def fail(command, **kwargs):
            kwargs["stdout"].write(b"pip failure diagnostic\n")
            return subprocess.CompletedProcess(command, 7)
        with self.assertRaisesRegex(RuntimeError, "pip install failed"):
            self.run_installer(process=fail)
        self.assertEqual(self.failure()["pip_install_returncode"], 7)
        self.assertIn(b"pip failure diagnostic", self.log.read_bytes())
        self.assertEqual(self.failure()["install_log_sha256"], p.digest(self.log))
        self.assertFalse(self.output.exists())
        with self.assertRaises(FileExistsError):
            self.run_installer()

    def test_timed_out_install_is_retained_without_success_claim(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_installer(process=lambda command, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(command, 1)))
        self.assertEqual(self.failure()["error_type"], "TimeoutExpired")
        self.assertEqual(self.failure()["phase"], "pip-install")
        self.assertFalse(self.output.exists())

    def test_failed_dependency_check_is_not_an_install_success(self):
        def process(command, **kwargs):
            if command[-1] == "check":
                return subprocess.CompletedProcess(command, 1, "broken example dependency\n", "")
            return self.fake_process(command, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "pip check failed"):
            self.run_installer(process=process)
        self.assertEqual(self.failure()["pip_check_returncode"], 1)
        self.assertIn(b"broken example dependency", self.log.read_bytes())
        self.assertFalse(self.output.exists())

    def test_inputs_changing_during_install_cannot_get_success_receipt(self):
        def process(command, **kwargs):
            result = self.fake_process(command, **kwargs)
            if command[-1] == "check":
                receipt = p.read(self.wheel_receipt)
                receipt["unrecorded_change"] = True
                self.wheel_receipt.write_text(json.dumps(receipt))
            return result
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.run_installer(process=process)
        self.assertEqual(self.failure()["phase"], "final-verification")
        self.assertFalse(self.output.exists())

    def test_version_only_success_does_not_replace_real_installation_log(self):
        def process(command, **kwargs):
            if "install" in command:
                kwargs["stdout"].write(b"Requirement already satisfied: example\n")
                return subprocess.CompletedProcess(command, 0)
            return self.fake_process(command, **kwargs)
        with self.assertRaisesRegex(ValueError, "complete locked reinstall"):
            self.run_installer(process=process)
        self.assertFalse(self.output.exists())

    def test_prior_outputs_and_colliding_paths_are_rejected_without_mutation(self):
        self.log.parent.mkdir()
        self.log.write_bytes(b"prior attempt")
        with self.assertRaises(FileExistsError):
            self.run_installer()
        self.assertEqual(self.log.read_bytes(), b"prior attempt")
        self.assertFalse(self.output.with_suffix(".started.json").exists())
        self.kwargs["output"] = self.lock
        before = self.lock.read_bytes()
        with self.assertRaises(ValueError):
            self.run_installer()
        self.assertEqual(self.lock.read_bytes(), before)

    def test_isolation_failure_precedes_installation(self):
        with patch.object(installer, "check_interpreter", side_effect=ValueError("inherited environment")), \
             patch.object(installer.subprocess, "run") as process:
            with self.assertRaisesRegex(ValueError, "inherited"):
                installer.run(**self.kwargs)
            process.assert_not_called()
        self.assertEqual(self.failure()["phase"], "preflight")


if __name__ == "__main__":
    unittest.main()
