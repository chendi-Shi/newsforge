import unittest
from pathlib import Path
import zipfile
from tests.support import workspace_temporary_directory

import numpy as np

from newsforge.delivery.retrain import compare_arrays, verify_protocol, digest, PHASES
import newsforge.delivery.retrain as replay
from newsforge.delivery.wheels import wheel_records, render_lock


class ReplayContracts(unittest.TestCase):
    def test_reject_changed_reference_and_empty_archive(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            for phase in PHASES:
                (root / f"{phase}.json").write_text("{}")
            protocol = {"runner_sha256": digest(replay.__file__), "reference_reports": {
                phase: digest(root / f"{phase}.json") for phase in PHASES}}
            verify_protocol(protocol, root)
            (root / "dev.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verify_protocol(protocol, root)
            np.savez(root / "empty.npz")
            with self.assertRaises(ValueError):
                compare_arrays(root / "empty.npz", root / "empty.npz", atol=0.)

    def test_float_tolerance_does_not_relax_integer_counts(self):
        with workspace_temporary_directory() as directory:
            a, b = Path(directory) / "a.npz", Path(directory) / "b.npz"
            np.savez(a, weights=np.array([1.]), counts=np.array([100], dtype=np.int32))
            np.savez(b, weights=np.array([1. + 1e-10]), counts=np.array([101], dtype=np.int32))
            result = compare_arrays(a, b, atol=1e-8)
            self.assertTrue(result["weights"]["passed"])
            self.assertFalse(result["weights"]["exact"])
            self.assertFalse(result["counts"]["passed"])

    def test_nonfinite_and_schema_mismatch_fail(self):
        with workspace_temporary_directory() as directory:
            a, b = Path(directory) / "a.npz", Path(directory) / "b.npz"
            np.savez(a, weights=np.array([1.]))
            np.savez(b, weights=np.array([np.nan]))
            self.assertFalse(compare_arrays(a, b, atol=1.)["weights"]["passed"])
            np.savez(a, weights=np.array([np.inf]))
            np.savez(b, weights=np.array([np.inf]))
            self.assertFalse(compare_arrays(a, b, atol=1.)["weights"]["passed"])
            np.savez(b, other=np.array([1.]))
            with self.assertRaises(ValueError):
                compare_arrays(a, b, atol=0.)


class WheelLockContracts(unittest.TestCase):
    def test_hash_is_from_archive_bytes_and_duplicate_packages_fail(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            wheel = root / "example-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("example-1.0.dist-info/METADATA", "Name: Example\nVersion: 1.0\n")
                archive.writestr("example/_vendor/dep-2.0.dist-info/METADATA", "Name: Dep\nVersion: 2.0\n")
            before = wheel_records(root)
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr("payload.txt", "changed")
            after = wheel_records(root)
            self.assertNotEqual(before[0]["sha256"], after[0]["sha256"])
            self.assertIn("example==1.0 --hash=sha256:" + after[0]["sha256"], render_lock(after))
            (root / "duplicate.whl").write_bytes(wheel.read_bytes())
            with self.assertRaises(ValueError):
                wheel_records(root)


if __name__ == "__main__":
    unittest.main()
