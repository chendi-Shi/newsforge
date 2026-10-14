"""Private bundle isolation, corruption and cross-platform path contracts."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from newsforge.delivery import bundle
from tests.support import workspace_temporary_directory


def fixture_manifest(root):
    files = []
    for relative, contents in (("data/processed/manifest.json", b'{"fixture":true}'),
                               ("artifacts/retriever.pt", b"opaque checkpoint; never deserialize")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        files.append({"path": relative, "size": len(contents),
                      "sha256": hashlib.sha256(contents).hexdigest()})
    source = root / "newsforge/v3/serve.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"# exact source\n")
    return {"format": bundle.FORMAT, "scope": bundle.SCOPE, "private_data": True,
            "source_commit": "a" * 40, "files": files,
            "required_checkout_files": [{"path": "newsforge/v3/serve.py",
                                         "size": source.stat().st_size,
                                         "sha256": bundle.sha256_file(source)}],
            "total_bytes": sum(record["size"] for record in files)}


def write_manifest(directory, manifest):
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "manifest.sha256").write_text(bundle.sha256_file(path) + "\n", encoding="ascii")


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.context = workspace_temporary_directory("delivery-bundle-test-")
        self.root = Path(self.context.__enter__())
        self.source, self.output, self.clone = (self.root / name for name in ("source", "bundle", "clone"))
        self.manifest = fixture_manifest(self.source)
        self.clone.mkdir()
        source = self.clone / "newsforge/v3/serve.py"
        source.parent.mkdir(parents=True)
        source.write_bytes((self.source / "newsforge/v3/serve.py").read_bytes())

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def export(self):
        with patch.object(bundle, "build_plan", return_value=self.manifest):
            return bundle.export_bundle(self.source, self.output)

    def test_roundtrip_resumable_and_input_unchanged(self):
        evidence = self.export()
        self.assertTrue(bundle.verify_bundle(self.output, expected_manifest_sha256=evidence["manifest_sha256"])["verified"])
        first = bundle.import_bundle(self.output, self.clone, expected_manifest_sha256=evidence["manifest_sha256"])
        self.assertEqual(first["copied_files"], 2)
        self.assertEqual(bundle.import_bundle(self.output, self.clone)["identical_existing_files"], 2)
        for record in self.manifest["files"]:
            self.assertEqual(bundle.sha256_file(self.source / record["path"]), record["sha256"])
            self.assertEqual(bundle.sha256_file(self.clone / record["path"]), record["sha256"])

    def test_copies_are_independent_not_source_hardlinks(self):
        self.export()
        bundle.import_bundle(self.output, self.clone)
        path = "artifacts/retriever.pt"
        before = (self.output / "payload" / path).read_bytes()
        (self.source / path).write_bytes(b"changed source")
        self.assertEqual((self.output / "payload" / path).read_bytes(), before)
        (self.clone / path).write_bytes(b"changed clone")
        self.assertEqual((self.output / "payload" / path).read_bytes(), before)
        self.assertEqual((self.output / "payload" / path).stat().st_nlink, 1)

    def test_corrupt_same_size_payload_blocks_all_writes(self):
        self.export()
        target = self.output / "payload/artifacts/retriever.pt"
        target.write_bytes(b"x" * target.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            bundle.import_bundle(self.output, self.clone)
        self.assertFalse((self.clone / "data").exists())

    def test_existing_conflicting_target_blocks_all_writes(self):
        self.export()
        target = self.clone / "artifacts/retriever.pt"
        target.parent.mkdir()
        target.write_bytes(b"x" * self.manifest["files"][1]["size"])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            bundle.import_bundle(self.output, self.clone)
        self.assertFalse((self.clone / "data").exists())
        self.assertEqual(target.read_bytes(), b"x" * self.manifest["files"][1]["size"])

    def test_existing_parent_file_blocks_all_writes(self):
        self.export()
        # The first inventory entry is data/processed/manifest.json. A later
        # entry's conflicting parent must be rejected before creating data/.
        target = self.clone / "artifacts"
        target.write_bytes(b"keep this existing regular file")
        with self.assertRaisesRegex(ValueError, "not a directory"):
            bundle.import_bundle(self.output, self.clone)
        self.assertFalse((self.clone / "data").exists())
        self.assertEqual(target.read_bytes(), b"keep this existing regular file")

    def test_symlinked_destination_ancestor_blocks_all_writes(self):
        self.export()
        target = self.clone / "artifacts"
        target.mkdir()
        # This exercises reparse rejection even on accounts without Windows
        # symlink creation privilege; the real-symlink test below is separate.
        original = bundle._linked
        with patch.object(bundle, "_linked", side_effect=lambda p: p == target or original(p)):
            with self.assertRaisesRegex(ValueError, "linked"):
                bundle.import_bundle(self.output, self.clone)
        self.assertFalse((self.clone / "data").exists())
        self.assertEqual(list(target.iterdir()), [])

    def test_checkout_mismatch_blocks_import(self):
        self.export()
        (self.clone / "newsforge/v3/serve.py").write_bytes(b"# wrong source\n")
        with self.assertRaises(ValueError):
            bundle.import_bundle(self.output, self.clone)
        self.assertFalse((self.clone / "data").exists())

    def test_export_fails_on_bound_source_corruption_without_manifest(self):
        (self.source / "artifacts/retriever.pt").write_bytes(b"x" * self.manifest["files"][1]["size"])
        with self.assertRaisesRegex(ValueError, "SHA256/size"):
            self.export()
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertFalse(list(self.output.rglob("*.delivery-part-*")))

    def test_manifest_pin_and_checksum(self):
        self.export()
        with self.assertRaisesRegex(ValueError, "trusted"):
            bundle.verify_bundle(self.output, expected_manifest_sha256="0" * 64)
        (self.output / "manifest.sha256").write_text("0" * 64, encoding="ascii")
        with self.assertRaisesRegex(ValueError, "checksum"):
            bundle.verify_bundle(self.output)

    def test_refuse_existing_export_directory(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_chunked_copy_refuses_source_larger_or_smaller_than_record(self):
        record = dict(self.manifest["files"][1])
        for change in (-1, 1):
            with self.subTest(change=change):
                record["size"] = self.manifest["files"][1]["size"] + change
                with patch.object(bundle, "CHUNK_BYTES", 3), self.assertRaises(ValueError):
                    bundle._copy_record(self.source, self.clone, record)
                self.assertFalse((self.clone / record["path"]).exists())
                self.assertFalse(list(self.clone.rglob("*.delivery-part-*")))

    def test_reject_absolute_traversal_drives_unc_and_aliases(self):
        for path in ("/tmp/escape", "../escape", "data/../escape", "C:/escape", "C:escape",
                     "\\\\server\\share", "data\\escape", "data//escape", "./data", "data/",
                     "data/CON", "data/NUL.txt", "data/file.", "data/file ", "data/a\x00b", "data/a:b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                bundle.relative_path(path)

    def test_reject_windows_superscript_device_aliases(self):
        for path in ("data/COM¹", "data/COM².json", "data/LPT³.npy"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                bundle.relative_path(path)

    def test_allow_only_the_frozen_retraining_comparison_counts(self):
        for path in ("artifacts/v2/dev_catalog_counts.npz", "artifacts/v3/confirm/counts.npz",
                     "artifacts/v3/dev/counts.npz"):
            self.assertTrue(bundle._payload_allowed(path))
        for path in ("artifacts/v3/fit/counts.npz", "artifacts/v3/dev/other.npz"):
            self.assertFalse(bundle._payload_allowed(path))

    def test_reject_corrupt_size_bool_hash_and_case_collisions(self):
        mutations = (lambda m: m["files"][0].update(size=True),
                     lambda m: m["files"][0].update(size=-1),
                     lambda m: m["files"][0].update(sha256="g" * 64),
                     lambda m: m.update(total_bytes=m["total_bytes"] + 1),
                     lambda m: m["files"].append(dict(m["files"][0])),
                     lambda m: m["files"].append({**m["files"][0], "path": "data/processed/MANIFEST.json"}))
        for mutate in mutations:
            manifest = json.loads(json.dumps(self.manifest))
            mutate(manifest)
            with self.assertRaises(ValueError):
                bundle._validate_manifest(manifest)

    def test_payload_cannot_install_code_environment_or_arbitrary_checkpoint(self):
        self.export()
        for path in ("newsforge/v3/serve.py", ".git/config", ".venv/pyvenv.cfg",
                     "artifacts/payload.py", "artifacts/evil.joblib", "data/processed/evil.py"):
            manifest = json.loads(json.dumps(self.manifest))
            manifest["files"][0]["path"] = path
            write_manifest(self.output, manifest)
            with self.subTest(path=path), self.assertRaises(ValueError):
                bundle.import_bundle(self.output, self.clone)

    def test_json_duplicate_keys_are_rejected(self):
        self.export()
        path = self.output / "manifest.json"
        path.write_text('{"format":"a","format":"b"}', encoding="utf-8")
        (self.output / "manifest.sha256").write_text(bundle.sha256_file(path), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bundle.verify_bundle(self.output)

    def test_oversized_manifest_rejected_before_parsing(self):
        self.export()
        with patch.object(bundle, "MAX_MANIFEST_BYTES", 10), self.assertRaisesRegex(ValueError, "oversized"):
            bundle.verify_bundle(self.output)

    def test_symlink_payload_is_rejected_when_supported(self):
        self.export()
        path = self.output / "payload/artifacts/retriever.pt"
        path.unlink()
        try:
            path.symlink_to(self.source / "artifacts/retriever.pt")
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is unavailable to this test account")
        with self.assertRaisesRegex(ValueError, "linked"):
            bundle.verify_bundle(self.output)


if __name__ == "__main__":
    unittest.main()
