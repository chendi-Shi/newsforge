"""Private external prerequisites, using only opaque synthetic file bytes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from newsforge.delivery import bundle as b
from newsforge.delivery import external_bundle as e


def put(root, relative, contents):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


class ExternalBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nf-ext-bundle-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.bundle, self.clone = (self.root / name for name in ("source", "bundle", "clone"))
        self.source.mkdir()
        self.clone.mkdir()
        records = {}
        for name in sorted(e.PAYLOAD_FILES | e.CHECKOUT_FILES):
            # These deliberately are not valid executable pickle/model bytes.
            data = ("opaque synthetic input: " + name).encode("utf-8")
            put(self.source, name, data)
            records[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.source_report = {"schema_version": 1, "frozen_mind_and_protocol_files": records}
        put(self.source, e.SOURCE_REPORT, json.dumps(self.source_report).encode("utf-8"))
        for name in e.CHECKOUT_FILES | {e.SOURCE_REPORT}:
            put(self.clone, name, (self.source / name).read_bytes())

    def export(self):
        return e.export_bundle(self.source, self.bundle)["manifest_sha256"]

    def rewrite_manifest(self, value):
        path = put(self.bundle, "manifest.json", json.dumps(value).encode("utf-8"))
        digest = b.sha256_file(path)
        put(self.bundle, "manifest.sha256", (digest + "\n").encode("ascii"))
        return digest

    def assert_no_new_payload(self):
        self.assertFalse((self.clone / "artifacts").exists())
        self.assertFalse((self.clone / "data").exists())
        self.assertFalse((self.clone / "reports/v3").exists())

    def test_roundtrip_exact_inventory_and_repeat_import(self):
        digest = self.export()
        result = e.verify_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assertEqual(result["payload_files"], 27)
        self.assertEqual(result["required_checkout_files"], 9)
        self.assertFalse(result["external_events_included"])
        self.assertFalse(result["model_deserialization_performed"])
        result = e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assertEqual(result["copied_files"], 27)
        result = e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assertEqual(result["copied_files"], 0)
        self.assertEqual(result["identical_existing_files"], 27)
        self.assertFalse((self.bundle / "payload/newsforge").exists())
        self.assertFalse((self.bundle / "payload/docs").exists())
        self.assertFalse((self.bundle / "payload" / e.SOURCE_REPORT).exists())

    def test_copies_are_independent_from_both_source_and_destination(self):
        digest = self.export()
        e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        name = "data/processed/tfidf.joblib"
        original = (self.bundle / "payload" / name).read_bytes()
        (self.source / name).write_bytes(b"changed source")
        (self.clone / name).write_bytes(b"changed recipient")
        self.assertEqual((self.bundle / "payload" / name).read_bytes(), original)
        self.assertEqual((self.bundle / "payload" / name).stat().st_nlink, 1)

    def test_source_corruption_prevents_any_export(self):
        (self.source / "data/processed/svd.joblib").write_bytes(b"corruption")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.export()
        self.assertFalse(self.bundle.exists())

    def test_payload_corruption_prevents_all_import_writes(self):
        digest = self.export()
        path = self.bundle / "payload/data/processed/svd.joblib"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assert_no_new_payload()

    def test_conflicting_last_destination_prevents_earlier_writes(self):
        digest = self.export()
        target = put(self.clone, "reports/v3/tune.json", b"keep existing")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assertFalse((self.clone / "artifacts").exists())
        self.assertFalse((self.clone / "data").exists())
        self.assertEqual(target.read_bytes(), b"keep existing")

    def test_protocol_or_code_mismatch_is_not_imported_over(self):
        digest = self.export()
        for name in ("docs/EXTERNAL_ADRESSA_PROTOCOL.md", "newsforge/v3/residual.py"):
            with self.subTest(name=name):
                path = self.clone / name
                previous = path.read_bytes()
                path.write_bytes(b"changed checkout")
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
                self.assertEqual(path.read_bytes(), b"changed checkout")
                self.assert_no_new_payload()
                path.write_bytes(previous)

    def test_trusted_pin_mandatory_even_for_verify(self):
        self.export()
        for expected in (None, "", "bad", "0" * 64):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, "trusted"):
                e.verify_bundle(self.clone, self.bundle, expected_manifest_sha256=expected)

    def test_receiving_preregistration_must_be_identical(self):
        digest = self.export()
        target = self.clone / e.SOURCE_REPORT
        value = deepcopy(self.source_report)
        value["extra"] = "different provenance"
        target.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assert_no_new_payload()

    def test_cannot_remove_binding_even_with_new_manifest_digest(self):
        self.export()
        manifest = b._json(self.bundle / "manifest.json")
        manifest["files"].pop()
        manifest["total_bytes"] = sum(row["size"] for row in manifest["files"])
        digest = self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "inventory"):
            e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assert_no_new_payload()

    def test_extra_joblib_source_or_traversal_payload_is_forbidden(self):
        self.export()
        original = b._json(self.bundle / "manifest.json")
        for name in ("newsforge/external/evil.py", "data/processed/other.joblib", "../escape", "C:/escape"):
            with self.subTest(name=name):
                value = deepcopy(original)
                value["files"][0]["path"] = name
                digest = self.rewrite_manifest(value)
                with self.assertRaises(ValueError):
                    e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
                self.assert_no_new_payload()

    def test_incomplete_or_extra_source_inventory_refused(self):
        for action in ("missing", "extra"):
            with self.subTest(action=action):
                value = deepcopy(self.source_report)
                records = value["frozen_mind_and_protocol_files"]
                if action == "missing":
                    records.pop("data/processed/svd.joblib")
                else:
                    records["data/external/events.json"] = next(iter(records.values()))
                put(self.source, e.SOURCE_REPORT, json.dumps(value).encode())
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    self.export()
                self.assertFalse(self.bundle.exists())

    def test_duplicate_source_json_keys_rejected(self):
        put(self.source, e.SOURCE_REPORT, b'{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.export()

    def test_linked_destination_parent_rejected_before_copy(self):
        digest = self.export()
        directory = self.clone / "data"
        directory.mkdir()
        original = b._linked
        with patch.object(b, "_linked", side_effect=lambda path: path == directory or original(path)):
            with self.assertRaisesRegex(ValueError, "linked"):
                e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assertFalse((self.clone / "artifacts").exists())
        self.assertEqual(list(directory.iterdir()), [])

    def test_changed_manifest_record_cannot_override_source_report(self):
        self.export()
        value = b._json(self.bundle / "manifest.json")
        value["files"][0]["sha256"] = "0" * 64
        digest = self.rewrite_manifest(value)
        with self.assertRaisesRegex(ValueError, "receiving preregistration"):
            e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assert_no_new_payload()

    def test_export_destination_and_import_inside_bundle_rejected(self):
        digest = self.export()
        with self.assertRaises(FileExistsError):
            self.export()
        with self.assertRaisesRegex(ValueError, "outside the bundle"):
            e.import_bundle(self.bundle / "checkout", self.bundle, expected_manifest_sha256=digest)

    def test_checkout_files_are_required_and_never_created(self):
        digest = self.export()
        missing = self.clone / "newsforge/data.py"
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            e.import_bundle(self.clone, self.bundle, expected_manifest_sha256=digest)
        self.assertFalse(missing.exists())
        self.assert_no_new_payload()


if __name__ == "__main__":
    unittest.main()
