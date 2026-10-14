"""Tiny completed-artifact fixtures; no real Adressa inputs or model inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("verify_external_results", Path(__file__).resolve().parents[1] / "scripts/verify_external_results.py")
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def inventory(folder, names):
    return {name: {"sha256": v.digest(folder / name), "bytes": (folder / name).stat().st_size} for name in names}


def summary(n):
    return {"queries": n, "users": 1, "methods": {method: {"denominator": n, "hits10": n, "hits100": n,
        "hit_rate10": 1., "hit_rate100": 1., "mrr10": 1.} for method in v.METHODS},
        "comparisons": {key: {"difference": 0., "ci95": [0., 0.], "user_clusters": 1,
            "bootstrap_repetitions": 1000, "bootstrap_rng_seed": 2026, "valid_repetitions": 1000}
            for key, _, _ in v.COMPARISONS}}


class Fixture:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.run, self.reports = self.root / "private-run", self.root / "reports/result"
        self.output = self.root / "public-verification.json"
        for name in ("docs/EXTERNAL_ADRESSA_PROTOCOL.md", "docs/EXTERNAL_ADRESSA_RESOURCE_AMENDMENT.md",
                     "requirements-delivery-windows.lock", "newsforge/external/common.py"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("Synthetic frozen input.\n")
        frozen = "docs/EXTERNAL_ADRESSA_PROTOCOL.md"
        source = {"frozen_mind_and_protocol_files": {frozen: {"sha256": v.digest(self.root / frozen),
                    "bytes": (self.root / frozen).stat().st_size}}}
        put(self.root / "reports/external/adressa_source.json", source)
        self.bindings = {"frozen_files": {frozen: v.digest(self.root / frozen)},
            "implementation": {"newsforge/external/common.py": v.digest(self.root / "newsforge/external/common.py")},
            "source_report_sha256": v.digest(self.root / "reports/external/adressa_source.json"),
            "dependency_lock_sha256": v.digest(self.root / "requirements-delivery-windows.lock"),
            "resource_amendment_sha256": v.digest(self.root / "docs/EXTERNAL_ADRESSA_RESOURCE_AMENDMENT.md"),
            "environment": {"numeric_threads": 1}}
        self.daily = dict(zip(v.DAYS, (43, 43, 42)))
        self.n = 128
        arrays = {"queries": np.column_stack((np.arange(128), np.concatenate([
            np.full(n, v.EVAL_START + day * 86400) for day, n in enumerate(self.daily.values())]),
            np.full(128, 93117), np.ones(128, dtype=int))).astype(np.int64),
            "pools": np.zeros((128, 100), dtype=np.int32), "sizes": np.full(128, 2, dtype=np.int16),
            "orders": np.full((128, 5, 100), -1, dtype=np.int16), "scores": np.zeros((128, 5, 100)),
            "hits10": np.ones((128, 5), dtype=np.uint8), "hits100": np.ones((128, 5), dtype=np.uint8),
            "reciprocal10": np.ones((128, 5))}
        arrays["pools"][:, :2] = [1, 2]
        arrays["orders"][:, :, :2] = [0, 1]
        arrays["scores"][:, :, :2] = [0., -1.]
        for stage in ("acquisition", "prepare", "evaluate", "aggregate"):
            (self.run / stage).mkdir(parents=True)
        for name, value in arrays.items():
            np.save(self.run / "evaluate" / f"{name}.npy", value, allow_pickle=False)
        flags = np.zeros((3, 4), dtype=np.uint8)
        flags[1:, 1:] = 1
        np.save(self.run / "prepare/coverage.npy", flags, allow_pickle=False)
        for name in v.PREPARED - {"coverage.npy"}:
            (self.run / "prepare" / name).write_bytes(b"synthetic bound bytes only")
        (self.run / "evaluate/reference_counts.sqlite").write_bytes(b"synthetic reference identity only")
        self.audit = {"schema_version": 1, "status": "passed", "queries": 128,
            "indices_sha256": __import__("hashlib").sha256(np.linspace(0, 127, 128, dtype=np.int64).tobytes()).hexdigest(),
            "candidate_id_equality": "exact", "ranking_equality": "exact", "hit_equality": "exact",
            "score_atol": 1e-8, "score_rtol": 0., "score_max_absolute_error": 0.,
            "all_method_per_query_candidate_set_invariant": True, "all_method_per_query_hit100_invariant": True}
        put(self.run / "evaluate/audit.json", self.audit)
        put(self.run / "acquisition/ranges.json", {"parts": [{"name": "retained-in-old-run-only"}]})
        self.diagnostics = {"candidate_items": 256, "history_items": 0, "shortfall": 98 * 128,
            "queries_with_shortfall": 128, "empty_history": 128, "unavailable_target": 0,
            "already_read_target": 0, "queries_with_fallback": 0, "fallback_items": 0,
            **{f"candidate_{name}": 0 if i == 0 else 256 for i, name in enumerate(v.COVERAGE)},
            **{f"history_{name}": 0 for name in v.COVERAGE}}
        self.counts = {"valid_before_dedup": 150, "deduplicated_events": 140, "duplicate_events_removed": 10,
                       "raw_lines": 150}
        self.report_edit = None
        self.refresh()

    def refresh(self):
        acquired = {"schema_version": 1, "status": "passed", "sha256": "a" * 64,
            "ranges_manifest_sha256": v.digest(self.run / "acquisition/ranges.json"),
            "bindings": {key: value for key, value in self.bindings.items() if key != "resource_amendment_sha256"}}
        put(self.run / "acquisition/receipt.json", acquired)
        prepared = {"schema_version": 1, "status": "passed", "bindings": self.bindings,
            "acquisition_receipt_sha256": v.digest(self.run / "acquisition/receipt.json"),
            "archive_sha256": "a" * 64, "sample_identity_sha256": "b" * 64,
            "target_independent_sampling": True, "external_model_fitting": False,
            "files": inventory(self.run / "prepare", v.PREPARED), "daily_samples": self.daily,
            "identity": {"articles": 2, "daily_populations": self.daily}, "counts": self.counts,
            "transformed": {"coverage": {"articles": 2, "empty_title": 0, "zero_tfidf": 2, "zero_svd": 2, "unknown_category": 2}}}
        put(self.run / "prepare/receipt.json", prepared)
        evaluated = {"schema_version": 1, "status": "passed", "bindings": self.bindings,
            "prepare_receipt_sha256": v.digest(self.run / "prepare/receipt.json"), "sample_identity_sha256": "b" * 64,
            "files": inventory(self.run / "evaluate", v.EVALUATED), "queries": 128,
            "methods": list(v.METHODS), "diagnostics": self.diagnostics}
        put(self.run / "evaluate/receipt.json", evaluated)
        report = {"schema_version": 1, "status": "passed", "bindings": self.bindings,
            "phase_receipts_sha256": {name: v.digest(self.run / name / "receipt.json") for name in ("acquisition", "prepare", "evaluate")},
            "per_query_files": evaluated["files"], "archive_sha256": "a" * 64, "sample_identity_sha256": "b" * 64,
            "external_model_fitting": False, "external_hyperparameter_selection": False,
            "prospective_future_generalization": False, "online_ab_test": False,
            "delta": 10, "seeds": [17, 29, 43], "audit": self.audit, "overall": summary(128),
            "by_day": {day: summary(n) for day, n in self.daily.items()}, "daily_samples": self.daily,
            "identity_counts": prepared["identity"], "source_counts": self.counts,
            "query_diagnostics": self.diagnostics, "article_text_coverage": prepared["transformed"]["coverage"],
            "positive_transfer_supported_on_preregistered_sample": False,
            "ignored_private_metadata": {"article_title": "secret-title", "user": "private-person", "path": "Z:/private-run"}}
        if self.report_edit:
            self.report_edit(report)
        put(self.reports / "evaluation.json", report)
        put(self.reports / "audit.json", self.audit)
        put(self.run / "aggregate/receipt.json", {"schema_version": 1, "status": "passed", "bindings": self.bindings,
            "public_evaluation_sha256": v.digest(self.reports / "evaluation.json"),
            "public_audit_sha256": v.digest(self.reports / "audit.json"),
            "all_prediction_counts_recomputed": True, "positive_transfer_supported": False})


class VerifyExternalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def verify(self):
        return v.verify(self.f.run, self.f.reports, repo=self.f.root)

    def test_valid_completed_artifacts_with_legacy_acquisition_and_no_parts(self):
        result = v.publish(self.f.run, self.f.reports, self.f.output, repo=self.f.root)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["queries_checked"], 128)
        self.assertTrue(result["resource_amendment_bound"])
        self.assertFalse(result["bootstrap_recomputed"])
        self.assertFalse(result["positive_transfer_supported"])
        public = self.f.output.read_text()
        for private in ("secret-title", "private-person", "Z:/", str(self.f.root)):
            self.assertNotIn(private, public)

    def test_valid_score_nan_rejected_even_when_all_hashes_match(self):
        path = self.f.run / "evaluate/scores.npy"
        values = np.load(path)
        values[0, 1, 0] = float("nan")
        np.save(path, values, allow_pickle=False)
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, "nonfinite valid candidate score"):
            self.verify()

    def test_nonconserved_candidate_counts_rejected(self):
        self.f.diagnostics["shortfall"] -= 1
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, "candidate/shortfall count conservation"):
            self.verify()

    def test_existing_output_never_overwritten(self):
        self.f.output.write_bytes(b"retained original")
        with self.assertRaises(FileExistsError):
            v.publish(self.f.run, self.f.reports, self.f.output, repo=self.f.root)
        self.assertEqual(self.f.output.read_bytes(), b"retained original")

    def test_modified_source_or_array_bytes_rejected(self):
        for path in (self.f.root / "docs/EXTERNAL_ADRESSA_RESOURCE_AMENDMENT.md", self.f.run / "evaluate/hits10.npy"):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    self.verify()
                path.write_bytes(original)

    def test_changed_ablation_sign_and_nonfinite_ci_rejected(self):
        for field, value in (("difference", 0.1), ("ci95", [float("nan"), 0.1])):
            with self.subTest(field=field):
                self.f.report_edit = lambda report: report["overall"]["comparisons"][v.COMPARISONS[1][0]].update({field: value})
                self.f.refresh()
                with self.assertRaises(ValueError):
                    self.verify()

    def test_query_denominator_and_reference_identity_rejected(self):
        self.f.report_edit = lambda report: report["by_day"][v.DAYS[0]].update({"queries": 42})
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, "query denominator differs"):
            self.verify()
        self.f.report_edit = None
        self.f.audit["indices_sha256"] = "c" * 64
        put(self.f.run / "evaluate/audit.json", self.f.audit)
        self.f.refresh()
        with self.assertRaisesRegex(ValueError, "reference audit contract"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
