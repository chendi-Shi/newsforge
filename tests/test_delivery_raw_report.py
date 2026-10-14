"""Synthetic end-to-end evidence only: no raw dataset, ML or server workload."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from newsforge.delivery import acceptance as shared
from newsforge.delivery import pipeline as p
from newsforge.delivery import raw_acceptance as a
from newsforge.delivery import raw_report as r


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ranking(mean=0.5):
    return {key: {"mean": mean, "eligible_impressions": 4} for key in r.METRICS}


def ci(delta=0.1):
    return {"delta": delta, "ci95": [delta - 0.02, delta + 0.02],
            "resampling_unit": "user", "replicates": 100, "seed": 2026}


def v3_metrics(mean=0.3):
    return {"baseline": {"hits10": 2, "recall10": 0.2, "hits100": 5, "recall100": 0.5},
            "full13": {"hits10": 3, "recall10": mean, "hits100": 5, "recall100": 0.5,
                       "versus_baseline": ci(mean - 0.2)}}


def phase(name, gate, mean):
    return {"phase": name, "bindings": {"rows": {"count": 4}},
            "all_clicks": 10, "eligible_clicks": 10, "seen_clicks": 1, "shortfall_requests": 0,
            "delta": 2, "metrics": v3_metrics(mean), "deployment_gate_passed": gate,
            "invariants": {"per_request_candidate_and_hit100_invariant": True},
            "deployed_recipe": "full13+gate" if gate else "baseline"}


class SyntheticRun:
    """Construct the actual stage DAG with tiny materialized output fixtures."""

    def __init__(self, root, *, gate=False, research_skips=0, fresh_mean=0.3):
        root = Path(root).resolve()  # Normalize Windows temporary-directory 8.3 aliases.
        self.root = root
        self.acceptance_path = root.parent / "acceptance.json"
        self.output = root.parent / "published"
        self.research_skips = research_skips
        self.python = str(root.parent / "private-user" / "python.exe")
        for module, name in ((p, "newsforge/delivery/pipeline.py"),
                             (shared, "newsforge/delivery/acceptance.py")):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(module.__file__).read_bytes())
        for name in ("newsforge/train.py", "newsforge/v3/experiment.py", "docs/V3_PROTOCOL.md",
                     "requirements-windows.lock", "requirements-plot.txt"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# synthetic fixture\n")
        (root / a.LOCK).write_text("example==1.2 --hash=sha256:" + "a" * 64 + "\n")
        self.manifest = {"schema_version": p.SCHEMA, "created_utc": "2026-01-01T00:00:00+00:00",
            "source_files": {name: p.digest(root / name) for name in p.source_files(root)},
            "plan_sha256": p.canonical_hash(p.make_plan()), "raw_mode": "provided",
            "raw_files": {name: "b" * 64 for name in a.RAW_NAMES}}
        put(root, p.STATE / "workspace.json", self.manifest)
        for stage in p.ordered_stages():
            for name in stage.outputs:
                put(root, name, {"fixture": stage.name})
        self.report("data/processed/manifest.json", {"rows": 10, "impressions": 10, "candidates": 50,
            "positives": 10, "users": 6, "news": 8, "official_train_rows": 6, "official_dev_rows": 4,
            "splits": {key: {"impressions": 4, "candidates": 20} for key in ("0", "1", "2")},
            "features": {"dimensions": 8}, "private_path": "Z:\\private-user\\dataset",
            "article_title": "DO NOT PUBLISH THIS TITLE", "user_id": "U-private-person"})
        self.report("reports/ranking_experiment.json", {
            "runs": {arm: {"impressions": 4, "candidate_exposures": 20, "metrics": ranking()}
                     for arm in r.V1_ARMS},
            "seed_summary": {arm: ranking() for arm in ("mean_random", "mean_mixed", "ranker")},
            "paired_user_bootstrap": {arm: {key: ci() for key in r.METRICS} for arm in ("mean_mixed", "ranker")}})
        for name in ("ranking", "ranking-full"):
            self.report(f"reports/v2/{name}.json", {"impressions": 4, "candidates": 20,
                "metrics": {arm: ranking() for arm in r.V2_ARMS},
                "seed_statistics": {arm: ranking() for arm in ("content_only", "context_only", "full")},
                "paired_user_intervals": {"full_minus_content_only": {key: ci() for key in r.METRICS}}})
        self.report("reports/v2/catalog.json", {"queries": 4, "positive_clicked_items": 10,
            "eligible_positive_items": 10, "metrics": {arm: {"click_micro_recall10": 0.2,
            "click_micro_recall100": 0.5} for arm in r.CATALOG_ARMS},
            "paired_user_intervals": {"full_minus_no_popular_recall100": ci(-0.1)}})
        for version in ("v2", "v3"):
            self.report(f"reports/{version}/release_integrity.json", {"passed": True, "checks": {"fixture": True}})
        self.report("reports/v3/fit.json", {"fit_rows": {"count": 4}, "runs": {
            str(seed): {"sampling": {"requests": 4, "pairs": 8},
                "models": {arm: {"fit_metadata": {"pairs": 8, "loss": 0.4}}
                           for arm in ("full13", "no_category", "no_recent")}} for seed in p.SEEDS}})
        self.report("reports/v3/tune.json", {"selected_delta": 2,
            "delta_grid": {delta: {"metrics": v3_metrics()} for delta in ("0", "2", "5", "10")}})
        for name in ("confirm", "dev"):
            self.report(f"reports/v3/{name}.json", phase(name, gate, fresh_mean))
            self.report(f"reports-delivered/v3/{name}.json", phase(name, True, 0.4))
        case = {"batch_size": 1, "concurrency": 1, "rounds": 1, "warmup_http_requests": 1,
            "measured_http_requests": 4, "measured_queries": 4, "wall_seconds": 2,
            "successful_queries_per_second": 2, "http_errors": 0, "failed_queries": 0,
            "offline_http_mismatches": 0, "latency_ms": {"p50": 2, "p95": 3, "p99": 4}}
        self.report("reports/v3/service_benchmark.json", {"passed": True, "configurations": [case],
            "protocol": {"cpu_threads": 1, "server_workers": 1},
            "resources": {"server_peak_rss_bytes": 100, "server_pid": 555},
            "api_contract_checks": [{"name": "fixture", "passed": True}]})
        implementation = {"measured_queries": 4, "measured_queries_per_second": 2,
                          "latency_ms": {"p50": 2, "p95": 3, "p99": 4}}
        self.report("reports/v3/fast_retrieval_benchmark.json", {"equivalence": {"exact": True,
            "mismatch_field_occurrences": 0, "candidate_batches_compared": 4, "numeric_tolerance": 0},
            "summary": {mode: {"v2": implementation, "v3_fast": implementation,
                "observed_wall_speedup_v2_over_v3": 1, "observed_instrumented_query_cpu_ratio_v2_over_v3": 1}
                for mode in ("cold", "warm")}})
        self.refresh()

    def report(self, name, value):
        return put(self.root, name, value)

    def refresh(self):
        records, receipt_hashes = {}, {}
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index, stage in enumerate(p.ordered_stages()):
            directory = self.root / p.STATE / stage.name
            directory.mkdir(parents=True, exist_ok=True)
            log = "Private Z:\\private-user\\path\n"
            if stage.name in a.TEST_LOGS:
                skipped = 2 if stage.name == "unit-tests" else self.research_skips
                log += "Ran 10 tests in 0.1s\n\nOK" + (f" (skipped={skipped})" if skipped else "") + "\n"
                target = self.root / a.TEST_LOGS[stage.name]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(log, encoding="utf-8")
            (directory / "output.log").write_text(log, encoding="utf-8")
            started = {"stage": stage.name, "status": "running", "python": self.python,
                "argv": [self.python, *stage.argv],
                "started_utc": (start + timedelta(seconds=2 * index + 1)).isoformat(),
                "stage_definition_sha256": p.canonical_hash(asdict(stage)),
                "workspace_manifest_sha256": p.digest(self.root / p.STATE / "workspace.json"),
                "dependencies": {name: receipt_hashes[name] for name in stage.dependencies},
                "inputs": {path: digest for name in stage.dependencies for path, digest in records[name]["outputs"].items()}}
            put(self.root, p.STATE / stage.name / "started.json", started)
            outputs = list(stage.outputs) + ([a.TEST_LOGS[stage.name]] if stage.name in a.TEST_LOGS else [])
            result = {**started, "status": "success", "returncode": 0,
                "finished_utc": (start + timedelta(seconds=2 * index + 2)).isoformat(), "elapsed_seconds": 1,
                "outputs": {name: p.digest(self.root / name) for name in outputs},
                "log_sha256": p.digest(directory / "output.log")}
            put(self.root, p.STATE / stage.name / "success.json", result)
            records[stage.name] = result
            receipt_hashes[stage.name] = p.digest(directory / "success.json")
        evidence = a.Evidence(self.root)
        chain = a.receipt_chain(evidence, self.manifest, python=self.python)
        comparison = a.comparisons(evidence)
        for name in self.manifest["source_files"]:
            evidence.digest(name)
        for name in a.RAW_NAMES:
            evidence.hashes[f"data/raw/{name}"] = "b" * 64
        receipt = {"schema_version": 1, "passed": True, "created_utc": "2026-01-01T00:02:00+00:00",
            "verifier_source_sha256": p.digest(a.__file__), "plan_sha256": self.manifest["plan_sha256"],
            "workspace_manifest_sha256": p.digest(self.root / p.STATE / "workspace.json"),
            "environment": {"passed": True, "locked_distributions": 1,
                "lock_sha256": p.digest(self.root / a.LOCK), "wheel_receipt_sha256": "c" * 64,
                "install_log_sha256": "d" * 64, "installation_receipt_sha256": "e" * 64},
            "raw_files": {name: {"sha256": "b" * 64, "bytes": 10} for name in a.RAW_NAMES},
            "completed_stages": chain, "comparison": comparison,
            "release_audits": {version: {"passed": True, "checks": {"fixture": True},
                "report_sha256": evidence.digest(f"reports/{version}/release_integrity.json")} for version in ("v2", "v3")},
            "http": {"passed": True, "measured_queries": 4, "configurations": 1, "api_contracts": 1},
            "verified_sha256_files": evidence.hashes}
        self.receipt = receipt
        put(self.acceptance_path.parent, self.acceptance_path.name, receipt)


class RawReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = SyntheticRun(Path(self.temp.name) / "workspace")

    def build(self):
        return r.build_summary(self.fixture.root, self.fixture.acceptance_path)

    def edit_receipt(self, **values):
        value = deepcopy(self.fixture.receipt)
        value.update(values)
        put(self.fixture.acceptance_path.parent, self.fixture.acceptance_path.name, value)

    def test_full_actual_dag_summary_and_markdown_with_no_heavy_audit(self):
        with patch.object(a, "audit", side_effect=AssertionError("full audit must not run")):
            value = r.publish(self.fixture.root, self.fixture.acceptance_path, self.fixture.output)
        self.assertEqual(len(value["stages"]), len(p.ordered_stages()))
        self.assertTrue(all(stage["status"] == "success" for stage in value["stages"]))
        self.assertEqual(p.read(self.fixture.output / "execution_summary.json"), value)
        rendered = (self.fixture.output / "RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("Confirm gate: **False**", rendered)
        self.assertIn("serving recipe: **baseline**", rendered)
        self.assertIn(value["acceptance_sha256"], rendered)
        self.assertIn("v1 paired user bootstrap", rendered)
        self.assertIn("v2 ranking ablation intervals", rendered)
        self.assertIn("[-0.120000, -0.080000]", rendered)

    def test_publication_is_append_only(self):
        r.publish(self.fixture.root, self.fixture.acceptance_path, self.fixture.output)
        original = (self.fixture.output / "RESULTS.md").read_bytes()
        with self.assertRaisesRegex(ValueError, "retain previous results"):
            r.publish(self.fixture.root, self.fixture.acceptance_path, self.fixture.output)
        self.assertEqual((self.fixture.output / "RESULTS.md").read_bytes(), original)

    def test_failed_or_foreign_acceptance_rejected(self):
        for change, message in (({"passed": False}, "passed raw acceptance"),
                                ({"workspace_manifest_sha256": "f" * 64}, "manifest differs"),
                                ({"verifier_source_sha256": "f" * 64}, "verifier source")):
            with self.subTest(change=change):
                self.edit_receipt(**change)
                with self.assertRaisesRegex(ValueError, message):
                    self.build()

    def test_receipt_report_model_and_log_changes_rejected(self):
        for name in (str(p.STATE / "data/success.json"), "reports/v3/dev.json", "artifacts/mean_random_seed17.pt",
                     str(p.STATE / "v3-audit/output.log"), "newsforge/train.py"):
            with self.subTest(name=name):
                target = self.fixture.root / name
                original = target.read_bytes()
                target.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    self.build()
                target.write_bytes(original)

    def test_claimed_chain_cannot_replace_verified_chain(self):
        chain = deepcopy(self.fixture.receipt["completed_stages"])
        chain[-1]["elapsed_seconds"] = 999
        self.edit_receipt(completed_stages=chain)
        with self.assertRaisesRegex(ValueError, "stage chain differs"):
            self.build()

    def test_only_bound_fresh_reports_allowed(self):
        del self.fixture.receipt["verified_sha256_files"]["reports/v3/dev.json"]
        self.edit_receipt()
        with self.assertRaisesRegex(ValueError, "absent from acceptance"):
            self.build()

    def test_missing_fresh_report_never_falls_back_to_historical(self):
        (self.fixture.root / "reports/v3/dev.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()

    def test_skip_counts_and_overlapping_suites_are_explicit(self):
        value = self.build()
        self.assertEqual(value["tests"]["unit-tests"]["passed"], 8)
        self.assertEqual(value["tests"]["unit-tests"]["skipped"], 2)
        self.assertEqual(value["tests"]["research-unit-tests"]["passed"], 10)
        self.assertNotIn("unique_tests", value)
        self.assertIn("overlapping coverage", r.markdown(value))

    def test_research_skip_is_rejected_even_with_successful_chain(self):
        self.fixture.research_skips = 1
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "research tests contain skips"):
            self.build()

    def test_source_titles_ids_private_paths_and_pid_not_published(self):
        value = self.build()
        public = json.dumps(value) + r.markdown(value)
        for secret in ("private-user", "DO NOT PUBLISH", "U-private-person", "server_pid", "C:\\", "Z:\\", str(self.fixture.root)):
            self.assertNotIn(secret, public)

    def test_negative_changes_are_retained_and_not_used_for_selection(self):
        value = self.build()
        self.assertAlmostEqual(value["comparison"]["dev"]["fresh_minus_previous"]["full13"]["recall10"], -0.1)
        self.assertEqual(value["v3"]["dev"]["metrics"]["full13"]["recall10"], 0.3)
        self.assertEqual(value["comparison"]["dev"]["previous"]["metrics"]["full13"]["recall10"], 0.4)
        self.assertFalse(value["comparison"]["dev"]["scientific_summary_exactly_equal"])
        self.assertEqual(value["v3"]["dev"]["deployed_recipe"], "baseline")
        self.assertLess(value["v2"]["catalog"]["paired_user_intervals"]["full_minus_no_popular_recall100"]["delta"], 0)

    def test_gate_inconsistent_recipe_is_rejected(self):
        value = p.read(self.fixture.root / "reports/v3/dev.json")
        value["deployed_recipe"] = "full13+gate"
        self.fixture.report("reports/v3/dev.json", value)
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "deployed recipe disagrees"):
            self.build()

    def test_nonfinite_aggregate_rejected_before_publication(self):
        value = p.read(self.fixture.root / "data/processed/manifest.json")
        value["users"] = float("nan")
        self.fixture.report("data/processed/manifest.json", value)
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "finite numeric aggregate"):
            r.publish(self.fixture.root, self.fixture.acceptance_path, self.fixture.output)
        self.assertFalse(self.fixture.output.exists())

    def test_empty_scientific_arms_cannot_be_published_as_success(self):
        value = p.read(self.fixture.root / "reports/ranking_experiment.json")
        value["runs"] = {}
        self.fixture.report("reports/ranking_experiment.json", value)
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "incomplete v1 experiment arms"):
            self.build()

    def test_incomplete_metric_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete ranking metrics"):
            r.ranking_metrics({"auc": {"mean": 0.5}})

    def test_http_failures_cannot_be_hidden_by_passed_flag(self):
        value = p.read(self.fixture.root / "reports/v3/service_benchmark.json")
        value["configurations"][0]["http_errors"] = 1
        self.fixture.report("reports/v3/service_benchmark.json", value)
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "HTTP failures"):
            self.build()


if __name__ == "__main__":
    unittest.main()
