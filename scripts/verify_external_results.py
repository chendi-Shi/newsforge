"""Supplemental read-only verification of completed Adressa artifacts.

Checks retained hashes, count identities and saved predictions. It does not
reparse the original archive, rerun model inference, or recompute bootstrap CIs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np


METHODS = ("baseline", "full13+gate", "no_category+gate", "no_recent+gate", "full13+no_gate")
DAYS = ("2017-01-05", "2017-01-06", "2017-01-07")
EVAL_START = 1483574400
PREPARED = {"events.sqlite", "features.npy", "coverage.npy", "first_seen.npy"}
EVALUATED = {"queries.npy", "pools.npy", "sizes.npy", "orders.npy", "scores.npy",
             "hits10.npy", "hits100.npy", "reciprocal10.npy", "audit.json", "reference_counts.sqlite"}
COMPARISONS = (("full13+gate_minus_baseline", 1, 0),
               ("no_category+gate_minus_full13+gate", 2, 1),
               ("no_recent+gate_minus_full13+gate", 3, 1),
               ("full13+no_gate_minus_full13+gate", 4, 1))
COVERAGE = ("empty_title", "zero_tfidf", "zero_svd", "unknown_category")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite numeric aggregate")
    return value


def count(value, upper=None):
    require(type(value) is int and value >= 0 and (upper is None or value <= upper), "invalid aggregate count")
    return value


def equal(left, right):
    require(math.isclose(finite(left), finite(right), rel_tol=0, abs_tol=1e-12), "numeric reconciliation failed")


def relative(root, name):
    require(isinstance(name, str) and bool(name) and "\\" not in name
            and not PureWindowsPath(name).drive and not PurePosixPath(name).is_absolute()
            and not any(part in ("", ".", "..") for part in name.split("/")), "unsafe evidence filename")
    root = Path(root).resolve()
    path = root / name
    require(path.resolve().is_relative_to(root), "evidence escapes its directory")
    require(not any(item.is_symlink() for item in (path, *path.parents) if item != root), "symlink evidence is unsupported")
    return path


class Files:
    def __init__(self):
        self.seen = {}

    @staticmethod
    def identity(path):
        value = path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def hash(self, path, expected=None, size=None):
        path = Path(path).resolve()
        identity = self.identity(path)
        if path not in self.seen:
            self.seen[path] = (identity, digest(path))
        previous, actual = self.seen[path]
        require(identity == previous, "evidence changed during verification")
        require(expected is None or actual == expected, "evidence SHA-256 mismatch")
        require(size is None or identity[0] == size, "evidence byte-size mismatch")
        return actual

    def read(self, path):
        self.hash(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def inventory(self, root, records, names):
        require(set(records) == names, "artifact inventory differs")
        return {name: self.hash(relative(root, name), records[name]["sha256"], count(records[name]["bytes"]))
                for name in sorted(names)}

    def finish(self):
        require(all(self.identity(path) == record[0] for path, record in self.seen.items()),
                "evidence changed during verification")


def check_period(summary, users, hits, hits100, reciprocal):
    n = len(users)
    require(summary["queries"] == n, "query denominator differs")
    if not n:
        require(summary.get("status") == "no_queries" and summary["methods"] == {}
                and summary["comparisons"] == {}, "empty day contains measurements")
        return
    unique_users = len(np.unique(users))
    require(summary["users"] == unique_users and set(summary["methods"]) == set(METHODS), "method/user inventory differs")
    require(set(summary["comparisons"]) == {row[0] for row in COMPARISONS}, "comparison inventory differs")
    for index, method in enumerate(METHODS):
        item = summary["methods"][method]
        require(count(item["denominator"]) == n and count(item["hits10"], n) == int(hits[:, index].sum())
                and count(item["hits100"], n) == int(hits100[:, index].sum()), "method counts differ from saved predictions")
        equal(item["hit_rate10"], float(hits[:, index].mean()))
        equal(item["hit_rate100"], float(hits100[:, index].mean()))
        equal(item["mrr10"], float(reciprocal[:, index].mean()))
    for key, left, right in COMPARISONS:
        item = summary["comparisons"][key]
        equal(item["difference"], float(hits[:, left].sum(dtype=np.int64) - hits[:, right].sum(dtype=np.int64)) / n)
        bounds = item["ci95"]
        require(len(bounds) == 2 and -1 <= finite(bounds[0]) <= finite(bounds[1]) <= 1,
                "invalid stored confidence interval")
        require(item["user_clusters"] == unique_users and item["bootstrap_repetitions"] == 1000
                and item["bootstrap_rng_seed"] == 2026 and item["valid_repetitions"] == 1000,
                "stored user-cluster bootstrap configuration differs")


def verify(run_dir, reports, *, repo=None):
    # Explicitly release read-only mappings on failure as well as success,
    # including when called in-process on Windows.
    with ExitStack() as handles:
        return _verify(run_dir, reports, repo=repo, handles=handles)


def _verify(run_dir, reports, *, repo, handles):
    run_dir, reports = Path(run_dir).resolve(), Path(reports).resolve()
    repo = Path(__file__).resolve().parents[1] if repo is None else Path(repo).resolve()
    files = Files()
    receipts, hashes = {}, {}
    for stage in ("acquisition", "prepare", "evaluate", "aggregate"):
        require(not (run_dir / stage / "failure.json").exists(), "completed run retains a failed stage marker")
        path = relative(run_dir, f"{stage}/receipt.json")
        receipts[stage] = files.read(path)
        require(receipts[stage].get("status") == "passed" and receipts[stage].get("schema_version") == 1,
                "phase has not completed successfully")
        hashes[f"{stage}_receipt"] = files.hash(path)
    acquired, prepared, evaluated, aggregated = (receipts[name] for name in ("acquisition", "prepare", "evaluate", "aggregate"))
    report = files.read(relative(reports, "evaluation.json"))
    audit = files.read(relative(reports, "audit.json"))
    hashes["evaluation_report"] = files.hash(reports / "evaluation.json", aggregated["public_evaluation_sha256"])
    hashes["audit_report"] = files.hash(reports / "audit.json", aggregated["public_audit_sha256"])
    require(report.get("status") == "passed" and report.get("schema_version") == 1
            and aggregated.get("all_prediction_counts_recomputed") is True, "aggregate did not complete recomputation")
    require(prepared["acquisition_receipt_sha256"] == hashes["acquisition_receipt"]
            and evaluated["prepare_receipt_sha256"] == hashes["prepare_receipt"], "phase receipt chain differs")
    require(report["phase_receipts_sha256"] == {name: hashes[f"{name}_receipt"] for name in
            ("acquisition", "prepare", "evaluate")}, "public phase receipt chain differs")
    if "ranges_manifest_sha256" in acquired:
        hashes["acquisition_ranges"] = files.hash(relative(run_dir, "acquisition/ranges.json"), acquired["ranges_manifest_sha256"])
    # Original acquisition can predate implementation and the explicit resource amendment.
    bindings = prepared["bindings"]
    require(all(key == "implementation" or bindings.get(key) == value for key, value in acquired["bindings"].items()),
            "acquisition scientific bindings differ")
    require(bindings == evaluated["bindings"] == aggregated["bindings"] == report["bindings"], "source binding chain differs")
    source_path = relative(repo, "reports/external/adressa_source.json")
    hashes["source_report"] = files.hash(source_path, bindings["source_report_sha256"])
    source = files.read(source_path)
    source_files = source["frozen_mind_and_protocol_files"]
    require({name: item["sha256"] for name, item in source_files.items()} == bindings["frozen_files"], "frozen source inventory differs")
    for name, item in source_files.items():
        files.hash(relative(repo, name), item["sha256"], item["bytes"])
    require(bool(bindings["implementation"]) and set(bindings["implementation"]) == {
        path.relative_to(repo).as_posix() for path in (repo / "newsforge/external").glob("*.py")}, "implementation inventory differs")
    for name, expected in bindings["implementation"].items():
        files.hash(relative(repo, name), expected)
    hashes["dependency_lock"] = files.hash(relative(repo, "requirements-delivery-windows.lock"), bindings["dependency_lock_sha256"])
    if "resource_amendment_sha256" in bindings:
        hashes["resource_amendment"] = files.hash(relative(repo, "docs/EXTERNAL_ADRESSA_RESOURCE_AMENDMENT.md"), bindings["resource_amendment_sha256"])
    hashes["frozen_inventory"] = canonical(bindings["frozen_files"])
    hashes["implementation_inventory"] = canonical(bindings["implementation"])
    hashes["prepared_files"] = files.inventory(run_dir / "prepare", prepared["files"], PREPARED)
    hashes["evaluated_files"] = files.inventory(run_dir / "evaluate", evaluated["files"], EVALUATED)
    require(report["per_query_files"] == evaluated["files"], "public prediction inventory differs")
    require(acquired["sha256"] == prepared["archive_sha256"] == report["archive_sha256"], "archive identity chain differs")
    require(isinstance(acquired["sha256"], str) and len(acquired["sha256"]) == 64
            and set(acquired["sha256"]) <= set("0123456789abcdef"), "invalid archive digest")
    hashes["archive_identity_from_receipts"] = acquired["sha256"]
    require(prepared["sample_identity_sha256"] == evaluated["sample_identity_sha256"] == report["sample_identity_sha256"],
            "sample identity chain differs")
    require(isinstance(prepared["sample_identity_sha256"], str) and len(prepared["sample_identity_sha256"]) == 64
            and set(prepared["sample_identity_sha256"]) <= set("0123456789abcdef"), "invalid sample digest")
    hashes["sample_identity_from_receipts"] = prepared["sample_identity_sha256"]
    require(prepared.get("target_independent_sampling") is True and prepared.get("external_model_fitting") is False,
            "preparation contract differs")
    require(all(report.get(key) is False for key in ("external_model_fitting", "external_hyperparameter_selection",
            "prospective_future_generalization", "online_ab_test")) and report["delta"] == 10 and report["seeds"] == [17, 29, 43],
            "scientific protocol flags differ")

    def load(path):
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        handles.callback(value._mmap.close)
        return value

    arrays = {name: load(run_dir / "evaluate" / f"{name}.npy")
              for name in ("queries", "pools", "sizes", "orders", "scores", "hits10", "hits100", "reciprocal10")}
    queries, pools, sizes, orders, scores, hits, hits100, reciprocal = arrays.values()
    n = len(queries)
    require(128 <= n <= 15000 and queries.shape == (n, 4) and np.issubdtype(queries.dtype, np.integer)
            and len(np.unique(queries[:, 0])) == n and np.all(queries[:, 2:] > 0), "invalid fixed query array")
    require(np.all((queries[:, 1] >= EVAL_START) & (queries[:, 1] < EVAL_START + 3 * 86400))
            and np.all(np.diff(queries[:, 1]) >= 0), "queries are outside the fixed dates/order")
    require(pools.shape == (n, 100) and sizes.shape == (n,) and orders.shape == scores.shape == (n, 5, 100)
            and hits.shape == hits100.shape == reciprocal.shape == (n, 5), "prediction array shape differs")
    require(all(np.issubdtype(value.dtype, np.integer) for value in (pools, sizes, orders, hits, hits100))
            and np.all((sizes >= 0) & (sizes <= 100)), "invalid candidate count or dtype")
    require(np.isin(hits, [0, 1]).all() and np.isin(hits100, [0, 1]).all() and np.all(hits <= hits100)
            and np.all(hits100 == hits100[:, :1]) and np.isfinite(reciprocal).all(), "hit100/binary-hit invariant failed")
    coverage = load(run_dir / "prepare/coverage.npy")
    articles = count(prepared["identity"]["articles"])
    require(coverage.shape == (articles + 1, 4) and np.isin(coverage, [0, 1]).all(), "invalid article coverage flags")
    candidate_coverage = np.zeros(4, dtype=np.int64)
    for row in range(n):
        m = int(sizes[row])
        pool = pools[row, :m]
        require(np.all((pool > 0) & (pool <= articles)) and len(np.unique(pool)) == m, "invalid candidate pool")
        require(np.isfinite(scores[row, :, :m]).all(), "nonfinite valid candidate score")
        candidate_coverage += coverage[pool].sum(axis=0, dtype=np.int64)
        for method in range(5):
            positions = orders[row, method, :m]
            require(np.array_equal(np.sort(positions), np.arange(m)), "ranking changed candidate membership")
            require(np.array_equal(positions, np.lexsort((np.arange(m), -scores[row, method, :m]))), "saved score/order mismatch")
            locations = np.flatnonzero(pool[positions] == queries[row, 3])
            hit = bool(len(locations) and locations[0] < 10)
            rr = 1. / (int(locations[0]) + 1) if hit else 0.
            require(hits[row, method] == int(hit) and hits100[row, method] == int(bool(len(locations)))
                    and reciprocal[row, method] == rr, "saved hit or reciprocal rank differs")
    require(evaluated["queries"] == n and evaluated["methods"] == list(METHODS), "evaluation query/method inventory differs")
    require(report["audit"] == audit == files.read(run_dir / "evaluate/audit.json"), "audit copies differ")
    indices_sha = hashlib.sha256(np.linspace(0, n - 1, 128, dtype=np.int64).tobytes()).hexdigest()
    require(audit["status"] == "passed" and audit["queries"] == 128 and audit["indices_sha256"] == indices_sha
            and all(audit[key] == "exact" for key in ("candidate_id_equality", "ranking_equality", "hit_equality"))
            and audit["score_atol"] == 1e-8 and audit["score_rtol"] == 0
            and 0 <= finite(audit["score_max_absolute_error"]) <= 1e-8
            and audit["all_method_per_query_candidate_set_invariant"] is True
            and audit["all_method_per_query_hit100_invariant"] is True, "128-query reference audit contract differs")
    check_period(report["overall"], queries[:, 2], hits, hits100, reciprocal)
    require(set(report["by_day"]) == set(DAYS), "fixed evaluation dates differ")
    require(report["daily_samples"] == prepared["daily_samples"]
            and report["identity_counts"] == prepared["identity"] and report["source_counts"] == prepared["counts"],
            "preparation/public counts differ")
    for index, day in enumerate(DAYS):
        mask = (queries[:, 1] >= EVAL_START + index * 86400) & (queries[:, 1] < EVAL_START + (index + 1) * 86400)
        daily_n = int(mask.sum())
        require(daily_n == prepared["daily_samples"].get(day, 0)
                == min(count(prepared["identity"]["daily_populations"][day]), 5000), "daily sampling denominator differs")
        check_period(report["by_day"][day], queries[mask, 2], hits[mask], hits100[mask], reciprocal[mask])
    require(sum(prepared["daily_samples"].values()) == n, "daily counts do not sum to all requests")
    totals = prepared["counts"]
    require(count(totals["valid_before_dedup"]) == count(totals["deduplicated_events"]) + count(totals["duplicate_events_removed"])
            and totals["valid_before_dedup"] <= count(totals["raw_lines"]), "source event count conservation failed")
    diagnostics = evaluated["diagnostics"]
    require(report["query_diagnostics"] == diagnostics, "query diagnostics differ")
    candidate_items = count(diagnostics["candidate_items"], 100 * n)
    history_items = count(diagnostics["history_items"], 20 * n)
    require(candidate_items == int(sizes.sum()) and candidate_items + count(diagnostics["shortfall"]) == 100 * n
            and count(diagnostics["queries_with_shortfall"], n) == int((sizes < 100).sum()), "candidate/shortfall count conservation failed")
    for name in ("empty_history", "unavailable_target", "already_read_target", "queries_with_fallback"):
        count(diagnostics[name], n)
    count(diagnostics["fallback_items"], candidate_items)
    article_coverage = prepared["transformed"]["coverage"]
    require(report["article_text_coverage"] == article_coverage and article_coverage["articles"] == articles,
            "article coverage denominator differs")
    for index, name in enumerate(COVERAGE):
        require(count(article_coverage[name], articles) == int(coverage[1:, index].sum())
                and count(diagnostics[f"candidate_{name}"], candidate_items) == int(candidate_coverage[index]),
                "coverage counts differ from retained flags")
        count(diagnostics[f"history_{name}"], history_items)
    primary = report["overall"]["comparisons"][COMPARISONS[0][0]]
    supported = primary["difference"] > 0 and primary["ci95"][0] > 0
    require(report["positive_transfer_supported_on_preregistered_sample"] is supported
            and aggregated["positive_transfer_supported"] is supported, "transfer conclusion differs from the overall primary comparison")
    hashes["script"] = files.hash(__file__)
    files.finish()
    return {"schema_version": 1, "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Supplemental verification of existing hashes, arrays and aggregates; not an independent raw-data reconstruction or new statistical experiment.",
        "bootstrap_recomputed": False, "model_inference_repeated": False, "original_archive_rehashed": False,
        "queries_checked": n, "stored_reference_queries": 128, "verified_file_count": len(files.seen),
        "positive_transfer_supported": supported, "resource_amendment_bound": "resource_amendment" in hashes,
        "checks": {name: True for name in ("source_and_receipt_hash_chain", "artifact_hashes", "fixed_all_request_denominators",
            "method_counts_and_ablation_directions", "stored_ci_bounds_and_bootstrap_configuration", "reference_index_identity",
            "candidate_permutations_and_hit100_identity", "finite_valid_scores", "saved_hit_and_rr_recalculation",
            "daily_count_conservation", "diagnostic_count_bounds", "article_and_candidate_coverage_counts", "overall_conclusion_rule")},
        "input_sha256": hashes,
        "limits": ["Stored CIs are checked for configuration, finite bounds and point-estimate direction, not resampled again.",
                   "The saved 128-query reference audit is hash-checked, not rerun. History coverage is bounded by its reported occurrence count; user histories are not reconstructed.",
                   "Archive and sample identities are linked through existing receipts. Original compressed bytes and target-independent sampling are not recomputed here.",
                   "A resource amendment, when bound, occurred after partial target-event access; the original scientific plan remained fixed. Verification success does not establish positive transfer."]}


def publish(run_dir, reports, output, *, repo=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("supplemental verification output already exists")
    result = verify(run_dir, reports, repo=repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = publish(args.run_dir, args.reports, args.output)
    print(json.dumps({"status": result["status"], "queries_checked": result["queries_checked"]}))


if __name__ == "__main__":
    main()
