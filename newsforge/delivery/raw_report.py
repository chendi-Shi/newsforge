"""Publish an aggregate-only summary of an accepted raw reproduction.

This is a lightweight follow-up to raw_acceptance, not a second full array or
raw-TSV audit. It revalidates the initialized source and complete stage chain,
then reads only acceptance-bound reports and test logs. It never trains models.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from newsforge.delivery import acceptance as shared
from newsforge.delivery import pipeline as p
from newsforge.delivery import raw_acceptance as a


METRICS = ("auc", "mrr", "ndcg5", "ndcg10")
V1_ARMS = ("popularity", "content", *(f"{v}-seed{s}" for v in
           ("mean_random", "mean_mixed", "ranker") for s in p.SEEDS))
V2_ARMS = tuple(f"{v}-seed{s}" for v in ("content_only", "context_only", "full") for s in p.SEEDS)
CATALOG_ARMS = ("full", "no_popular", "no_content", "no_tower", "popular_only", "content_only",
                "tower_only", "no_freshness", "frozen_feedback", "selected",
                "selected_rerank_content_only", "selected_rerank_context_only", "selected_rerank_full")
V3_ARMS = ("baseline", "full13", "no_category", "no_recent", "no_gate",
           *(f"full13_seed{s}" for s in p.SEEDS))
LIMITS = [
    "The acceptance receipt previously rehashed raw TSVs, model/cache arrays and independent auditor bindings. This summary rehashes initialized sources, stage receipts, declared outputs, logs and summarized reports; it does not repeat the full raw/array audit or numerical predictions.",
    "MIND dev has already been observed. This run establishes neither unseen external validation nor a production deployment, distributed benchmark or online A/B test.",
    "Full-suite and research-only test runs overlap. Passed counts exclude skipped tests and must not be added as unique coverage.",
    "v1 paired intervals compare per-impression predictions averaged over seeds, not a score ensemble. v3 dev ablations are exploratory, with no multiple-comparison adjustment; removing explicit columns does not erase the same information from other features.",
    "The frozen confirm gate determines the serving recipe. Historical equality and dev ablation results do not select a new recipe.",
    "HTTP throughput and request latency describe this shared desktop run. Retrieval microbenchmarks are not HTTP throughput; cold means derived-cache cold, not operating-system cache cold. Observed speed ratios are not confidence intervals.",
    "Only aggregate values and content hashes are published. Source data, user identities, article text, model bytes, raw request/response logs and local absolute paths are excluded.",
]


def number(value):
    a.require(type(value) in (int, float) and math.isfinite(value), "expected a finite numeric aggregate")
    return value


def numbers(value, keys):
    """Fixed field allowlist: never recursively copy arbitrary report content."""
    return {key: number(value[key]) for key in keys if key in value}


def sha(value):
    a.require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
              "invalid evidence SHA-256")
    return value


def interval(value):
    result = numbers(value, ("delta", "replicates", "seed"))
    a.require("delta" in result and len(value.get("ci95", [])) == 2, "incomplete confidence interval")
    result["ci95"] = [number(x) for x in value["ci95"]]
    a.require(result["ci95"][0] <= result["ci95"][1], "reversed confidence interval")
    if "resampling_unit" in value:
        a.require(value["resampling_unit"] == "user", "unexpected bootstrap unit")
        result["resampling_unit"] = "user"
    return result


def ranking_metrics(value):
    a.require(set(METRICS) <= value.keys() and all("mean" in value[key] for key in METRICS),
              "incomplete ranking metrics")
    return {metric: numbers(value[metric], ("mean", "eligible_impressions", "std", "sample_std"))
            for metric in METRICS if metric in value}


def ranking_intervals(value, names):
    return {name: {metric: interval(value[name][metric]) for metric in METRICS if metric in value[name]}
            for name in names if name in value}


def v3_metrics(value):
    result = {}
    for arm in V3_ARMS:
        if arm not in value:
            continue
        metrics = numbers(value[arm], ("hits10", "recall10", "hits100", "recall100"))
        a.require(len(metrics) == 4, "incomplete v3 metrics")
        for other in ("baseline", "no_category", "no_recent", "no_gate"):
            key = f"versus_{other}"
            if key in value[arm]:
                metrics[key] = interval(value[arm][key])
        result[arm] = metrics
    a.require({"baseline", "full13"} <= result.keys(), "v3 baseline or full model absent")
    return result


def phase_summary(report, phase):
    a.require(report["phase"] == phase and type(report.get("deployment_gate_passed")) is bool,
              "phase or confirm gate is missing")
    result = numbers(report, ("all_clicks", "eligible_clicks", "seen_clicks", "shortfall_requests", "delta"))
    result.update(rows=number(report["bindings"]["rows"]["count"]), metrics=v3_metrics(report["metrics"]),
                  deployment_gate_passed=report["deployment_gate_passed"])
    a.require(report["invariants"].get("per_request_candidate_and_hit100_invariant") is True,
              "candidate/hit100 invariance was not established")
    result["candidate_and_hit100_invariant"] = True
    if phase == "dev":
        recipe = report["deployed_recipe"]
        a.require(recipe == ("full13+gate" if result["deployment_gate_passed"] else "baseline"),
                  "deployed recipe disagrees with confirm gate")
        result["deployed_recipe"] = recipe
    return result


def verify_controls(workspace, acceptance_path):
    """Recheck receipt chain without a second scan of raw TSVs/cache arrays."""
    acceptance_path = Path(acceptance_path)
    identity, receipt_sha = a.Evidence.identity(acceptance_path), p.digest(acceptance_path)
    receipt = p.read(acceptance_path)
    a.require(receipt.get("schema_version") == 1 and receipt.get("passed") is True,
              "a passed raw acceptance receipt is required")
    a.require(receipt.get("verifier_source_sha256") == p.digest(a.__file__),
              "raw acceptance verifier source changed")
    evidence = a.Evidence(workspace)
    manifest = evidence.read(p.STATE / "workspace.json")
    a.require(manifest.get("schema_version") == p.SCHEMA
              and evidence.digest(p.STATE / "workspace.json") == receipt["workspace_manifest_sha256"],
              "accepted workspace manifest differs")
    a.require(manifest["plan_sha256"] == receipt["plan_sha256"] == p.canonical_hash(p.make_plan()),
              "accepted pipeline plan differs")
    a.require(set(manifest["source_files"]) == set(p.source_files(evidence.root)),
              "workspace source inventory differs")
    for name, expected in manifest["source_files"].items():
        evidence.digest(name, sha(expected))
    for module, name in ((p, "newsforge/delivery/pipeline.py"), (shared, "newsforge/delivery/acceptance.py")):
        a.require(p.digest(module.__file__) == manifest["source_files"][name],
                  "summarizer and executed pipeline helpers differ")
    first = p.ordered_stages()[0].name
    python = evidence.read(p.STATE / first / "started.json")["python"]
    chain = a.receipt_chain(evidence, manifest, python=python)
    a.require(chain == receipt["completed_stages"], "stage chain differs from accepted execution")
    # All lightweight reverified evidence must also belong to the accepted run.
    for name, digest in evidence.hashes.items():
        a.require(receipt["verified_sha256_files"].get(name) == digest,
                  f"reverified file absent from acceptance binding: {name}")
    a.require(a.Evidence.identity(acceptance_path) == identity and p.digest(acceptance_path) == receipt_sha,
              "acceptance receipt changed while reading")
    return evidence, manifest, receipt, receipt_sha, identity


def build_summary(workspace, acceptance_path):
    evidence, manifest, receipt, receipt_sha, receipt_identity = verify_controls(workspace, acceptance_path)

    def read(name):
        a.require(name in receipt["verified_sha256_files"], f"report absent from acceptance binding: {name}")
        evidence.digest(name, receipt["verified_sha256_files"][name])
        return evidence.read(name)

    for version in ("v2", "v3"):
        name = f"reports/{version}/release_integrity.json"
        report = read(name)
        accepted = receipt["release_audits"][version]
        a.require(report.get("passed") is True and accepted.get("passed") is True
                  and evidence.digest(name) == accepted["report_sha256"]
                  and report["checks"] == accepted["checks"], "release audit identity differs")

    processed = read("data/processed/manifest.json")
    data = numbers(processed, ("rows", "impressions", "candidates", "positives", "users", "news",
                              "news_including_padding", "official_train_rows", "official_dev_rows",
                              "processing_seconds"))
    a.require({"rows", "impressions", "candidates", "users", "news"} <= data.keys(), "data volumes absent")
    data["splits"] = {key: numbers(processed["splits"][key],
        ("impressions", "candidates", "positives", "users", "candidate_news", "positive_rate"))
        for key in ("0", "1", "2")}
    data["features"] = numbers(processed["features"], ("dimensions", "text_dimensions", "svd_actual_components",
        "tfidf_vocabulary_size", "tfidf_max_features", "fit_news", "svd_explained_variance_ratio_sum"))
    raw = {}
    a.require(set(receipt["raw_files"]) == a.RAW_NAMES, "accepted raw file inventory differs")
    for name in sorted(a.RAW_NAMES):
        item = receipt["raw_files"][name]
        digest = sha(item["sha256"])
        a.require(receipt["verified_sha256_files"][f"data/raw/{name}"] == digest
                  and (manifest["raw_mode"] != "provided" or manifest["raw_files"][name] == digest),
                  "raw input binding differs")
        raw[name] = {"sha256": digest, "bytes": number(item["bytes"])}
    lock_sha = evidence.digest(a.LOCK, receipt["environment"]["lock_sha256"])
    a.require(receipt["environment"].get("passed") is True, "installation acceptance missing")
    environment = {key: sha(receipt["environment"][key]) for key in
                   ("lock_sha256", "wheel_receipt_sha256", "install_log_sha256", "installation_receipt_sha256")}
    environment["distributions"] = shared.parse_lock(evidence.path(a.LOCK))
    environment["locked_distributions"] = len(environment["distributions"])
    a.require(environment["locked_distributions"] == receipt["environment"]["locked_distributions"],
              "dependency count differs")

    tests = {}
    for stage, name in a.TEST_LOGS.items():
        evidence.digest(name, receipt["verified_sha256_files"][name])
        parsed = shared.parse_test_summary(evidence.path(name).read_text(encoding="utf-8-sig"))
        tests[stage] = {"total": parsed["count"], "passed": parsed["executed_count"],
                        "skipped": parsed["skipped"], "log_sha256": evidence.digest(name)}
    a.require(tests["research-unit-tests"]["skipped"] == 0, "research tests contain skips")

    ranking = read("reports/ranking_experiment.json")
    a.require(set(V1_ARMS) <= ranking["runs"].keys()
              and {"mean_random", "mean_mixed", "ranker"} <= ranking["seed_summary"].keys()
              and {"mean_mixed", "ranker"} <= ranking["paired_user_bootstrap"].keys(),
              "incomplete v1 experiment arms")
    v1 = {"runs": {arm: {**numbers(ranking["runs"][arm], ("impressions", "candidate_exposures")),
                             "metrics": ranking_metrics(ranking["runs"][arm]["metrics"])}
                    for arm in V1_ARMS if arm in ranking["runs"]},
          "seed_summary": {arm: ranking_metrics(ranking["seed_summary"][arm]) for arm in
                           ("mean_random", "mean_mixed", "ranker") if arm in ranking["seed_summary"]},
          "paired_user_bootstrap": ranking_intervals(ranking["paired_user_bootstrap"], ("mean_mixed", "ranker"))}
    v2 = {}
    interval_names = tuple(f"{left}_minus_{right}" for left, right in
        (("full", "content_only"), ("full", "context_only"), ("context_only", "content_only")))
    for phase in ("ranking", "ranking-full"):
        source = read(f"reports/v2/{phase}.json")
        a.require(bool(source["metrics"]) and set(source["metrics"]) <= set(V2_ARMS)
                  and (phase != "ranking" or set(source["metrics"]) == set(V2_ARMS)),
                  "incomplete or unknown v2 ranking arms")
        v2[phase] = {**numbers(source, ("impressions", "candidates")),
                    "metrics": {arm: ranking_metrics(source["metrics"][arm]) for arm in V2_ARMS if arm in source["metrics"]},
                    "seed_statistics": {arm: ranking_metrics(source.get("seed_statistics", {})[arm]) for arm in
                        ("content_only", "context_only", "full") if arm in source.get("seed_statistics", {})},
                    "paired_user_intervals": ranking_intervals(source.get("paired_user_intervals", {}), interval_names)}
    catalog = read("reports/v2/catalog.json")
    a.require(set(CATALOG_ARMS) <= catalog["metrics"].keys(), "incomplete v2 catalog ablations")
    comparisons = [f"full_minus_{arm}_recall100" for arm in CATALOG_ARMS if arm != "full"]
    comparisons += [f"selected_rerank_{arm}_minus_no_rerank_recall10" for arm in ("content_only", "context_only", "full")]
    v2["catalog"] = {**numbers(catalog, ("queries", "positive_clicked_items", "eligible_positive_items")),
        "metrics": {arm: numbers(catalog["metrics"][arm], ("click_micro_recall10", "click_micro_recall100",
            "eligible_click_micro_recall100", "cold_eligible_recall100", "zero_train_click_eligible_recall100",
            "hits10", "hits100", "shortfall_queries", "empty_history_queries"))
                    for arm in CATALOG_ARMS if arm in catalog["metrics"]},
        "paired_user_intervals": {name: interval(catalog["paired_user_intervals"][name]) for name in comparisons
                                   if name in catalog["paired_user_intervals"]}}

    fit, tune = read("reports/v3/fit.json"), read("reports/v3/tune.json")
    v3 = {"fit_rows": number(fit["fit_rows"]["count"]), "fit": {},
          "selected_delta": number(tune["selected_delta"]), "tune_delta_grid": {}}
    for seed in map(str, p.SEEDS):
        run = fit["runs"][seed]
        v3["fit"][seed] = {"sampling": numbers(run["sampling"], ("requests", "pairs", "negative_draws_per_positive",
            "seed", "skipped_no_positive", "skipped_no_negative")),
            "models": {arm: numbers(run["models"][arm]["fit_metadata"],
                ("pairs", "l2", "max_iterations", "optimizer_status", "iterations", "function_evaluations", "loss",
                 "gradient_max_absolute")) for arm in ("full13", "no_category", "no_recent")}}
    for delta in ("0", "2", "5", "10"):
        v3["tune_delta_grid"][delta] = v3_metrics(tune["delta_grid"][delta]["metrics"])
    comparison = {}
    for phase in ("confirm", "dev"):
        fresh = read(f"reports/v3/{phase}.json")
        previous = read(f"reports-delivered/v3/{phase}.json")
        accepted = receipt["comparison"]["phases"][phase]
        a.require(a.scientific_summary(fresh) == accepted["fresh"]
                  and a.scientific_summary(previous) == accepted["previous"]
                  and evidence.digest(f"reports/v3/{phase}.json") == accepted["fresh_report_sha256"]
                  and evidence.digest(f"reports-delivered/v3/{phase}.json") == accepted["reference_report_sha256"],
                  "fresh/historical comparison differs from acceptance")
        a.require(accepted["scientific_summary_exactly_equal"]
                  == (a.scientific_summary(fresh) == a.scientific_summary(previous)),
                  "historical equality flag differs from reports")
        v3[phase] = phase_summary(fresh, phase)
        old = phase_summary(previous, phase)
        comparison[phase] = {"fresh_report_sha256": accepted["fresh_report_sha256"],
            "reference_report_sha256": accepted["reference_report_sha256"],
            "scientific_summary_exactly_equal": a.scientific_summary(fresh) == a.scientific_summary(previous),
            "previous": old,
            "fresh_minus_previous": {arm: {metric: v3[phase]["metrics"][arm][metric] - old["metrics"][arm][metric]
                for metric in ("recall10", "recall100")} for arm in v3[phase]["metrics"] if arm in old["metrics"]}}
    a.require(v3["confirm"]["deployment_gate_passed"] == v3["dev"]["deployment_gate_passed"],
              "confirm and dev gate disagree")

    service = read("reports/v3/service_benchmark.json")
    a.require(service.get("passed") is True and receipt["http"].get("passed") is True,
              "HTTP acceptance did not pass")
    http = {"configurations": [], "protocol": numbers(service["protocol"], ("cpu_threads", "server_workers")),
            "resources": numbers(service["resources"], ("server_peak_rss_bytes", "minimum_system_available_bytes",
                                                         "resource_samples", "sample_interval_seconds"))}
    for case in service["configurations"]:
        row = numbers(case, ("batch_size", "concurrency", "rounds", "warmup_http_requests", "measured_http_requests",
             "measured_queries", "wall_seconds", "successful_queries_per_second", "http_errors", "failed_queries",
             "offline_http_mismatches", "queries_with_output_shortfall", "queries_with_retrieval_shortfall"))
        row["latency_ms"] = numbers(case["latency_ms"], ("p50", "p95", "p99", "max"))
        a.require(all(row[key] == 0 for key in ("http_errors", "failed_queries", "offline_http_mismatches")),
                  "HTTP failures or mismatches present")
        http["configurations"].append(row)
    http["measured_queries"] = sum(row["measured_queries"] for row in http["configurations"])
    a.require(http["measured_queries"] == receipt["http"]["measured_queries"]
              and len(http["configurations"]) == receipt["http"]["configurations"], "HTTP aggregate differs")
    http["api_contracts_passed"] = sum(item["passed"] is True for item in service["api_contract_checks"])
    a.require(http["api_contracts_passed"] == len(service["api_contract_checks"]) == receipt["http"]["api_contracts"],
              "HTTP contract check count differs")
    http["latency_scope"] = "One HTTP request, including every query in its batch; client serialization, connection, server work and response decoding. Warmup excluded."
    http["resource_scope"] = "One owned HTTP server process, excluding the benchmark client and startup; sampled during API checks and all HTTP cases."
    fast = read("reports/v3/fast_retrieval_benchmark.json")
    a.require(fast["equivalence"].get("exact") is True, "retrieval exact equivalence absent")
    micro = {"equivalence": {"exact": True, **numbers(fast["equivalence"],
             ("candidate_batches_compared", "mismatch_field_occurrences", "numeric_tolerance"))}, "summary": {}}
    for mode in ("cold", "warm"):
        row = fast["summary"][mode]
        micro["summary"][mode] = numbers(row, ("observed_wall_speedup_v2_over_v3",
                                               "observed_instrumented_query_cpu_ratio_v2_over_v3"))
        for implementation in ("v2", "v3_fast"):
            value = row[implementation]
            micro["summary"][mode][implementation] = {**numbers(value, ("measured_queries", "measured_query_wall_seconds",
                "measured_process_cpu_seconds", "measured_queries_per_second", "maximum_derived_cache_array_bytes",
                "maximum_query_boundary_rss_bytes")), "latency_ms": numbers(value["latency_ms"], ("p50", "p95", "p99"))}

    evidence.finish()
    a.require(a.Evidence.identity(Path(acceptance_path)) == receipt_identity
              and p.digest(acceptance_path) == receipt_sha, "acceptance receipt changed during summarization")
    result = {"schema_version": 1, "passed": True, "created_utc": p.utc(),
        "acceptance_sha256": receipt_sha, "acceptance_created_utc": receipt["created_utc"],
        "summarizer_source_sha256": p.digest(__file__), "workspace_manifest_sha256": receipt["workspace_manifest_sha256"],
        "plan_sha256": manifest["plan_sha256"], "environment": environment, "raw_files": raw, "data": data,
        "stages": [{**item, "status": "success"} for item in receipt["completed_stages"]],
        "tests": tests, "v1": v1, "v2": v2, "v3": v3, "comparison": comparison,
        "http": http, "retrieval_microbenchmark": micro,
        "reverified_sha256_files": dict(sorted(evidence.hashes.items())), "limits": LIMITS}
    # Defensive final check, in addition to the fixed aggregate allowlists above.
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
    a.require(re.search(r"[A-Za-z]:[\\/]|\\\\|/(?:home|Users|tmp|private)/", serialized) is None,
              "public result contains a local absolute path")
    return result


def markdown(result):
    lines = ["# Raw reproduction results", "", f"Acceptance: **passed**. Receipt SHA-256: `{result['acceptance_sha256']}`.",
             "", "Fresh metrics and stages come from this accepted execution; historical comparisons are explicitly labeled. Full precision and source bindings are in [execution_summary.json](execution_summary.json).", "",
             f"Confirm gate: **{result['v3']['confirm']['deployment_gate_passed']}**; serving recipe: **{result['v3']['dev']['deployed_recipe']}**.",
             "", "## Data and tests", "", "| Aggregate | Value |", "| --- | ---: |"]
    lines += [f"| {key} | {value:,} |" for key, value in result["data"].items() if type(value) in (int, float)]
    lines += ["", "| Test run (overlapping coverage) | Passed | Skipped | Total |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {name} | {row['passed']} | {row['skipped']} | {row['total']} |" for name, row in result["tests"].items()]

    def metric_table(title, rows):
        lines.extend(["", f"## {title}", "", "| Arm | AUC | MRR | nDCG@5 | nDCG@10 |", "| --- | ---: | ---: | ---: | ---: |"])
        for name, metrics in rows.items():
            values = [f"{metrics[m]['mean']:.6f}" if m in metrics else "—" for m in METRICS]
            lines.append("| " + " | ".join([name, *values]) + " |")

    metric_table("v1 official-dev ranking", {key: row["metrics"] for key, row in result["v1"]["runs"].items()})

    def interval_table(title, rows):
        lines.extend(["", f"### {title}", "", "| Comparison / metric | Delta | 95% CI |",
                      "| --- | ---: | --- |"])
        for name, value in rows:
            lines.append(f"| {name} | {value['delta']:+.6f} | [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}] |")

    interval_table("v1 paired user bootstrap versus mean_random",
        [(f"{arm} / {metric}", value) for arm, metrics in result["v1"]["paired_user_bootstrap"].items()
         for metric, value in metrics.items()])
    metric_table("v2 sampled ranking, seed means", result["v2"]["ranking"]["seed_statistics"])
    interval_table("v2 ranking ablation intervals",
        [(f"{arm} / {metric}", value) for arm, metrics in result["v2"]["ranking"]["paired_user_intervals"].items()
         for metric, value in metrics.items()])
    metric_table("v2 full ranking, internally selected model", result["v2"]["ranking-full"]["metrics"])
    lines += ["", "## v2 catalog ablations", "", "| Arm | Recall@10 | Recall@100 |", "| --- | ---: | ---: |"]
    lines += [f"| {arm} | {value['click_micro_recall10']:.6f} | {value['click_micro_recall100']:.6f} |"
              for arm, value in result["v2"]["catalog"]["metrics"].items()]
    interval_table("v2 catalog ablation intervals", list(result["v2"]["catalog"]["paired_user_intervals"].items()))
    lines += ["", "## v3 confirmation and dev ablations", "", "| Phase / arm | Recall@10 | Recall@100 | ΔR@10 vs baseline [95% CI] |", "| --- | ---: | ---: | --- |"]
    for phase in ("confirm", "dev"):
        for arm, value in result["v3"][phase]["metrics"].items():
            ci = value.get("versus_baseline")
            comparison = f"{ci['delta']:+.6f} [{ci['ci95'][0]:+.6f}, {ci['ci95'][1]:+.6f}]" if ci else "—"
            lines.append(f"| {phase} / {arm} | {value['recall10']:.6f} | {value['recall100']:.6f} | {comparison} |")
    interval_table("v3 dev full13 versus explicit ablations (exploratory)",
        [(other, result["v3"]["dev"]["metrics"]["full13"][other])
         for other in ("versus_no_category", "versus_no_recent", "versus_no_gate")
         if other in result["v3"]["dev"]["metrics"]["full13"]])
    lines += ["", "## Fresh versus previous scientific results", "", "Descriptive v3 comparison only; the historical reports never supply the fresh result or select the serving model.", "", "| Phase / arm | Fresh − previous R@10 | Fresh − previous R@100 |", "| --- | ---: | ---: |"]
    for phase, row in result["comparison"].items():
        for arm, value in row["fresh_minus_previous"].items():
            lines.append(f"| {phase} / {arm} | {value['recall10']:+.8f} | {value['recall100']:+.8f} |")
    lines += ["", "## Local HTTP measurements", "", result["http"]["latency_scope"], "", "| Batch | Concurrency | Queries | Queries/s | Request p50 ms | p95 ms | p99 ms |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["http"]["configurations"]:
        latency = row["latency_ms"]
        lines.append(f"| {row['batch_size']} | {row['concurrency']} | {row['measured_queries']} | {row['successful_queries_per_second']:.3f} | {latency['p50']:.3f} | {latency['p95']:.3f} | {latency['p99']:.3f} |")
    resources = result["http"]["resources"]
    lines += ["", f"Peak server RSS: {resources['server_peak_rss_bytes']:,} bytes. " + result["http"]["resource_scope"], "", "| Retrieval microbenchmark | Observed wall speed ratio v2/v3 |", "| --- | ---: |"]
    for mode, row in result["retrieval_microbenchmark"]["summary"].items():
        lines.append(f"| {mode} | {row['observed_wall_speedup_v2_over_v3']:.4f} |")
    lines += ["", "## Execution receipts", "", "| Stage | Status | Started UTC | Finished UTC | Seconds | Receipt SHA-256 |", "| --- | --- | --- | --- | ---: | --- |"]
    for row in result["stages"]:
        lines.append(f"| {row['name']} | {row['status']} | {row['started_utc']} | {row['finished_utc']} | {row['elapsed_seconds']:.3f} | `{row['receipt_sha256']}` |")
    lines += ["", "## Input bindings", "", f"Dependency lock: `{result['environment']['lock_sha256']}` ({result['environment']['locked_distributions']} pinned distributions).", "", "| Raw input | Bytes | SHA-256 |", "| --- | ---: | --- |"]
    lines += [f"| {name} | {item['bytes']:,} | `{item['sha256']}` |" for name, item in result["raw_files"].items()]
    lines += ["", "## Interpretation limits", ""] + [f"- {item}" for item in result["limits"]]
    return "\n".join(lines) + "\n"


def publish(workspace, acceptance_path, output_dir):
    output_dir = Path(output_dir)
    outputs = [output_dir / "execution_summary.json", output_dir / "RESULTS.md"]
    a.require(not any(path.exists() for path in outputs), "retain previous results; choose a new output directory")
    result = build_summary(workspace, acceptance_path)
    rendered = markdown(result)  # Validate both products before creating either.
    p.write_new(outputs[0], result)
    with outputs[1].open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/raw_rebuild"))
    args = parser.parse_args()
    result = publish(args.workspace, args.acceptance, args.output_dir)
    print(json.dumps({"passed": True, "stages": len(result["stages"]),
                      "confirm_gate_passed": result["v3"]["confirm"]["deployment_gate_passed"]}))


if __name__ == "__main__":
    main()
