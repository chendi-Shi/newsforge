"""Independent release audit: aggregate saved arrays, never fit or predict.

Run only after catalog.json, ranking.json and ranking-full.json are complete.
Missing/mismatching evidence raises before release_integrity.json is written.
This audit reads local per-request arrays and publishes aggregate checks only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


METRICS = ("auc", "mrr", "ndcg5", "ndcg10")
VARIANTS = ("content_only", "context_only", "full")
SEEDS = (17, 29, 43)
RUNS = tuple(f"{variant}-seed{seed}" for variant in VARIANTS for seed in SEEDS)
LOWER, UPPER, FIRST_DEV = 1573722000, 1573772400, 1573776000
CATALOG_NAMES = ("full", "no_popular", "no_content", "no_tower", "popular_only",
                 "content_only", "tower_only", "no_freshness", "frozen_feedback", "selected",
                 "selected_rerank_content_only", "selected_rerank_context_only", "selected_rerank_full")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(left, right, message, tolerance=1e-12):
    require(np.isfinite(left) and np.isfinite(right) and abs(float(left) - float(right)) <= tolerance,
            f"{message}: {left!r} != {right!r}")


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


class Audit:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.digests, self.checks = {}, {}
        self.data = self.root / "data/processed"
        self.split = np.load(self.data / "all_split.npy", mmap_mode="r", allow_pickle=False)
        self.timestamps = np.load(self.data / "all_timestamps.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(self.data / "all_offsets.npy", mmap_mode="r", allow_pickle=False)

    def path(self, path):
        return self.root / Path(path)

    def digest(self, path, expected=None):
        absolute = self.path(path).resolve()
        key = str(absolute.relative_to(self.root)).replace("\\", "/")
        if key not in self.digests:
            digest = hashlib.sha256()
            with absolute.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            self.digests[key] = digest.hexdigest()
        if expected is not None:
            require(self.digests[key] == expected, f"SHA256 mismatch: {key}")
        return self.digests[key]

    def read(self, path):
        self.digest(path)
        return json.loads(self.path(path).read_text(encoding="utf-8"))

    def load(self, path):
        self.digest(path)
        return np.load(self.path(path), mmap_mode="r", allow_pickle=False)

    def expected_rows(self, split, count=0, mature=False):
        rows = np.flatnonzero(self.split == split)
        if mature:
            rows = rows[(self.timestamps[rows] >= LOWER) & (self.timestamps[rows] < UPPER)]
        if count:
            require(len(rows) >= count, "sampling population smaller than declared count")
            rows = np.sort(np.random.default_rng(2026).choice(rows, count, replace=False))
        return rows

    def ranking(self, report, directory, expected_rows, expected_runs, name):
        rows = self.load(f"{directory}/rows.npy")
        require(np.array_equal(rows, expected_rows), f"{name}: wrong fixed row population/order")
        require(report["row_indices_sha256"] == row_hash(rows), f"{name}: row SHA")
        require(report["impressions"] == len(rows), f"{name}: impression denominator")
        candidates = int(np.asarray(self.offsets[rows + 1] - self.offsets[rows], dtype=np.int64).sum())
        require(report["candidates"] == candidates, f"{name}: candidate denominator")
        require(set(report["metrics"]) == set(expected_runs), f"{name}: missing or extra model results")
        per_seed, per_row = {}, {}
        for run in expected_runs:
            values = self.load(f"{directory}/{run}_metrics.npy")
            require(values.shape == (len(rows), 4), f"{name}/{run}: metric shape")
            require(not np.isinf(values).any(), f"{name}/{run}: infinite metric")
            finite = values[np.isfinite(values)]
            require(np.all((finite >= -1e-12) & (finite <= 1 + 1e-12)), f"{name}/{run}: metric range")
            per_seed[run] = np.nanmean(values, axis=0)
            per_row[run] = np.asarray(values)
            for column, metric in enumerate(METRICS):
                saved = report["metrics"][run][metric]
                close(per_seed[run][column], saved["mean"], f"{name}/{run}/{metric}")
                require(int(np.isfinite(values[:, column]).sum()) == saved["eligible_impressions"],
                        f"{name}/{run}/{metric}: finite denominator")
        variant_rows = {}
        for variant in VARIANTS:
            names = [f"{variant}-seed{seed}" for seed in SEEDS]
            if not all(run in per_seed for run in names):
                continue
            summaries = np.stack([per_seed[run] for run in names])
            variant_rows[variant] = np.stack([per_row[run] for run in names]).mean(axis=0)
            for column, metric in enumerate(METRICS):
                saved = report["seed_statistics"][variant][metric]
                close(summaries[:, column].mean(), saved["mean"], f"{name}/{variant}/{metric}: seed mean")
                close(summaries[:, column].std(ddof=1), saved["sample_std"], f"{name}/{variant}/{metric}: seed sample SD")
        for before, after in (("content_only", "full"), ("context_only", "full"), ("content_only", "context_only")):
            if before not in variant_rows or after not in variant_rows:
                continue
            delta = np.nanmean(variant_rows[after] - variant_rows[before], axis=0)
            comparison = report["paired_user_intervals"][f"{after}_minus_{before}"]
            for column, metric in enumerate(METRICS):
                close(delta[column], comparison[metric]["delta"], f"{name}/{after}-{before}/{metric}: paired point delta")
                low, high = comparison[metric]["ci95"]
                require(np.isfinite([low, high]).all() and low <= high, f"{name}: invalid CI")
        self.digest(f"{directory}/full_features.npy", report["feature_sha256"])
        require(report["feature_columns"] == 22, f"{name}: feature schema")
        require(all(pool["num_threads"] == 1 for pool in report["native_thread_pools"]), f"{name}: thread cap")
        self.checks[name] = {"impressions": len(rows), "candidates": candidates, "models": len(expected_runs),
                             "row_indices_sha256": row_hash(rows), "metric_means_and_denominators_match": True,
                             "three_seed_variants_verified": len(variant_rows)}
        return per_seed

    def catalog(self, report):
        private = "artifacts/v2/dev_catalog_counts.npz"
        self.digest(private)
        with np.load(self.path(private), allow_pickle=False) as saved:
            require(set(saved.files) == {"rows", "denominators", *CATALOG_NAMES}, "catalog: expected 13 strategies")
            rows, denominators = saved["rows"], saved["denominators"]
            require(np.array_equal(rows, self.expected_rows(2, 5000)), "catalog: fixed dev row population/order")
            require(denominators.shape == (5000, 5), "catalog: denominator shape")
            require(report["queries"] == 5000 and report["row_indices_sha256"] == row_hash(rows), "catalog: queries/row SHA")
            candidates = np.load(self.data / "all_candidates.npy", mmap_mode="r", allow_pickle=False)
            labels = np.load(self.data / "all_labels.npy", mmap_mode="r", allow_pickle=False)
            histories = np.load(self.data / "all_history.npy", mmap_mode="r", allow_pickle=False)
            first_seen = np.load(self.data / "news_first_seen.npy", mmap_mode="r", allow_pickle=False)
            clicks = np.load(self.data / "news_train_clicks.npy", mmap_mode="r", allow_pickle=False)
            train_cutoff = self.read("data/processed/manifest.json")["train_cutoff_timestamp"]
            reconstructed = np.zeros_like(denominators)
            for position, row in enumerate(rows):
                start, end = self.offsets[row:row + 2]
                positive = set(map(int, candidates[start:end][labels[start:end] == 1]))
                seen = set(map(int, histories[row][histories[row] > 0]))
                cutoff = int(self.timestamps[row]) // 3600 * 3600 - 3600
                eligible = {item for item in positive if first_seen[item] < cutoff and item not in seen}
                reconstructed[position] = [len(positive), len(eligible),
                    sum(first_seen[item] > train_cutoff for item in eligible),
                    sum(clicks[item] == 0 for item in eligible), len(positive & seen)]
            require(np.array_equal(reconstructed, denominators), "catalog: denominators differ from actual source logs")
            totals = denominators.sum(axis=0)
            for field, total in zip(("positive_clicked_items", "eligible_positive_items", "cold_eligible_positive_items",
                                     "zero_train_click_eligible_positive_items", "repeat_clicks"), totals):
                require(report[field] == int(total), f"catalog: {field}")
            require(set(report["metrics"]) == set(CATALOG_NAMES), "catalog report: expected 13 strategies")
            for name in CATALOG_NAMES:
                values = saved[name]
                require(values.shape == (5000, 5) and np.issubdtype(values.dtype, np.integer), f"catalog/{name}: count schema")
                require(np.all(values >= 0), f"catalog/{name}: negative hit counts")
                require(np.all(values[:, 0] <= np.minimum(10, denominators[:, 0])), f"catalog/{name}: top10 hit bounds")
                require(np.all(values[:, 1] <= np.minimum(100, denominators[:, 1])), f"catalog/{name}: top100 hit bounds")
                require(np.all(values[:, 2:4] <= denominators[:, 2:4]), f"catalog/{name}: subgroup hit bounds")
                require(np.all(values[:, 4] >= values[:, 1]) and np.all(values[:, 4] <= denominators[:, 1]),
                        f"catalog/{name}: union oracle bounds")
                sums, metric = values.sum(axis=0), report["metrics"][name]
                expected = {"click_micro_recall10": sums[0] / max(totals[0], 1),
                            "click_micro_recall100": sums[1] / max(totals[0], 1),
                            "eligible_click_micro_recall100": sums[1] / max(totals[1], 1),
                            "cold_eligible_recall100": sums[2] / max(totals[2], 1),
                            "zero_train_click_eligible_recall100": sums[3] / max(totals[3], 1),
                            "route_union_oracle_recall": sums[4] / max(totals[0], 1)}
                for field, value in expected.items():
                    close(value, metric[field], f"catalog/{name}/{field}")
                require(metric["hits10"] == int(sums[0]) and metric["hits100"] == int(sums[1]), f"catalog/{name}: hit sums")
                if name.startswith("selected_rerank_"):
                    require(np.array_equal(values[:, 1:], saved["selected"][:, 1:]), f"catalog/{name}: rerank changed pool")
                if name in self.selection_configs:
                    require(metric["candidate_config"] == self.selection_configs[name], f"catalog/{name}: frozen config")
            selected_name = self.selection["candidate_selected_name"]
            require(np.array_equal(saved["selected"], saved[selected_name]), "catalog: selected strategy differs from frozen named recipe")
            for name, interval in report["paired_user_intervals"].items():
                if name.startswith("full_minus_"):
                    other = name[len("full_minus_"):-len("_recall100")]
                    delta = (saved["full"][:, 1] - saved[other][:, 1]).sum() / totals[0]
                else:
                    other = name[:-len("_minus_no_rerank_recall10")]
                    delta = (saved[other][:, 0] - saved["selected"][:, 0]).sum() / totals[0]
                close(delta, interval["delta"], f"catalog/{name}: paired micro point delta")
                require(interval["resampling_unit"] == "user" and interval["replicates"] == 1000
                        and interval["seed"] == 2026,
                        f"catalog/{name}: bootstrap metadata")
                low, high = interval["ci95"]
                require(np.isfinite([low, high]).all() and low <= high, f"catalog/{name}: invalid CI")
            require(report["execution_schedule"]["reuse_current_query_semantic_routes"] is True
                    and report["execution_schedule"]["service_default_reuses_query_routes"] is False,
                    "catalog: cache/service timing semantics")
            require(all(pool["num_threads"] == 1 for pool in report["native_thread_pools"]), "catalog: thread cap")
            self.checks["catalog"] = {"queries": 5000, "strategies": len(CATALOG_NAMES),
                                      "denominators_reconstructed_from_source_logs": True,
                                      "all_hit_sums_and_recall_ratios_match": True,
                                      "positive_clicked_items": int(totals[0]),
                                      "eligible_positive_items": int(totals[1])}

    def frozen_inputs(self, selection, reports):
        self.selection = selection
        frozen = selection["bindings"]
        required_sources = {"experiment.py", "retrieval.py", "temporal.py", "ranker.py",
                            "verify_cache.py", "mature_selection.py", "select_backbone.py"}
        require(set(frozen["source"]) == required_sources, "selection: seven frozen calculation sources")
        for name, digest in frozen["source"].items():
            self.digest(f"newsforge/v2/{name}", digest)
        self.digest("newsforge/metrics.py", frozen["metrics_source"])
        self.digest("data/processed/manifest.json", frozen["data_manifest"])
        self.digest("data/temporal_v2/manifest.json", frozen["temporal_manifest"])
        temporal = self.read("data/temporal_v2/manifest.json")
        for name, record in temporal["arrays"].items():
            self.digest(f"data/temporal_v2/{name}", record["sha256"])
        for name, digest in frozen["processed_files"].items():
            self.digest(f"data/processed/{name}", digest)
        paths = {"article_embeddings": "artifacts/article_embeddings.npy", "retriever": "artifacts/retriever.pt",
                 "encoder": "artifacts/v2/user_encoder.npz", "encoder_manifest": "artifacts/v2/user_encoder.npz.json"}
        for name, path in paths.items():
            self.digest(path, frozen["retrieval_artifacts"][name])
        require(set(frozen["ranker_artifacts"]) == set(RUNS), "selection: nine weight identities")
        for run, record in frozen["ranker_artifacts"].items():
            self.digest(record["model_path"], record["model_sha256"])
            self.digest(record["model_path"] + ".manifest.json", record["manifest_sha256"])
        self.digest("reports/v2/ranker_matured_selection.json", frozen["ranker_selection"])
        self.digest("reports/v2/query_cache_equivalence.json", frozen["query_cache_equivalence"])
        # Other explicit evidence bindings can use their own semantic key names.
        for key, path in (("backbone_matured_selection", "reports/v2/backbone_matured_selection.json"),
                          ("matured_internal_evaluation", "reports/v2/ranker_matured_internal_evaluation.json")):
            if key in frozen:
                self.digest(path, frozen[key])
        selection_hash = self.digest("artifacts/v2/selection.json")
        for phase, report in reports.items():
            require(report["phase"] == phase, f"{phase}: phase tag")
            require(report["selection_sha256"] == selection_hash and report["bindings"] == frozen,
                    f"{phase}: configuration changed after selection")
            require(report["selected_before_v2_dev"] == selection["created_utc"], f"{phase}: selection timestamp")
        v1 = self.read("reports/ranking_experiment.json")["source_sha256"]
        require(len(v1) == 20, "v1: expected 20 frozen sources")
        for path, digest in v1.items():
            self.digest(path, digest)
        self.checks["frozen_inputs"] = {"v1_source_files": 20, "v2_selection_source_files": 7,
                                         "ranker_models": 9, "all_saved_sha256_bindings_match": True,
                                         "dev_phases_share_one_frozen_selection": True}

    def mature(self, selection, evaluation, backbone):
        original = self.read("reports/v2/ranker/selection.json")
        source_hash = self.digest("reports/v2/ranker/selection.json")
        self.digest("reports/v2/ranker_matured_internal_evaluation.json", selection["mature_internal_evaluation_sha256"])
        require(selection["source_selection_sha256"] == source_hash, "mature: original nine-model evidence SHA")
        audit = selection["maturity_audit"]
        require(audit == evaluation["maturity_audit"], "mature: audit differs between selection/evaluation")
        require(audit["validation_min_inclusive"] == LOWER and audit["validation_max_exclusive"] == UPPER
                and audit["first_dev_timestamp"] == FIRST_DEV and audit["first_dev_snapshot_cutoff"] == UPPER,
                "mature: strict first-dev feedback boundary")
        rows = self.expected_rows(1, 6000, mature=True)
        require(audit["sampled_rows"] == 6000 and audit["row_indices_sha256"] == row_hash(rows), "mature: fixed 6000 rows")
        require(audit["sampled_timestamp_min"] == int(self.timestamps[rows].min())
                and audit["sampled_timestamp_max"] == int(self.timestamps[rows].max())
                and LOWER <= int(self.timestamps[rows].min()) <= int(self.timestamps[rows].max()) < UPPER,
                "mature: observed timestamp bounds")
        require(audit["all_rows_split1"] is True and audit["all_selection_feedback_strictly_before_first_dev_cutoff"] is True,
                "mature: internal-only maturity flags")
        reused = selection["weight_reuse_audit"]
        expected = {entry["run_id"]: entry["model_sha256"] for entry in original["models"]}
        require(reused == evaluation["weight_reuse_audit"] and reused["model_sha256_before"] == expected
                and reused["model_sha256_after"] == expected and reused["fit_invoked"] is False
                and reused["all_weights_unchanged"] is True and reused["fixed_max_iter"] == 80
                and reused["early_stopping"] is False, "mature: immutable fixed-budget weights")
        quality = self.ranking(evaluation, "artifacts/v2/matured_internal_ranking", rows, RUNS, "mature_internal_ranking")
        best = {variant: max((f"{variant}-seed{seed}" for seed in SEEDS), key=lambda run: quality[run][3]) for variant in VARIANTS}
        winner = max(VARIANTS, key=lambda variant: quality[best[variant]][3])
        require(selection["selected"]["run_id"] == best[winner], "mature: selected model not internal argmax")
        for variant, run in best.items():
            require(selection["best_by_variant"][variant]["run_id"] == run, "mature: per-variant seed argmax")
        require(backbone["validation"]["row_indices_sha256"] == row_hash(rows)
                and backbone["validation"]["impressions"] == 6000
                and backbone["validation"]["min_timestamp_inclusive"] == LOWER
                and backbone["validation"]["max_timestamp_exclusive"] == UPPER,
                "backbone/ranker must share the same mature internal 6000 rows")
        for record in backbone["candidates"]:
            self.digest(record["checkpoint_path"], record["checkpoint_sha256"])
            require(record["epoch"] == 2 and record["state_matches_epoch2"] is True
                    and record["complete_training_epochs"] == 2, "backbone: fixed final-epoch identity")
        backbone_winner = max(sorted(backbone["candidates"], key=lambda record: record["seed"]),
                              key=lambda record: record["metrics"]["ndcg10"]["mean"])
        require(backbone["selected"]["seed"] == backbone_winner["seed"], "backbone: mature internal argmax")
        require(backbone["matches_current_retriever_state"] is True, "backbone: existing serving vectors not mature winner")
        self.digest("artifacts/retriever.pt", backbone["current_retriever_audit"]["current_file_sha256"])
        self.checks["mature_selection"] = {"rows": 6000, "row_indices_sha256": row_hash(rows),
                                             "backbone_and_ranker_same_rows": True, "strict_cutoff_passed": True,
                                             "old_nine_weight_files_unchanged": True, "selected_run": best[winner]}

    def internal_selection(self, report):
        require(report["frozen_selection"] == self.selection, "select.json/selection.json mismatch")
        expected_rows = self.expected_rows(1, 2000, mature=True)
        for name in ("window_selection", "candidate_ablation", "reranker_selection"):
            part = report[name]
            require(part["queries"] == 2000 and part["row_indices_sha256"] == row_hash(expected_rows), f"{name}: mature 2000 rows")
        windows = report["window_selection"]["metrics"]
        window = max((f"window_{hours}" for hours in (24, 48, 72)), key=lambda name: windows[name]["click_micro_recall100"])
        require(self.selection["ablation_window_hours"] == int(window.split("_")[1]), "select: window argmax")
        candidates = report["candidate_ablation"]["metrics"]
        allowed = CATALOG_NAMES[:8]
        chosen = max(allowed, key=lambda name: candidates[name]["click_micro_recall100"])
        require(self.selection["candidate_selected_name"] == chosen, "select: candidate recipe argmax")
        self.selection_configs = {name: candidates[name]["candidate_config"] for name in CATALOG_NAMES[:9]}
        self.selection_configs["selected"] = self.selection["serving"]["candidate_config"]
        require(self.selection_configs["selected"] == candidates[chosen]["candidate_config"], "select: serving recipe")
        reordering = report["reranker_selection"]["metrics"]
        rerank = max(("selected", *(f"selected_rerank_{variant}" for variant in VARIANTS)),
                     key=lambda name: reordering[name]["click_micro_recall10"])
        require(self.selection["serving"]["use_reranker"] == (rerank != "selected"), "select: rerank decision")
        if rerank != "selected":
            variant = rerank[len("selected_rerank_"):]
            require(self.selection["serving"]["ranker_identity"]["variant"] == variant, "select: reranker identity")
        else:
            require(self.selection["serving"]["ranker_identity"] is None and self.selection["serving"]["ranker_path"] is None,
                    "select: no-rerank serving contains ranker identity")
        self.checks["internal_selection"] = {"rows": 2000, "row_indices_sha256": row_hash(expected_rows),
                                               "all_decisions_reconstructed_from_internal_metrics": True,
                                               "candidate_selected_name": chosen, "use_reranker": rerank != "selected"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", default="reports/v2/release_integrity.json")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.root.resolve() / args.output
    require(not output.exists(), "release integrity report already exists; preserve it and choose a fresh --output")
    required = ["reports/v2/catalog.json", "reports/v2/ranking.json", "reports/v2/ranking-full.json",
                "reports/v2/select.json", "artifacts/v2/selection.json", "artifacts/v2/dev_catalog_counts.npz",
                "reports/v2/ranker_matured_selection.json", "reports/v2/ranker_matured_internal_evaluation.json",
                "reports/v2/backbone_matured_selection.json", "reports/v2/query_cache_equivalence.json",
                "artifacts/v2/dev_ranking_sample/rows.npy", "artifacts/v2/dev_ranking_full/rows.npy"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Formal evidence incomplete; no pass report generated: " + ", ".join(missing))
    started = time.perf_counter()
    audit = Audit(root)
    reports = {phase: audit.read(f"reports/v2/{phase}.json") for phase in ("catalog", "ranking", "ranking-full")}
    selected = audit.read("artifacts/v2/selection.json")
    audit.frozen_inputs(selected, reports)
    matured = audit.read("reports/v2/ranker_matured_selection.json")
    audit.mature(matured, audit.read("reports/v2/ranker_matured_internal_evaluation.json"),
                 audit.read("reports/v2/backbone_matured_selection.json"))
    require(selected["ranking_selected"] == matured["selected"], "full ranking model differs from mature internal winner")
    audit.internal_selection(audit.read("reports/v2/select.json"))
    audit.catalog(reports["catalog"])
    audit.ranking(reports["ranking"], "artifacts/v2/dev_ranking_sample", audit.expected_rows(2, 12000), RUNS, "dev_ranking_sample")
    chosen = selected["ranking_selected"]["run_id"]
    audit.ranking(reports["ranking-full"], "artifacts/v2/dev_ranking_full", audit.expected_rows(2), (chosen,), "dev_ranking_full")
    cache = audit.read("reports/v2/query_cache_equivalence.json")
    require(cache["passed"] is True and cache["candidate_batches_compared"] == 576
            and cache["mismatch_count"] == 0 and cache["max_absolute_score_error"] == 0, "query cache: exact equivalence evidence")
    audit.digest("newsforge/v2/retrieval.py", cache["source_sha256"]["retrieval.py"])
    cache_paths = {"features": "data/processed/news_features.npy", "first_seen": "data/processed/news_first_seen.npy",
                   "train_clicks": "data/processed/news_train_clicks.npy", "train_exposures": "data/processed/news_train_exposures.npy",
                   "news_ids": "data/processed/news_ids.json", "embeddings": "artifacts/article_embeddings.npy",
                   "retriever": "artifacts/retriever.pt", "encoder": "artifacts/v2/user_encoder.npz",
                   "encoder_manifest": "artifacts/v2/user_encoder.npz.json", "data_manifest": "data/processed/manifest.json",
                   "temporal_manifest": "data/temporal_v2/manifest.json"}
    for name, path in cache_paths.items():
        audit.digest(path, cache["bindings"][name])
    report = {"passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Independent saved-array aggregation and provenance audit; no model fitting, prediction or dev re-selection",
              "checks": audit.checks, "verified_sha256_files": audit.digests,
              "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "seconds": time.perf_counter() - started,
              "limitations": ["Saved per-query hit/metric arrays are aggregated independently; predictions are not recomputed.",
                              "Bootstrap point deltas and metadata are checked; CI resampling is not rerun."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"passed": True, "report": str(output), "checks": audit.checks,
                      "verified_files": len(audit.digests)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
