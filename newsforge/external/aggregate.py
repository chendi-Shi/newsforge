"""Recompute all reported effect sizes from immutable per-query predictions."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .common import (DAYS, EVAL_START, METHODS, check_files, check_previous_bindings,
                     read_json, sha256, stage, write_new_json)


def paired_interval(users, differences, *, repeats=1000, seed=2026):
    users = np.asarray(users)
    difference = np.asarray(differences, dtype=np.float64)
    if users.ndim != 1 or difference.shape != users.shape or not len(users):
        raise ValueError("bootstrap inputs must be nonempty aligned vectors")
    unique, inverse = np.unique(users, return_inverse=True)
    numerator = np.bincount(inverse, weights=difference)
    denominator = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = rng.integers(0, len(unique), size=len(unique))
        values[index] = numerator[sample].sum() / denominator[sample].sum()
    lo, hi = np.quantile(values, [.025, .975])
    return {"difference": float(difference.sum() / len(users)), "ci95": [float(lo), float(hi)],
            "user_clusters": len(unique), "bootstrap_repetitions": repeats,
            "bootstrap_rng_seed": seed, "valid_repetitions": int(np.isfinite(values).sum()),
            "estimand": "total paired hit difference / total requests in resampled user clusters"}


def summarize(users, hits, hits100, reciprocal):
    users = np.asarray(users)
    hits = np.asarray(hits, dtype=np.int64)
    hits100 = np.asarray(hits100, dtype=np.int64)
    reciprocal = np.asarray(reciprocal, dtype=np.float64)
    if not len(users):
        return {"queries": 0, "status": "no_queries", "methods": {}, "comparisons": {}}
    if (hits.shape != (len(users), len(METHODS)) or hits100.shape != hits.shape
            or reciprocal.shape != hits.shape or not np.isin(hits, [0, 1]).all()
            or not np.isin(hits100, [0, 1]).all() or not np.all(hits <= hits100)
            or not np.isfinite(reciprocal).all() or not np.all((reciprocal >= 0) & (reciprocal <= 1))
            or not np.array_equal(reciprocal > 0, hits.astype(bool))
            or not np.all(hits100 == hits100[:, :1])):
        raise ValueError("invalid per-query arrays or Recall@100 invariant")
    result = {"queries": len(users), "users": len(np.unique(users)), "methods": {}}
    for index, method in enumerate(METHODS):
        result["methods"][method] = {"hits10": int(hits[:, index].sum()),
            "hit_rate10": float(hits[:, index].mean()), "hits100": int(hits100[:, index].sum()),
            "hit_rate100": float(hits100[:, index].mean()), "mrr10": float(reciprocal[:, index].mean()),
            "denominator": len(users)}
    result["comparisons"] = {"full13+gate_minus_baseline": paired_interval(users, hits[:, 1] - hits[:, 0])}
    for index in (2, 3, 4):
        result["comparisons"][METHODS[index] + "_minus_full13+gate"] = paired_interval(users, hits[:, index] - hits[:, 1])
    return result


def aggregate(repo, run_dir, reports):
    run_dir, reports = Path(run_dir), Path(reports)
    with stage(run_dir, "aggregate") as (out, log):
        began = time.perf_counter()
        reports.mkdir(parents=True, exist_ok=False)
        prep = read_json(run_dir / "prepare/receipt.json")
        evaluated = read_json(run_dir / "evaluate/receipt.json")
        bindings = check_previous_bindings(repo, evaluated["bindings"])
        if evaluated["prepare_receipt_sha256"] != sha256(run_dir / "prepare/receipt.json"):
            raise ValueError("evaluation is bound to a different preparation")
        check_files(run_dir / "evaluate", evaluated["files"])
        audit = read_json(run_dir / "evaluate/audit.json")
        if audit["status"] != "passed" or audit["queries"] != 128:
            raise ValueError("independent external audit has not passed")
        queries = np.load(run_dir / "evaluate/queries.npy", allow_pickle=False)
        hits = np.load(run_dir / "evaluate/hits10.npy", allow_pickle=False)
        hits100 = np.load(run_dir / "evaluate/hits100.npy", allow_pickle=False)
        reciprocal = np.load(run_dir / "evaluate/reciprocal10.npy", allow_pickle=False)
        # Aggregate independently recomputes targets from saved candidate IDs and orders.
        pools = np.load(run_dir / "evaluate/pools.npy", mmap_mode="r", allow_pickle=False)
        orders = np.load(run_dir / "evaluate/orders.npy", mmap_mode="r", allow_pickle=False)
        sizes = np.load(run_dir / "evaluate/sizes.npy", allow_pickle=False)
        for row in range(len(queries)):
            m = int(sizes[row])
            pool = pools[row, :m]
            for method in range(5):
                positions = orders[row, method, :m]
                if not np.array_equal(np.sort(positions), np.arange(m)):
                    raise ValueError("saved ranking is not a permutation")
                ranked = pool[positions]
                locations = np.flatnonzero(ranked == queries[row, 3])
                correct_hit = bool(len(locations) and locations[0] < 10)
                correct_rr = 1. / (int(locations[0]) + 1) if correct_hit else 0.
                if (int(hits[row, method]) != int(correct_hit)
                        or int(hits100[row, method]) != int(bool(len(locations)))
                        or reciprocal[row, method] != correct_rr):
                    raise ValueError("independent aggregate prediction/target reconciliation failed")
        with threadpool_limits(limits=1):
            overall = summarize(queries[:, 2], hits, hits100, reciprocal)
            by_day = {}
            for index, day in enumerate(DAYS):
                mask = ((queries[:, 1] >= EVAL_START + index * 86400)
                        & (queries[:, 1] < EVAL_START + (index + 1) * 86400))
                by_day[day] = summarize(queries[mask, 2], hits[mask], hits100[mask], reciprocal[mask])
        primary = overall["comparisons"]["full13+gate_minus_baseline"]
        supported = primary["difference"] > 0 and primary["ci95"][0] > 0
        report = {"schema_version": 1, "status": "passed", "dataset": "Adressa light one week",
            "scope": "Independent retrospective cross-publisher and cross-language frozen-MIND-model stress test",
            "external_model_fitting": False, "external_hyperparameter_selection": False,
            "prospective_future_generalization": False, "online_ab_test": False,
            "feedback": "observed page visits; no exposure negatives or calibrated CTR",
            "text_representation": "frozen English MIND TF-IDF/SVD applied to unmodified Norwegian first-observed titles",
            "category_representation": "frozen exact MIND category lookup; unknown categories retain UNKNOWN one-hot",
            "delta": 10, "seeds": [17, 29, 43], "overall": overall, "by_day": by_day,
            "positive_transfer_supported_on_preregistered_sample": supported,
            "conclusion": ("positive external transfer evidence on this preregistered sample" if supported else
                           "this external test did not support positive transfer"),
            "selection_policy": "Report all fixed methods; never change the production recipe based on these outcomes.",
            "interval_limit": "User-cluster intervals condition on three realized days and common feedback; no across-date or production guarantee.",
            "source_counts": prep["counts"], "identity_counts": prep["identity"],
            "daily_samples": prep["daily_samples"], "sample_identity_sha256": prep["sample_identity_sha256"],
            "article_text_coverage": prep["transformed"]["coverage"],
            "query_diagnostics": evaluated["diagnostics"],
            "audit": audit, "evaluation_elapsed_seconds": evaluated["elapsed_seconds"],
            "evaluation_memory": evaluated["memory"],
            "preparation_elapsed_seconds": prep["elapsed_seconds"],
            "preparation_memory": prep["memory"],
            "bindings": bindings, "archive_sha256": prep["archive_sha256"],
            "phase_receipts_sha256": {name: sha256(run_dir / name / "receipt.json")
                                     for name in ("acquisition", "prepare", "evaluate")},
            "per_query_files": evaluated["files"], "aggregate_elapsed_seconds": time.perf_counter() - began}
        check_previous_bindings(repo, bindings)
        check_files(run_dir / "evaluate", evaluated["files"])
        write_new_json(reports / "evaluation.json", report)
        write_new_json(reports / "audit.json", audit)
        lines = ["# Adressa 外部冻结模型迁移结果", "",
            "本次只测量另一个出版商和语言的历史阅读事件覆盖，未训练外部模型、未调参，也不是线上 A/B 或未来时间测试。", "",
            "| 方法 | Top10 命中 / 请求 | HitRate@10 | HitRate@100 | MRR@10 |",
            "|---|---:|---:|---:|---:|"]
        for method in METHODS:
            result = overall["methods"][method]
            lines.append(f"| {method} | {result['hits10']} / {result['denominator']} | {result['hit_rate10']:.8f} | {result['hit_rate100']:.8f} | {result['mrr10']:.8f} |")
        lines += ["", f"主差（full13+gate − baseline）：{primary['difference'] * 100:.6f} 个百分点；"
                  f"配对用户簇 95% 区间 [{primary['ci95'][0] * 100:.6f}, {primary['ci95'][1] * 100:.6f}] 个百分点。", "",
                  "结论：" + ("此预注册样本提供正向迁移证据。" if supported else "本外部测试未支持正向迁移。"), "",
                  "128 个固定请求的独立慢速审计通过；所有方法每请求候选集合和 HitRate@100 完全一致。", "",
                  "英文词表对挪威语的覆盖及未知类别比例见 evaluation.json；未知/空文本请求保留在分母。"
                  "这不是跨语言语义编码能力或生产推荐收益的证明。", "",
                  "完整逐日结果、全部消融、输入及产物 SHA、覆盖诊断均见 [evaluation.json](evaluation.json)。", ""]
        with (reports / "RESULTS.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
        write_new_json(out / "receipt.json", {"schema_version": 1, "status": "passed",
            "bindings": bindings, "public_evaluation_sha256": sha256(reports / "evaluation.json"),
            "public_audit_sha256": sha256(reports / "audit.json"),
            "all_prediction_counts_recomputed": True, "positive_transfer_supported": supported})
        log("aggregate_complete", queries=len(queries), positive_transfer_supported=supported)
        return report

