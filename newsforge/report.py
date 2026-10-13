"""Build an auditable results report only after all declared experiments finish."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np

from .evaluate import select_validation_rows
from .gbdt_final_evaluation import EVIDENCE_FILE as GBDT_SCHEDULING_EVIDENCE, validate_internal_audit
from .metrics import METRICS, user_cluster_interval


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def controlled_sampling_summary(training, seeds):
    """Describe both second epochs separately from validation-selected weights."""
    rows = []
    for seed in seeds:
        random = training[f"mean_random-seed{seed}"]
        mixed = training[f"mean_mixed-seed{seed}"]
        scores = {name: [epoch["validation"]["metrics"]["ndcg10"]["mean"]
                         for epoch in record["history"]]
                  for name, record in (("random", random), ("mixed", mixed))}
        if (len(scores["random"]) != 2 or len(scores["mixed"]) != 2
                or not np.isclose(scores["random"][0], scores["mixed"][0], rtol=0, atol=1e-12)):
            raise ValueError(f"Common warmup validation evidence differs for seed {seed}")
        for name, record in (("random", random), ("mixed", mixed)):
            if (record["best_epoch"] != int(np.argmax(scores[name])) + 1
                    or not np.isclose(record["validation_ndcg10"], max(scores[name]), rtol=0, atol=1e-12)):
                raise ValueError(f"Selected checkpoint does not follow validation ndcg10: {name}, seed {seed}")
        rows.append({"seed": seed, "common_epoch1_ndcg10": scores["random"][0],
                     "random_epoch2_ndcg10": scores["random"][1],
                     "mixed_epoch2_ndcg10": scores["mixed"][1],
                     "epoch2_mixed_minus_random_pp": (scores["mixed"][1] - scores["random"][1]) * 100,
                     "random_selected_epoch": random["best_epoch"],
                     "mixed_selected_epoch": mixed["best_epoch"]})
    return {"rows": rows, "seeds": len(rows),
            "random_selected_warmup": sum(row["random_selected_epoch"] == 1 for row in rows),
            "mixed_selected_warmup": sum(row["mixed_selected_epoch"] == 1 for row in rows)}


def validate_final_evidence(benchmark, retrieval, manifest, manifest_hash, artifacts):
    """Reject reports from different serving bundles or empty recall populations."""
    ann, health = benchmark["ann"], benchmark["http"]["health"]
    equivalent = (("model_version", "model_version"), ("ann_index_version", "index_version"),
                  ("indexed_items", "catalogue_items"), ("catalogue_rows", "catalogue_rows"),
                  ("ann_kind", "index_kind"))
    if any(ann[key] != health[health_key] for key, health_key in equivalent):
        raise ValueError("ANN and HTTP evidence refer to different models, indexes or catalogues")
    if ann["ann_kind"] != "hnsw":
        raise ValueError("Final serving evidence must use the declared HNSW index")
    artifacts = Path(artifacts)
    hashes = {"retriever": hashlib.sha256((artifacts / "retriever.pt").read_bytes()).hexdigest(),
              "reranker": hashlib.sha256((artifacts / "reranker.pt").read_bytes()).hexdigest()}
    # Exact load_engine algorithm: default JSON separators, with sorted keys.
    current_model_version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if ann["model_version"] != current_model_version:
        raise ValueError("Serving benchmark does not match the current retriever/reranker checkpoints")
    index = load(artifacts / "index_hnsw" / "manifest.json")
    if (index["index_version"] != ann["ann_index_version"]
            or index["model_version"] != hashes["retriever"]
            or index["kind"] != "hnsw"
            or index["indexed_items"] != ann["indexed_items"]
            or index["catalogue_rows"] != ann["catalogue_rows"]):
        raise ValueError("Current HNSW manifest is not bound to the benchmark and retriever checkpoint")
    if benchmark["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("Serving benchmark uses a different processed-data manifest")
    expected_queries = min(5000, manifest["official_dev_rows"])
    if retrieval["sample_impressions"] != expected_queries or expected_queries < 1:
        raise ValueError("Retrieval evidence must cover the declared fixed sample of official-dev requests")
    for denominator in ("positive_clicked_items", "eligible_positive_items", "cold_eligible_positive_items"):
        value = retrieval[denominator]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Retrieval metric denominator must be a positive count: {denominator}")


def validate_scheduling_evidence(ranking, reports, artifacts, data, manifest_hash, training, original_rows):
    """Bind all three internal equivalence audits to the final ranker evaluation."""
    code = Path(__file__).resolve().parent
    reports, artifacts, data = Path(reports), Path(artifacts), Path(data)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    core_hashes = {name: digest(code / name) for name in
                   ("models.py", "train.py", "store.py", "evaluate.py", "metrics.py")}
    scheduler_hash = digest(code / "evaluation_scheduler.py")
    schedule = ranking["evaluation_scheduling"]
    if (schedule["version"] != "stable_candidate_count_v1"
            or schedule["scheduler_source_sha256"] != scheduler_hash
            or schedule["entrypoint_source_sha256"] != digest(code / "final_evaluation.py")
            or schedule["ranker_only"] is not True or schedule["ranker_batch_size"] != 32
            or schedule["per_impression_equivalence_atol"] != 1e-12
            or schedule["original_metric_row_order_restored"] is not True
            or schedule["training_and_checkpoint_selection_unchanged"] is not True
            or schedule["original_evaluation_rows_sha256"] != hashlib.sha256(
                np.ascontiguousarray(original_rows).tobytes()).hexdigest()):
        raise ValueError("Final evaluation scheduling provenance or original row order is inconsistent")
    split = np.load(data / "all_split.npy", mmap_mode="r")
    try:
        validation_rows = select_validation_rows(SimpleNamespace(split=split), 12000)
    finally:
        split._mmap.close()
    if len(validation_rows) != 12000:
        raise ValueError("Scheduler equivalence requires the full fixed 12000-row internal sample")
    row_hash = hashlib.sha256(np.ascontiguousarray(validation_rows).tobytes()).hexdigest()
    audits = schedule["internal_validation_audits"]
    expected_ids = {f"ranker-seed{seed}" for seed in (17, 29, 43)}
    if len(audits) != 3 or {item["run_id"] for item in audits} != expected_ids:
        raise ValueError("Three distinct ranker scheduling audits are required")
    validated = []
    for seed in (17, 29, 43):
        run_id = f"ranker-seed{seed}"
        reference = next(item for item in audits if item["run_id"] == run_id)
        expected_file = ("evaluation_scheduler_validation_12000.json" if seed == 17 else
                         f"evaluation_scheduler_ranker_seed{seed}_validation_12000.json")
        if reference["passed"] is not True or reference["evidence_file"] != expected_file:
            raise ValueError(f"Missing or failed scheduler audit: {run_id}")
        path = reports / expected_file
        if reference["evidence_sha256"] != digest(path):
            raise ValueError(f"Scheduler evidence file hash changed: {run_id}")
        record = load(path)
        if (record["run_id"] != run_id or record["selected_epoch"] != training[run_id]["best_epoch"]
                or record["checkpoint_sha256"] != digest(artifacts / f"ranker_seed{seed}.pt")
                or record["data_manifest_sha256"] != manifest_hash
                or record["training_source_sha256"] != core_hashes
                or training[run_id]["training_source_sha256"] != core_hashes
                or record["scheduler_source_sha256"] != scheduler_hash
                or record["rows"] != 12000 or record["fixed_validation_rows"] != 12000
                or record["selected_row_indices_sha256"] != row_hash
                or record["metric_equivalence_passed"] is not True
                or record["per_impression_metric_equivalence_atol"] != 1e-12
                or record["environment"]["threads"] != 1
                or record["environment"]["torch"] != ranking["environment"]["torch"]):
            raise ValueError(f"Scheduler audit does not match current ranker/data/core/sample: {run_id}")
        for metric in METRICS:
            difference = record["metric_differences"][metric]
            if (difference["same_nan_mask"] is not True
                    or difference["pairs_differing_more_than_1e_12"] != 0
                    or not 0 <= difference["max_absolute_difference"] <= 1e-12
                    or not 0 < difference["finite_pairs"] <= 12000):
                raise ValueError(f"Per-impression metric equivalence failed: {run_id}, {metric}")
        for plan, count in ((record["scheduling"], 12000),
                            (ranking["runs"][run_id]["evaluation_scheduling"], len(original_rows))):
            if (plan["strategy"] != "stable_candidate_count" or plan["batch_size"] != 32
                    or plan["impressions"] != count or plan["ranker_only"] is not True
                    or any(plan[key] is not True for key in
                        ("same_population", "same_candidates_within_each_impression", "metric_rows_restored_to_original_order"))
                    or not 0 < plan["candidate_exposures"] <= plan["scheduled_candidate_slots"]
                    or plan["original_candidate_slots"] < plan["candidate_exposures"]):
                raise ValueError(f"Scheduler changed population/candidates/row order or has invalid slot counts: {run_id}")
        if any(not np.isfinite(record[key]["seconds"]) or record[key]["seconds"] <= 0
               for key in ("original", "scheduled")):
            raise ValueError(f"Scheduler audit has missing or invalid timings: {run_id}")
        validated.append(record)
    return validated


def validate_gbdt_scheduling_evidence(tree, reports, artifacts, data, manifest_hash, row_hash, original_rows):
    """Verify the final GBDT also uses the internally audited immutable schedule."""
    reports, artifacts, data = Path(reports), Path(artifacts), Path(data)
    code = Path(__file__).resolve().parent
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected = {"checkpoint_sha256": digest(artifacts / "gbdt.joblib"),
        "data_manifest_sha256": manifest_hash, "selected_row_indices_sha256": row_hash,
        "feature_mode": tree["feature_mode"], "feature_count": tree["feature_count"],
        "evaluation_source_sha256": {name: digest(code / name) for name in ("gbdt.py", "store.py", "evaluate.py", "metrics.py")},
        "scheduler_source_sha256": digest(code / "gbdt_evaluation_scheduler.py"),
        "order_helper_source_sha256": digest(code / "evaluation_scheduler.py")}
    evidence_path = reports / GBDT_SCHEDULING_EVIDENCE
    preflight = tree["evaluation_preflight"]
    selection_path = reports / "gbdt_feature_ablation.json"
    selection = load(selection_path)
    if (any(preflight[key] != value for key, value in expected.items())
            or preflight["status"] != "passed" or preflight["official_dev_evaluated_by_preflight"] is not False
            or preflight["features_sha256"] != digest(data / "news_features.npy")
            or preflight["entrypoint_source_sha256"] != digest(code / "gbdt_final_evaluation.py")
            or preflight["selection_evidence_sha256"] != digest(selection_path)
            or expected["checkpoint_sha256"] != selection["configurations"]["rich"]["preserved_model_sha256"]
            or preflight["internal_validation_evidence_file"] != GBDT_SCHEDULING_EVIDENCE
            or preflight["internal_validation_evidence_sha256"] != digest(evidence_path)):
        raise ValueError("Final GBDT scheduling preflight no longer matches current artifacts and evidence")
    record = load(evidence_path)
    validate_internal_audit(record, expected)
    plan = tree["evaluation_scheduling"]
    if (plan["strategy"] != "stable_candidate_count" or plan["gbdt_only"] is not True
            or plan["batch_size"] != 64 or plan["impressions"] != len(original_rows)
            or plan["candidate_exposures"] != tree["candidate_exposures"]
            or not 0 < plan["candidate_exposures"] <= min(plan["original_candidate_slots"], plan["scheduled_candidate_slots"])
            or any(plan[key] is not True for key in
                ("same_population", "same_candidates_within_each_impression", "metric_rows_restored_to_original_order"))
            or plan["original_evaluation_rows_sha256"] != hashlib.sha256(np.ascontiguousarray(original_rows).tobytes()).hexdigest()
            or plan["internal_validation_evidence_file"] != GBDT_SCHEDULING_EVIDENCE
            or plan["internal_validation_evidence_sha256"] != digest(evidence_path)):
        raise ValueError("Final GBDT schedule changed the candidates, population or original metric row order")
    return record


def main():
    root = Path(__file__).resolve().parent.parent
    reports, data, artifacts = root / "reports", root / "data/processed", root / "artifacts"
    ranking = load(reports / "ranking_experiment.json")
    tree = load(reports / "gbdt_evaluation.json")
    tree_train = load(reports / "gbdt_training.json")
    tree_ablation = load(reports / "gbdt_feature_ablation.json")
    retrieval = load(reports / "retrieval_experiment.json")
    benchmark = load(reports / "service_benchmark.json")
    manifest = load(data / "manifest.json")
    cache = load(data / "training_cache.json")
    configuration = ranking["configuration"]
    seeds, variants = [17, 29, 43], ["mean_random", "mean_mixed", "ranker"]
    if (sorted(configuration["seeds"]) != seeds
            or sorted(configuration["variants"]) != sorted(variants)
            or configuration["epochs"] != 2 or configuration.get("smoke_steps", 0)):
        raise ValueError("Report requires all three declared variants/seeds and two full epochs")
    rows = np.load(artifacts / "evaluation_rows.npy")
    expected_rows = np.flatnonzero(np.load(data / "all_split.npy", mmap_mode="r") == 2)
    if not np.array_equal(rows, np.load(artifacts / "evaluation_rows_gbdt.npy")):
        raise ValueError("GBDT and neural model test populations differ")
    if not np.array_equal(rows, expected_rows) or len(rows) != manifest["official_dev_rows"]:
        raise ValueError("Evaluation rows must be the complete official-dev split in its original row order")
    ann, http = benchmark["ann"], benchmark["http"]
    if (not http["all_requests_passed"] or not http["cases"]
            or not ann["exact_vs_numpy"]["all_passed"] or not ann["exact_vs_numpy"]["checks"]
            or ann["queries"] < 1 or ann["mean_ann_recall_at_k"] is None):
        raise ValueError("Incomplete benchmark or failed HTTP/ANN integrity checks")
    for case in http["cases"]:
        if (case["error_count"] or case["measured_http_requests"] < 1
                or case["successful_http_requests"] != case["measured_http_requests"]
                or case["successful_recommendation_queries"] < 1
                or case["request_latency"]["p95_ms"] is None):
            raise ValueError("HTTP benchmark has an empty or incomplete workload")
    single = next((case for case in http["cases"]
                   if case["http_batch_size"] == 1 and case["client_concurrency"] == 1), None)
    if single is None:
        raise ValueError("Resume requires the measured batch=1, concurrency=1 HTTP case")
    if not tree.get("official_development_test_evaluated") or tree["impressions"] != len(rows):
        raise ValueError("GBDT final evaluation is missing or incomplete")
    for run in ["popularity", "content"] + [f"{variant}-seed{seed}" for variant in variants for seed in seeds]:
        if ranking["runs"][run]["impressions"] != len(rows):
            raise ValueError(f"Incomplete ranking evaluation: {run}")
    eligible = cache["positive_training_examples"]
    manifest_hash = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    validate_final_evidence(benchmark, retrieval, manifest, manifest_hash, artifacts)
    training = {}
    for variant in variants:
        for seed in seeds:
            run = f"{variant}-seed{seed}"
            record = load(reports / f"{variant}_seed{seed}_training.json")
            epochs = record["history"]
            expected_policies = ["four_random", "four_random" if variant == "mean_random" else "two_hard_two_random"]
            if (record.get("trained_on_all_eligible_clicks") is not True
                    or record["eligible_positive_examples_per_epoch"] != eligible
                    or record["data_manifest_sha256"] != manifest_hash
                    or record["seed"] != seed or record["variant"] != variant
                    or record["training_configuration"]["epochs"] != 2
                    or [epoch["epoch"] for epoch in epochs] != [1, 2]
                    or [epoch["negative_policy"] for epoch in epochs] != expected_policies
                    or any(epoch["training_examples"] != eligible for epoch in epochs)
                    or record["best_epoch"] not in [1, 2]):
                raise ValueError(f"Incomplete or mismatched training evidence: {run}")
            training[run] = record
    for seed in seeds:
        random_digest = training[f"mean_random-seed{seed}"]["history"][0]["state_sha256"]
        mixed_digest = training[f"mean_mixed-seed{seed}"]["history"][0]["state_sha256"]
        if (random_digest != mixed_digest
                or ranking["matched_warmup_checks"][str(seed)]["identical"] is not True):
            raise ValueError(f"Matched-seed warmup audit failed: {seed}")
    sampling_comparison = controlled_sampling_summary(training, seeds)
    scheduling_audits = validate_scheduling_evidence(
        ranking, reports, artifacts, data, manifest_hash, training, rows)
    gbdt_scheduling_audit = validate_gbdt_scheduling_evidence(tree, reports, artifacts, data,
        manifest_hash, scheduling_audits[0]["selected_row_indices_sha256"], rows)
    checkpoint_explanation = (
        f"混合方案有 {sampling_comparison['mixed_selected_warmup']}/{len(seeds)} 个种子选中仅用随机负例训练的第 1 轮 warm-up；"
        f"随机方案有 {sampling_comparison['random_selected_warmup']}/{len(seeds)} 个种子选中第 1 轮。"
        "最终测试比较的是按内部验证选择 epoch 的训练方案；对于选中 warm-up 的混合方案，"
        "最终权重未包含第 2 轮难负例更新，不能把测试差值归因于未被选中的难负例权重。")
    sampling = tree_train["sampling"]
    if (sampling["selected_positive_pairs"] != min(500000, eligible)
            or sampling["binary_training_examples"] != 5 * sampling["selected_positive_pairs"]
            or sampling["negatives_per_positive"] != 4 or sampling["seed"] != 2026
            or tree_train["config"]["max_iter"] != 80 or tree_train["config"]["early_stopping"]):
        raise ValueError("GBDT training evidence does not match the declared baseline budget")
    tree_mode = tree_ablation["selected_feature_mode"]
    if (set(tree_ablation["configurations"]) != {"similarity", "rich"}
            or tree_mode != tree_train["feature_mode"] or tree_mode != tree["feature_mode"]
            or tree_ablation["official_development_test_evaluated"]
            or tree_ablation["shared_selected_positive_indices_sha256"] != sampling["selected_positive_indices_sha256"]
            or tree_ablation["shared_validation_row_indices_sha256"] != tree_train["validation_row_indices_sha256"]):
        raise ValueError("GBDT feature selection evidence is missing or inconsistent")
    expected_tree_mode = max(("similarity", "rich"), key=lambda mode:
        tree_ablation["configurations"][mode]["internal_validation"]["metrics"]["ndcg10"]["mean"])
    if tree_mode != expected_tree_mode:
        raise ValueError("GBDT selection does not follow internal-validation ndcg10")
    users = np.load(data / "all_users.npy", mmap_mode="r")[rows]
    base_values = np.mean([np.load(artifacts / f"metrics_mean_random-seed{seed}.npy") for seed in seeds], axis=0)
    tree_values = np.load(artifacts / "metrics_gbdt.npy")
    if base_values.shape != (len(rows), len(METRICS)) or tree_values.shape != base_values.shape:
        raise ValueError("Per-impression metric matrices have incompatible populations or columns")
    comparisons = dict(ranking["paired_user_bootstrap"])
    comparisons["gbdt"] = user_cluster_interval(base_values, tree_values, users)
    (reports / "gbdt_paired_comparison.json").write_text(json.dumps(comparisons["gbdt"], indent=2), encoding="utf-8")
    lines = ["# NewsForge 实测结果", "", "以下数字来自本目录保存的完整运行记录。指标使用 0–1 标度；差值使用百分点（pp）。", "",
             "## 数据与实验范围", "",
             f"本次 MIND-large 官方 train/dev 共 **{manifest['impressions']:,}** 个曝光组、**{manifest['candidates']:,}** 条候选曝光、**{manifest['news']:,}** 篇目录文章。",
             "", "| 切分 | 曝光组 | 候选曝光 | 点击 |", "|---|---:|---:|---:|"]
    for name, number in [("训练", "0"), ("内部时间验证", "1"), ("锁定官方 dev", "2")]:
        split = manifest["splits"][number]
        lines.append(f"| {name} | {split['impressions']:,} | {split['candidates']:,} | {split['positives']:,} |")
    lines += ["", f"每个神经模型每轮使用全部 **{cache['positive_training_examples']:,}** 个可构造负例的点击对；跳过无正例或无有效负例的曝光组 **{cache['skipped_no_positive_or_negative']:,}** 个。每个变体使用 {len(seeds)} 个种子，每种子两轮，内部验证选择 epoch。",
              f"每种 GBDT 特征配置均使用相同 **{tree_train['sampling']['selected_positive_pairs']:,}** 个正例、**{tree_train['sampling']['binary_training_examples']:,}** 条二分类样本；固定 80 轮、{tree_train['threads']} 个 CPU 线程。两种特征经内部验证选择正式模型 `{tree_mode}`（{tree_train['feature_count']} 维）。此项是不同预算的强基线比较。", "",
              "### 训练预算与内部选择", "",
              "训练耗时不含每轮内部验证；难负例挖掘计入训练时间。GBDT 特征构造与拟合分开报告。梯度样本预算相同不代表 CPU 时间相同。", "",
              "| 运行 | 可学习参数 | 每轮正例对 | 训练秒数 | 所选 epoch | 内部验证 nDCG@10 |", "|---|---:|---:|---:|---:|---:|"]
    for run, record in training.items():
        lines.append(f"| {run} | {record['parameters']:,} | {eligible:,} | {record['training_seconds']:.1f} | {record['best_epoch']} | {record['validation_ndcg10']:.5f} |")
    lines += ["", "### 随机与混合负例的内部受控对照", "",
              "下表只使用固定内部验证。相同种子的第 1 轮权重已通过摘要一致性检查；第 2 轮差值比较两种负例策略在相同累计训练轮数下的验证表现，不混同于最终选中 checkpoint 的测试差值。", "",
              "| Seed | 共同第 1 轮 nDCG@10 | 随机第 2 轮 | 混合第 2 轮 | 第 2 轮混合−随机（pp） | 随机所选 epoch | 混合所选 epoch |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for record in sampling_comparison["rows"]:
        lines.append(f"| {record['seed']} | {record['common_epoch1_ndcg10']:.5f} | {record['random_epoch2_ndcg10']:.5f} | {record['mixed_epoch2_ndcg10']:.5f} | {record['epoch2_mixed_minus_random_pp']:+.3f} | {record['random_selected_epoch']} | {record['mixed_selected_epoch']} |")
    lines += ["", checkpoint_explanation]
    lines += ["", "GBDT 内容特征对照是在内部验证阶段、官方 dev 仍锁定时增加，并非实验开始前预注册。原 15 维模型及结果保留；新增配置补充候选与历史均值的前 16 维 SVD 和全部类别列。两种配置用同一内部验证 nDCG@10 选择，其余超参数不变，均不提前停止。", "",
              "| GBDT 特征配置 | 维数 | 特征构造秒数 | 拟合秒数 | 内部验证 nDCG@10 | 进程最大观测 RSS（GiB） |", "|---|---:|---:|---:|---:|---:|"]
    for mode, record in tree_ablation["configurations"].items():
        memory = record.get("memory_observed")
        rss = f"{memory['max_observed_rss_bytes'] / 2**30:.3f}" if memory else "未采样"
        lines.append(f"| {mode} | {record['feature_count']} | {record['sampling']['feature_construction_seconds']:.1f} | {record['fit_seconds']:.1f} | {record['internal_validation']['metrics']['ndcg10']['mean']:.5f} | {rss} |")
    rich_memory = tree_ablation["configurations"]["rich"].get("memory_observed")
    if rich_memory:
        lines += ["", f"rich 训练期间系统最小观测可用内存为 {rich_memory['minimum_system_available_bytes'] / 2**20:.3f} MiB；本次静态预检低估瞬时压力，后续应串行运行并额外留出系统余量。"]
    lines += ["", "RSS 为训练进程内 0.1 秒采样，包含映射页；观测最大值不是精确内存分配峰值。", "",
              "### CPU 评估调度：固定内部验证对照", "",
              "对 ranker 与 GBDT 按候选数量稳定排序后组 batch，分别维持原 batch size 32 和 64，复用冻结的评估器、模型和指标，最后将逐曝光指标恢复为原始行序。三个已选 ranker checkpoint 和所选 GBDT 在同一固定 12,000 组内部验证上，AUC、MRR、nDCG@5/10 的逐组最大绝对差均不超过 1e-12，且 NaN 位置相同；这是指标等价检查，不声称候选 logits、树输入特征或概率逐位相同。权重、数据、特征、核心代码和调度代码摘要均经核验。", "",
              "| 模型 | 所选 epoch | 原序秒数 | 分桶秒数 | 原序计算槽 | 分桶计算槽 | 实际候选曝光 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for audit in scheduling_audits:
        plan = audit["scheduling"]
        lines.append(f"| {audit['run_id']} | {audit['selected_epoch']} | {audit['original']['seconds']:.3f} | {audit['scheduled']['seconds']:.3f} | {plan['original_candidate_slots']:,} | {plan['scheduled_candidate_slots']:,} | {plan['candidate_exposures']:,} |")
    plan = gbdt_scheduling_audit["scheduling"]
    lines.append(f"| GBDT rich | — | {gbdt_scheduling_audit['original']['seconds']:.3f} | {gbdt_scheduling_audit['scheduled']['seconds']:.3f} | {plan['original_candidate_slots']:,} | {plan['scheduled_candidate_slots']:,} | {plan['candidate_exposures']:,} |")
    lines += ["", "计算槽为各 batch 的样本数乘该 batch 最大候选数后求和，包含 padding；实际候选不删减。每个 checkpoint 只有一次原序→分桶的配对观测，计时覆盖整个评估调用，共享桌面负载与缓存热度未受控，不能将该耗时差外推为通用加速倍数或分布式加速。", "",
              "## 全候选排名", "", "每组所有候选均参与评估，无测试负采样。神经模型均使用内部验证选择的 epoch；三种子结果为均值 ± 样本标准差，单次基线不填标准差。", "",
              "| 模型 | AUC | MRR | nDCG@5 | nDCG@10 |", "|---|---:|---:|---:|---:|"]
    for name in ["popularity", "content"]:
        values = ranking["runs"][name]["metrics"]
        lines.append("| " + name + " | " + " | ".join(f"{values[m]['mean']:.5f}" for m in METRICS) + " |")
    lines.append("| GBDT | " + " | ".join(f"{tree['metrics'][m]['mean']:.5f}" for m in METRICS) + " |")
    for name in variants:
        values = ranking["seed_summary"][name]
        lines.append("| " + name + " | " + " | ".join(f"{values[m]['mean']:.5f} ± {values[m]['std']:.5f}" for m in METRICS) + " |")
    lines += ["", "### 相对随机负例双塔的配对差值", "",
              "按用户聚类 bootstrap 500 次；比较的是先对种子取均值的每曝光指标，未集成模型分数。该区间反映测试用户抽样，不包含全部训练不确定性。", "",
              "| 比较模型 | Δ AUC（95% CI，pp） | Δ nDCG@10（95% CI，pp） |", "|---|---:|---:|"]
    for name, values in comparisons.items():
        cells = []
        for metric in ["auc", "ndcg10"]:
            value = values[metric]
            cells.append(f"{value['delta'] * 100:+.3f} [{value['ci95'][0] * 100:+.3f}, {value['ci95'][1] * 100:+.3f}]")
        display_name = "mean_mixed（按验证选择 epoch）" if name == "mean_mixed" else name
        lines.append(f"| {display_name} | " + " | ".join(cells) + " |")
    hard = comparisons["mean_mixed"]["ndcg10"]
    if hard["ci95"][0] > 0:
        conclusion = "混合负例训练方案（按验证选择 epoch）在本设置下的 nDCG@10 高于同样按验证选择 epoch 的随机负例方案，用户聚类差值区间为正。"
    elif hard["ci95"][1] < 0:
        conclusion = "混合负例训练方案（按验证选择 epoch）在本设置下的 nDCG@10 低于同样按验证选择 epoch 的随机负例方案；不能把它写成有效优化。后续可在新的内部实验中研究假负例、样本难度与训练稳定性。"
    else:
        conclusion = "混合负例训练方案（按验证选择 epoch）相对随机负例方案的 nDCG@10 差值区间包含零，本次实验不足以证明稳定收益。"
    conclusion += " " + checkpoint_explanation
    lines += ["", conclusion, "", "### 用户与文章分组", "",
              "下表为 nDCG@10。训练未见文章组表示曝光中至少一个点击文章在训练截止后才首次出现；各组存在重叠。", "",
              "| 分组 | 曝光组 | 随机负例双塔 | 混合负例双塔 | 注意力重排 | GBDT |", "|---|---:|---:|---:|---:|---:|"]
    group_names = ranking["runs"][f"mean_random-seed{seeds[0]}"]["groups"]
    for group, detail in group_names.items():
        values = []
        for variant in variants:
            selected = [ranking["runs"][f"{variant}-seed{seed}"]["groups"][group]["metrics"] for seed in seeds]
            values.append(f"{np.mean([x['ndcg10']['mean'] for x in selected]):.5f}" if selected[0] else "—")
        metric = tree["groups"][group]["metrics"]
        values.append(f"{metric['ndcg10']['mean']:.5f}" if metric else "—")
        lines.append(f"| {group} | {detail['impressions']:,} | " + " | ".join(values) + " |")
    lines += ["", "## 全目录日志点击召回", "",
              f"固定抽样 {retrieval['sample_impressions']:,} 个官方 dev 曝光组；目录随请求时间过滤。目录大小范围为 {retrieval['catalogue_size_min_max']}。使用内部验证选择的 `{retrieval['selected_retriever']}` 与 `{retrieval['selected_ranker']}`。",
              f"共 {retrieval['positive_clicked_items']:,} 个点击文章，其中 {retrieval['repeat_clicks_excluded_by_service']:,} 个被已读规则排除；同时报告原始分母和可召回分母。", "",
              "| 策略 | 全点击 Recall@10 | 全点击 Recall@100 | 可召回点击 Recall@100 | 新文章可召回点击 Recall@100 |", "|---|---:|---:|---:|---:|"]
    for name, values in retrieval["metrics"].items():
        lines.append(f"| {name} | " + " | ".join(f"{values[key]:.5f}" for key in ["click_micro_recall10", "click_micro_recall100", "eligible_click_micro_recall100", "cold_eligible_click_recall100"]) + " |")
    lines += ["", "全目录中未曝光新闻没有真实负标签。这些是日志点击召回指标，不能解释为在线满意度。对同一个 100 候选集合重排不改变 Recall@100。", "",
              "retriever/ranker checkpoint 按内部验证选择，但暖用户默认 two-stage 是预先指定的实验架构，没有在内部验证或 dev 上对服务策略择优；上表并列报告 retrieval-only 与 two-stage 效果，不能看完 dev 后改变默认策略并仍声称 dev 未参与选择。", "",
              "## CPU 检索与真实 HTTP", "",
              f"在 {ann['queries']:,} 个内部验证查询、{ann['faiss_threads']} 个 Faiss 线程下，ANN 对精确检索的平均邻居 Recall@{ann['k']} 为 **{ann['mean_ann_recall_at_k']:.5f}**；精确检索 p95 **{ann['exact_latency']['p95_ms']:.3f} ms**，HNSW p95 **{ann['ann_latency']['p95_ms']:.3f} ms**。计时包含检索及过滤，不含用户编码；该近似率与上面的点击召回分开。", "",
              "| HTTP 批大小 | 客户端并发 | 请求 p50 ms | 请求 p95 ms | 推荐查询/秒 | 错误数 |", "|---:|---:|---:|---:|---:|---:|"]
    for case in benchmark["http"]["cases"]:
        lines.append(f"| {case['http_batch_size']} | {case['client_concurrency']} | {case['request_latency']['p50_ms']:.2f} | {case['request_latency']['p95_ms']:.2f} | {case['recommendation_queries_per_second']:.2f} | {case['error_count']} |")
    lines += ["", "批请求延迟不是单条查询延迟。测量经过预热，实际经过本地 HTTP、请求解析、检索和重排；单服务进程、模型锁控制并发访问。共享 Windows 桌面上的其他任务未停止，CPU 负载不可控，数字仅代表本次环境。完整硬件、线程、负载、模型/索引摘要见 JSON。", "",
              "## 可复核证据", "", "- `ranking_experiment.json`：完整分组指标、三种子汇总、源文件摘要。",
              "- 各模型 `*_training.json`：每轮数据量、负例策略、loss、内部验证与耗时。",
              "- `gbdt_feature_ablation.json`：dev 锁定阶段新增的两种特征对照、原结果保留及正式模型选择。",
              "- `evaluation_scheduler*validation_12000.json`：三个 ranker 的内部等价性、原序/分桶时间与计算槽；最终排序报告保存对应证据摘要。",
              "- `gbdt_evaluation_scheduler_validation_12000.json`：所选 GBDT 的对应配对证据；最终 GBDT 报告绑定模型、特征与调度摘要。",
              "- `data_manifest.json` / `data_provenance.json` / `training_cache.json`：数据、来源与采样摘要的可分发副本。",
              "- `service_benchmark.json`：真实请求计数、错误、延迟、吞吐、内存与 ANN 校验。",
              "- `unit_tests.txt`：数据边界、采样、模型与服务测试的实际输出。", ""]
    (reports / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    for source, name in [("manifest.json", "data_manifest.json"), ("provenance.json", "data_provenance.json"), ("training_cache.json", "training_cache.json")]:
        shutil.copyfile(data / source, reports / name)
    random_result = ranking["seed_summary"]["mean_random"]["ndcg10"]
    content_result = ranking["runs"]["content"]["metrics"]["ndcg10"]["mean"]
    cv = f"""# 简历写法与证据边界

项目名称：**NewsForge：基于真实曝光日志的两阶段新闻推荐研究与推理服务**

- 构建 MIND-large 新闻推荐实验管线，流式处理 {manifest['impressions'] / 1e4:.1f} 万曝光组、{manifest['candidates'] / 1e4:.1f} 万候选记录；实现时间切分、仅训练集拟合文本特征及全候选评估。
- 实现双塔召回、候选相关注意力重排及 GBDT 对照，在 {manifest['official_dev_rows']:,} 组完整 dev 上完成神经模型三种子评估；随机负例双塔 nDCG@10 为 {random_result['mean']:.4f} ± {random_result['std']:.4f}，固定内容相似度基线为 {content_result:.4f}。
- 开发 FAISS HNSW + FastAPI 服务，支持批量推理、时效与已读过滤及冷启动回退；本机 CPU 单请求串行 p95 {single['request_latency']['p95_ms']:.2f} ms、{single['recommendation_queries_per_second']:.2f} 查询/秒，ANN 邻居 Recall@{ann['k']} 为 {ann['mean_ann_recall_at_k']:.4f}。
- 可选：优化 CPU 离线评估分批，在固定 {gbdt_scheduling_audit['rows']:,} 个内部曝光上，将 GBDT 候选计算槽从 {gbdt_scheduling_audit['scheduling']['original_candidate_slots']:,} 减至 {gbdt_scheduling_audit['scheduling']['scheduled_candidate_slots']:,}，四项逐组指标在 1e-12 容差内等价；此次原序/分桶耗时为 {gbdt_scheduling_audit['original']['seconds']:.2f}/{gbdt_scheduling_audit['scheduled']['seconds']:.2f} 秒。该结果为共享桌面单次配对，不代表通用或生产加速。

以上四条可选，投递时优先保留前三条；岗位更重视 CPU 性能时可用第四条替换。面试时展示 `reports/RESULTS.md`、原始 JSON 和复现命令。

统计口径：双塔数值是预先指定的随机负例方案在三个种子上的均值 ± 样本标准差，不是选取 dev 最佳种子；各神经模型的 epoch 均按内部验证选择，dev 不参与选择。内容相似度为固定基线，GBDT 使用不同训练预算。随机/混合负例及注意力模型的完整对照、按用户聚类 bootstrap 区间见实测报告。

服务性能口径：单请求串行数据来自 {single['measured_http_requests']:,} 次本地 HTTP 请求，ANN 校验使用 {ann['queries']:,} 个内部验证查询；共享 Windows 桌面的 CPU 负载未受控，延迟与吞吐只代表本次环境，不是生产 SLO。

项目身份是个人开源研究项目，以上表述只能在你能解释并演示代码与实验后使用。不能写成腾讯生产业务、真实线上 CTR 提升、在线 A/B、分布式集群加速、预训练语言模型微调或完整 NRMS 复现；本次实现和实测未覆盖这些内容。

混合负例训练方案的实验结论：{conclusion}

岗位对应：用户行为建模与本报告规模的日志处理；PyTorch 模型/优化与实验设计；CPU 检索和批量推理；本机推荐服务实现。分布式训练与在线实验属于后续扩展，当前不列为完成经历。
"""
    (root / "docs/RESUME_CN.md").write_text(cv, encoding="utf-8")
    included = [root / name for name in ["README.md", "LICENSE", "requirements.txt", "requirements-windows.lock", ".gitignore"]]
    for directory, suffixes in [("newsforge", {".py"}), ("tests", {".py"}), ("docs", {".md"}), ("reports", {".json", ".md", ".txt", ".png"})]:
        included += [path for path in (root / directory).glob("*") if path.is_file() and path.suffix in suffixes]
    bundle = root / "newsforge-source-and-results.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(included):
            archive.write(path, Path("newsforge") / path.relative_to(root))
    print(json.dumps({"report": str(reports / "RESULTS.md"), "resume": str(root / "docs/RESUME_CN.md"),
                      "bundle": str(bundle), "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
