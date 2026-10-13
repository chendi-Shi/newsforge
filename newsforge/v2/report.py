"""Render Chinese v2 results from complete, mutually bound aggregate evidence.

This module uses only the standard library. It does not load data or models,
run experiments, fill missing results, or infer an improvement from its design.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


VARIANTS = ("content_only", "context_only", "full")
METRICS = ("auc", "mrr", "ndcg5", "ndcg10")
CATALOG_NAMES = ("full", "no_popular", "no_content", "no_tower", "popular_only",
                 "content_only", "tower_only", "no_freshness", "frozen_feedback", "selected")
LABELS = {"full": "全部模块", "no_popular": "去热度通道", "no_content": "去内容通道",
          "no_tower": "去双塔通道", "popular_only": "仅热度", "content_only": "仅内容",
          "tower_only": "仅双塔", "no_freshness": "取消时效窗口", "frozen_feedback": "冻结训练反馈",
          "selected": "内部选定候选（不重排）", "context_only": "仅时间上下文"}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant {value}: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def number(value, digits=6):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Report metrics must be finite")
    return f"{value:.{digits}f}"


def percent(value):
    return f"{number(float(value) * 100, 4)}%"


def delta(value):
    return f"{float(value) * 100:+.4f}"


def interval(value):
    low, high = value["ci95"]
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError("Invalid confidence interval")
    return f"{delta(value['delta'])} [{delta(low)}, {delta(high)}]"


def verdict(value):
    low, high = value["ci95"]
    if low > 0:
        return "支持正向贡献"
    if high < 0:
        return "观察到负向贡献"
    if low == high == 0 and value["delta"] == 0:
        return "该样本差异为零"
    return "区间跨零，未支持稳定增益"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    result = ["| " + " | ".join(map(cell, headers)) + " |",
              "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(result)


def validate_cache(cache):
    """Check the audit schema without loading the model or request arrays."""
    observations = cache["observations"]
    if (cache["queries"] != 16 or cache["candidate_batches_compared"] != 576
            or set(cache["recipes"]) != set(CATALOG_NAMES) - {"selected"}
            or [value["reuse_query_routes"] for value in observations] != [False, True, True, False]
            or [value["observation"] for value in observations] != [1, 2, 3, 4]
            or any(value["batches_compared"] != 144 for value in observations)):
        raise ValueError("Query cache audit does not contain the declared 16-query ABBA protocol")
    if (cache["passed"] is not True or cache["mismatch_count"] != 0
            or cache["max_absolute_score_error"] != 0
            or cache["official_dev_metrics_evaluated"] is not False
            or cache["catalogue_and_query_caches_cleared_before_each_observation"] is not True):
        raise ValueError("Query cache exact-equivalence audit failed")
    expected_fields = {"ids", "scores", "source_mask", "sources", "snapshot_index", "cutoff",
                       "as_of", "shortfall", "metadata"}
    if set(cache["field_names_compared"]) != expected_fields:
        raise ValueError("Query cache audit did not compare all CandidateBatch fields")
    for value in observations:
        if (not math.isfinite(value["seconds"]) or value["seconds"] <= 0
                or not isinstance(value["matrix_scoring_calls"], int) or value["matrix_scoring_calls"] < 0):
            raise ValueError("Invalid query cache timing/call-count observation")
    uncached_calls = sum(value["matrix_scoring_calls"] for value in observations if not value["reuse_query_routes"])
    cached_calls = sum(value["matrix_scoring_calls"] for value in observations if value["reuse_query_routes"])
    if not uncached_calls or not math.isclose(1 - cached_calls / uncached_calls, cache["matrix_scoring_call_reduction"], abs_tol=1e-12):
        raise ValueError("Query cache call-count reduction is inconsistent")


def cache_section(cache):
    validate_cache(cache)
    observations = cache["observations"]
    uncached_seconds = sum(value["seconds"] for value in observations if not value["reuse_query_routes"])
    cached_seconds = sum(value["seconds"] for value in observations if value["reuse_query_routes"])
    timing_finding = ("这四次观测中缓存组总耗时更低，但样本不足以给出稳定加速结论。"
                      if cached_seconds < uncached_seconds else
                      "这四次观测中缓存组总耗时没有降低，未显示耗时收益；不能把调用减少换算成加速倍数。")
    rows = [[value["observation"], "启用缓存（B）" if value["reuse_query_routes"] else "未缓存（A）",
             value["batches_compared"], value["matrix_scoring_calls"], number(value["seconds"], 3)]
            for value in observations]
    return ["## 消融调度缓存：等价验证与实际计时", "",
            f"从内部验证固定抽取 {cache['queries']} 个请求（seed={cache['seed']}），对 9 个候选方案按 ABBA 顺序执行，"
            f"共逐字段比较 **{cache['candidate_batches_compared']} 个 CandidateBatch**。"
            f"不一致 {cache['mismatch_count']} 个，最大分数绝对误差 {number(cache['max_absolute_score_error'])}；"
            "ID、分数、dtype、来源、mask、时间截止、shortfall 和 metadata 均精确一致。"
            "这项审计没有做 dev 相关性评估，也不参与选模型。", "",
            table(["观测顺序", "模式", "比较批数", "矩阵评分 _dot 调用数", "整轮耗时（秒）"], rows), "",
            f"两次缓存与两次未缓存观测的 `_dot` 调用数合计减少 **{percent(cache['matrix_scoring_call_reduction'])}**。"
            + timing_finding, "",
            "每次观测前清除目录与请求路由缓存，另有一次不计时的相同请求预热；"
            "表中计时包含整轮检索、精确等价检查和调用计数 instrumentation。"
            "这里只复用同一请求、同一语义路由的 top IDs，热度反馈、融合与过滤仍按方案执行。"
            "它改变实验调度，不改变推荐结果；少量共享桌面的观测不是稳定性能保证。"
            "HTTP 服务默认关闭该请求路由复用，不能把此处的调用减少或任何整轮计时解释为服务延迟提升。", ""]


def validate_mature_selection(evidence, paths):
    mature, development = evidence["ranker-selection"], evidence["ranker-development-selection"]
    evaluation = evidence["ranker-matured-internal-evaluation"]
    audit, reuse = mature["maturity_audit"], mature["weight_reuse_audit"]
    if evaluation["maturity_audit"] != audit or evaluation["weight_reuse_audit"] != reuse:
        raise ValueError("Mature selection and evaluation audits differ")
    if mature["mature_internal_evaluation_sha256"] != sha256(paths["ranker-matured-internal-evaluation"]):
        raise ValueError("Mature internal evaluation SHA mismatch")
    expected_window = {"validation_min_inclusive": 1573722000, "validation_max_exclusive": 1573772400,
                       "first_dev_timestamp": 1573776000, "first_dev_snapshot_cutoff": 1573772400,
                       "sampled_rows": 6000}
    if any(audit[name] != value for name, value in expected_window.items()):
        raise ValueError("Ranker selection window does not precede the first dev feedback cutoff")
    if (not 1573722000 <= audit["sampled_timestamp_min"] <= audit["sampled_timestamp_max"] < 1573772400
            or audit["all_rows_split1"] is not True
            or audit["all_selection_feedback_strictly_before_first_dev_cutoff"] is not True
            or audit["row_indices_sha256"] != mature["validation_row_indices_sha256"]):
        raise ValueError("Mature ranker validation population audit failed")
    source_sha = sha256(paths["ranker-development-selection"])
    if source_sha != mature["source_selection_sha256"] or source_sha != reuse["source_selection_sha256"]:
        raise ValueError("Mature selection does not bind the retained development record")
    old = {value["run_id"]: value for value in development["models"]}
    new = {value["run_id"]: value for value in mature["models"]}
    identities = ("variant", "seed", "model_path", "model_sha256")
    if set(old) != set(new) or len(new) != 9:
        raise ValueError("Mature re-evaluation requires the same nine trained models")
    if any(old[key][field] != new[key][field] for key in old for field in identities):
        raise ValueError("A model identity changed during mature-window re-evaluation")
    hashes = {key: value["model_sha256"] for key, value in old.items()}
    if (reuse["fit_invoked"] is not False or reuse["all_weights_unchanged"] is not True
            or reuse["fixed_max_iter"] != 80 or reuse["early_stopping"] is not False
            or reuse["model_sha256_before"] != hashes or reuse["model_sha256_after"] != hashes):
        raise ValueError("Mature selection weight reuse audit failed")


def validate_backbone(evidence, paths, bindings):
    report = evidence["backbone-matured-selection"]
    population = report["validation"]
    if (population["split"] != 1 or population["impressions"] != 6000
            or population["requested_impressions"] != 6000 or population["seed"] != 2026
            or population["min_timestamp_inclusive"] != 1573722000
            or population["max_timestamp_exclusive"] != 1573772400
            or not 1573722000 <= population["observed_min_timestamp"] <= population["observed_max_timestamp"] < 1573772400):
        raise ValueError("Backbone selection must use the mature 6000-request internal window")
    if population["row_indices_sha256"] != evidence["ranker-selection"]["validation_row_indices_sha256"]:
        raise ValueError("Backbone and ranker internal selection populations differ")
    candidates = report["candidates"]
    if sorted(value["seed"] for value in candidates) != [17, 29, 43]:
        raise ValueError("Backbone selection requires the three declared fixed-budget seeds")
    for value in candidates:
        if (value["epoch"] != 2 or value["state_matches_epoch2"] is not True
                or value["complete_training_epochs"] != 2
                or value["epoch_state_sha256"][1] != value["state_sha256"]):
            raise ValueError("Backbone checkpoint is not the fixed final epoch-2 state")
    winner = max(candidates, key=lambda value: (value["metrics"]["ndcg10"]["mean"], -value["seed"]))
    selected = report["selected"]
    if any(selected[key] != winner[key] for key in selected):
        raise ValueError("Backbone winner does not follow mature nDCG10 selection and seed tie break")
    current = report["current_backbone"]
    if (report["matches_current_retriever_state"] is not True
            or selected["state_sha256"] != current["state_sha256"]):
        raise ValueError("Serving backbone is not the mature-selected parameter state")
    if current["checkpoint_sha256"] != bindings["retrieval_artifacts"]["retriever"]:
        raise ValueError("Serving backbone artifact differs from the audited current retriever")
    if (sha256(paths["backbone-matured-selection"]) != bindings["backbone_matured_selection"]
            or report["data_manifest_sha256"] != bindings["data_manifest"]
            or report["source_sha256"] != bindings["source"]["select_backbone.py"]):
        raise ValueError("Mature backbone audit is not bound to the final data/source/selection")


def validate(evidence, paths):
    validate_mature_selection(evidence, paths)
    selection = evidence["select"]["frozen_selection"]
    bound = selection["bindings"]
    validate_backbone(evidence, paths, bound)
    for name in ("catalog", "ranking", "ranking-full"):
        result = evidence[name]
        if result["bindings"] != bound:
            raise ValueError(f"{name} bindings differ from the frozen internal selection")
        if result["selected_before_v2_dev"] != selection["created_utc"]:
            raise ValueError(f"{name} selection timestamp mismatch")
    expected_selection_sha = evidence["catalog"]["selection_sha256"]
    for name in ("ranking", "ranking-full", "service_benchmark"):
        if evidence[name]["selection_sha256"] != expected_selection_sha:
            raise ValueError(f"{name} selection SHA mismatch")
    health = evidence["service_benchmark"]["health"]
    if health["bindings"]["selection_sha256"] != expected_selection_sha:
        raise ValueError("HTTP health selection SHA mismatch")
    if evidence["service_benchmark"]["processed_manifest_sha256"] != bound["data_manifest"]:
        raise ValueError("HTTP data manifest mismatch")
    if health["bindings"]["temporal_manifest_sha256"] != bound["temporal_manifest"]:
        raise ValueError("HTTP temporal manifest mismatch")
    if health["candidate_config"] != selection["serving"]["candidate_config"]:
        raise ValueError("HTTP candidate recipe differs from the internal selection")
    if sha256(paths["ranker-selection"]) != bound["ranker_selection"]:
        raise ValueError("Ranker internal selection evidence SHA mismatch")
    if evidence["ranker-selection"]["selected"] != selection["ranking_selected"]:
        raise ValueError("Full ranking model differs from internal selection")
    cache = evidence["query_cache_equivalence"]
    validate_cache(cache)
    if sha256(paths["query_cache_equivalence"]) != bound["query_cache_equivalence"]:
        raise ValueError("Query cache audit differs from the frozen selection binding")
    for name in ("experiment.py", "retrieval.py", "temporal.py"):
        if cache["source_sha256"][name] != bound["source"][name]:
            raise ValueError(f"Cache audit source differs from selection: {name}")
    for name, value in cache["input_array_sha256"].items():
        if value != bound["processed_files"][name]:
            raise ValueError(f"Cache audit population differs from selection: {name}")
    for name in ("data_manifest", "temporal_manifest"):
        if cache["bindings"][name] != bound[name]:
            raise ValueError(f"Cache audit input differs from selection: {name}")
    for source, target in (("embeddings", "article_embeddings"), ("retriever", "retriever"),
                           ("encoder", "encoder"), ("encoder_manifest", "encoder_manifest")):
        if cache["bindings"][source] != bound["retrieval_artifacts"][target]:
            raise ValueError(f"Cache audit model differs from selection: {source}")
    for result in [evidence["catalog"], *(evidence["select"][key] for key in
                     ("window_selection", "candidate_ablation", "reranker_selection"))]:
        schedule = result["execution_schedule"]
        if schedule["reuse_current_query_semantic_routes"] is not True or schedule["service_default_reuses_query_routes"] is not False:
            raise ValueError("Unexpected experiment versus serving cache policy")
    expected_catalog = set(CATALOG_NAMES) | {f"selected_rerank_{variant}" for variant in VARIANTS}
    if set(evidence["catalog"]["metrics"]) != expected_catalog:
        raise ValueError("Missing or unexpected catalogue ablation recipes")
    selected_pool = evidence["catalog"]["metrics"]["selected"]
    for variant in VARIANTS:
        reranked = evidence["catalog"]["metrics"][f"selected_rerank_{variant}"]
        for key in ("hits100", "click_micro_recall100", "eligible_click_micro_recall100",
                    "cold_eligible_recall100", "zero_train_click_eligible_recall100"):
            if reranked[key] != selected_pool[key]:
                raise ValueError("Reranking changed the candidate pool or recall100")
    internal_rows = evidence["select"]["window_selection"]["row_indices_sha256"]
    if selection["selection_label_time_bounds"] != {
            "minimum_inclusive": 1573722000, "maximum_exclusive": 1573772400}:
        raise ValueError("Candidate selection label window is not strictly mature before dev")
    for phase in ("window_selection", "candidate_ablation", "reranker_selection"):
        value = evidence["select"][phase]
        low, high = value["query_time_min_max"]
        if (not 1573722000 <= low <= high < 1573772400 or value["queries"] != 2000
                or value["query_split_counts"] != {"1": 2000}):
            raise ValueError(f"Candidate selection uses unavailable or non-internal feedback: {phase}")
    for phase in ("candidate_ablation", "reranker_selection"):
        if evidence["select"][phase]["row_indices_sha256"] != internal_rows:
            raise ValueError("Internal candidate selection populations differ")
    expected_models = {f"{variant}-seed{seed}" for variant in VARIANTS for seed in (17, 29, 43)}
    if set(evidence["ranking"]["metrics"]) != expected_models:
        raise ValueError("Ranking ablation requires all nine models")
    selected = selection["ranking_selected"]
    selected_name = f"{selected['variant']}-seed{selected['seed']}"
    if set(evidence["ranking-full"]["metrics"]) != {selected_name}:
        raise ValueError("Full dev ranking contains a model other than the selected model")
    if (evidence["catalog"]["queries"] != 5000 or evidence["ranking"]["impressions"] != 12000
            or evidence["ranking-full"]["impressions"] != 376471
            or evidence["ranking-full"]["candidates"] != 14085557):
        raise ValueError("Population differs from the preregistered v2 benchmark sizes")
    if set(evidence["ranking"]["seed_statistics"]) != set(VARIANTS):
        raise ValueError("Missing three-seed statistics")
    if set(evidence["ranking"]["paired_user_intervals"]) != {
            "full_minus_content_only", "full_minus_context_only", "context_only_minus_content_only"}:
        raise ValueError("Missing paired ranking comparisons")
    configs = evidence["service_benchmark"]["configurations"]
    if {(value["batch_size"], value["concurrency"]) for value in configs} != {(1, 1), (8, 1), (1, 4), (8, 4)}:
        raise ValueError("Missing HTTP benchmark configuration")
    if len(configs) != 4:
        raise ValueError("Duplicate HTTP benchmark configuration")
    for value in configs:
        if value["rounds"] != 3 or value["warmup_http_requests"] != 8 or value["measured_queries"] != 192:
            raise ValueError("HTTP benchmark differs from the declared 64-query protocol")


def render(evidence, paths, output):
    validate(evidence, paths)
    select, catalog, ranking, full, service = (evidence[name] for name in
                                               ("select", "catalog", "ranking", "ranking-full", "service_benchmark"))
    frozen, internal_ranker = select["frozen_selection"], evidence["ranker-selection"]
    backbone = evidence["backbone-matured-selection"]
    selected_name = frozen["candidate_selected_name"]
    serving = frozen["serving"]
    selected_ranker = frozen["ranking_selected"]
    ranker_name = f"{selected_ranker['variant']}-seed{selected_ranker['seed']}"
    serve_variant = serving["ranker_identity"]["variant"] if serving["use_reranker"] else None
    serving_catalog_key = f"selected_rerank_{serve_variant}" if serve_variant else "selected"
    final_catalog = catalog["metrics"][serving_catalog_key]
    full_metrics = full["metrics"][ranker_name]
    total, eligible, cold = (catalog[key] for key in
                             ("positive_clicked_items", "eligible_positive_items", "cold_eligible_positive_items"))
    lines = ["# NewsForge v2：可复核的时间回放、召回消融与 CPU 服务结果", "",
             f"内部冻结候选为 `{selected_name}`，服务重排为 `{serve_variant or '关闭'}`；"
             f"5,000 个 dev 请求的全点击 Recall@100 为 **{percent(final_catalog['click_micro_recall100'])}**，"
             f"Recall@10 为 **{percent(final_catalog['click_micro_recall10'])}**。"
             f"内部选出的独立曝光排序模型 `{ranker_name}` 在全部 {full['impressions']:,} 个 dev 曝光组上，"
             f"nDCG@10 为 **{number(full_metrics['ndcg10']['mean'])}**。候选召回、曝光内排序和 HTTP 性能是三种不同测量。", "",
             "本轮是在看过 v1 dev 结果之后开展的后续研究。窗口、候选和模型仅用内部验证选择，"
             "随后冻结配置并完整报告预定 dev 消融；这不能恢复 dev 为从未使用的独立测试集。"
             "滚动方法还使用已成熟的过去验证/dev 反馈，因此与 v1 冻结反馈的数值差不能直接归因为相同信息下的算法改进。", "",
             "## 时间与实验口径", "",
             "请求时刻 t 对应整点 T=floor(t/3600)×3600，统计及目录截止 C=T−3600；"
             "只允许 source_timestamp<C 的事件，first_seen<C 的文章。时间无源时区，UTC 只是确定性排序坐标。"
             "首次观察年龄不是发布时间，至少 1 小时并按整点成熟的反馈是回放假设。EMA 半衰期 12 小时，事件按小时结束聚合。", "",
             "三种排序输入为内容 15 维、时间上下文 7 维及合并 22 维；共享 100,000 个训练正例，"
             "各配 4 个同曝光负例，共 500,000 行。固定 HGB 80 轮、15 叶、63 bins、学习率 0.1、L2=1，"
             "关闭提前停止，单 CPU 线程。seed 为 17/29/43；这是分箱等随机性的重复，不能称为神经网络初始化。"
             "1:4 采样输出只用于排序，不能解释为总体校准 CTR。", "",
             "## 内部验证与冻结选择", "",
             f"最终内部选择只使用 2019-11-14 09:00:00 ≤ t < 23:00:00（1573722000 ≤ t < 1573772400）的请求。"
             f"候选用 {select['window_selection']['queries']:,} 行，"
             "曝光排序用 6,000 行，抽样 seed=2026。选择依据分别为全点击 Recall@100、重排后的 Recall@10、曝光内 nDCG@10；"
             "不把 dev 最优项替换成部署选择。", "",
             "最早 dev 请求为 11 月 15 日 00:00，对应严格反馈 cutoff 为前一日 23:00；"
             "23:00 及之后的验证标签在这时尚未成熟，不能用于选择服务模型。"
             "早期含晚间验证请求的结果保留为开发记录，不作为最终选择。"
             "九个已训练模型采用固定 80 轮、early_stopping=False，成熟窗口阶段只重新评分和选择，"
             "不重新拟合；完整模型身份与权重摘要保持不变。", ""]
    backbone_rows = [[value["seed"], value["epoch"], *[number(value["metrics"][metric]["mean"]) for metric in METRICS],
                      "最终骨干" if value["seed"] == backbone["selected"]["seed"] else ""]
                     for value in backbone["candidates"]]
    lines += [table(["双塔 seed", "固定 epoch", "AUC", "MRR", "nDCG@5", "nDCG@10", "成熟窗口选择"], backbone_rows), "",
              "双塔也在同一成熟 6,000 请求窗口内重新选择。三个候选必须各自等于训练日志中第二个完整 epoch 的实际参数，"
              "不根据旧晚间验证分数重选 epoch，也不再训练；最终骨干 state SHA 与服务读取的 retriever 参数一致。"
              "同一参数以不同 torch 文件封装时文件 SHA 可以不同，所以先比较 state SHA，再将实际服务文件 SHA 绑定到选择。", ""]
    window_rows = [[name, percent(value["click_micro_recall100"]), value["hits100"]]
                   for name, value in select["window_selection"]["metrics"].items()]
    lines += [table(["内部时效窗口", "Recall@100", "命中"], window_rows), "",
              f"冻结消融窗口：**{frozen['ablation_window_hours']} 小时**。", ""]
    candidate_rows = [[LABELS.get(name, name), percent(value["click_micro_recall100"]),
                       "选中" if name == selected_name else ""]
                      for name, value in select["candidate_ablation"]["metrics"].items()]
    lines += [table(["内部候选方案", "Recall@100", "选择"], candidate_rows), ""]
    rerank_rows = [[name, percent(value["click_micro_recall10"]), "选中" if name == serving_catalog_key else ""]
                  for name, value in select["reranker_selection"]["metrics"].items()]
    lines += [table(["内部选定候选上的重排", "Recall@10", "服务选择"], rerank_rows), ""]
    model_rows = [[entry["run_id"], *[number(entry["metrics"][metric]["mean"]) for metric in METRICS],
                   "全量曝光评估选中" if entry["run_id"] == ranker_name else ""]
                  for entry in internal_ranker["models"]]
    lines += [table(["内部排序模型", "AUC", "MRR", "nDCG@5", "nDCG@10", "选择"], model_rows), "",
              table(["最终内部选择审计", "值"], [
                  ["6,000 排序请求的实际时间 min/max", " / ".join(str(internal_ranker["maturity_audit"][key])
                                                         for key in ("sampled_timestamp_min", "sampled_timestamp_max"))],
                  ["2,000 候选请求的实际时间 min/max", " / ".join(map(str, select["window_selection"]["query_time_min_max"]))],
                  ["首个 dev 请求 / 其严格反馈 cutoff", "1573776000 / 1573772400"],
                  ["九模型重新拟合", str(internal_ranker["weight_reuse_audit"]["fit_invoked"]).lower()],
                  ["九模型权重 SHA 全部不变", str(internal_ranker["weight_reuse_audit"]["all_weights_unchanged"]).lower()],
              ]), "",
              f"冻结时间：`{frozen['created_utc']}`；配置 SHA256：`{catalog['selection_sha256']}`。", "",
              "## 5,000 请求的全目录召回消融", "",
              f"所有方案共享 {total:,} 个日志正点击主分母；严格截止且去已读后的 eligible 点击为 {eligible:,}，"
              f"其中训练截止后首次观察的 cold 点击为 {cold:,}，零训练点击新闻为 {catalog['zero_train_click_eligible_positive_items']:,}。"
              f"已读正点击 {catalog['repeat_clicks']:,} 个；总不可召回点击 {total-eligible:,} 个（目录或已读规则的并集）。"
              f"目录大小范围 {catalog['catalogue_size_min_max'][0]:,}–{catalog['catalogue_size_min_max'][1]:,}。"
              "时效窗口变窄不缩小主要分母。", ""]
    rows = []
    for name in (*CATALOG_NAMES, *(f"selected_rerank_{variant}" for variant in VARIANTS)):
        value = catalog["metrics"][name]
        cold_hits = round(value["cold_eligible_recall100"] * cold)
        if cold and not math.isclose(cold_hits / cold, value["cold_eligible_recall100"], abs_tol=1e-12):
            raise ValueError("Cold recall is inconsistent with its integer hit denominator")
        rows.append([name, percent(value["click_micro_recall10"]), percent(value["click_micro_recall100"]),
                     f"{value['hits10']:,} / {value['hits100']:,}", percent(value["eligible_click_micro_recall100"]),
                     f"{percent(value['cold_eligible_recall100'])} ({cold_hits:,}/{cold:,})",
                     percent(value["zero_train_click_eligible_recall100"]), value.get("shortfall_queries", "同 selected 池")])
    lines += [table(["配置", "R@10", "R@100", "命中@10 / @100", "eligible R@100", "cold R@100（命中/分母）",
                     "零训练点击 R@100", "候选不足请求"], rows), "",
              "冷启动命中数由记录的冷启动比例和整数分母还原并做一致性校验。重排只改变同一 100 候选的顺序，"
              "所以三个重排模型的 Recall@100 必须与 selected 相同；新增价值看 Recall@10。", ""]
    comparison_rows = [[name, interval(value), verdict(value)]
                       for name, value in catalog["paired_user_intervals"].items()]
    lines += [table(["配对比较", "Δ（百分点）[95% CI]", "观察"], comparison_rows), "",
              "这里的区间是固定请求、固定主分母下，按用户有放回抽样 1,000 次（seed=2026）的 micro Recall 差。"
              "full−no_* 衡量完整融合中移除一个部件的变化，不是各部件可相加的独立收益；"
              "full−frozen_feedback 包含可用历史反馈数量与滚动衰减方式的变化。以上是多项探索性比较，未校正多重检验。", ""]
    route_rows = [[name, percent(value["route_union_oracle_recall"]),
                   ", ".join(f"{route}={count}" for route, count in value["route_hits_before_fusion"].items()),
                   " / ".join(number(ms, 3) for ms in value["candidate_ms_p50_p95"])]
                  for name, value in catalog["metrics"].items() if "route_hits_before_fusion" in value]
    lines += [table(["候选方案", "各路并集 oracle recall", "融合前各路命中（可重叠）", "评估边际 p50/p95 ms"], route_rows), "",
              "各路最多预取 200 个，融合后固定 100；并集 oracle 的预算更大，仅是候选覆盖上界。"
              "逐方案计时是共享消融循环中的边际耗时：后运行方案可能复用当前请求已计算的语义路由。"
              "因此不能横向比较独立策略速度，也不能当作 HTTP 请求延迟。", ""]
    lines += cache_section(evidence["query_cache_equivalence"])
    lines += [
              "## 12,000 曝光组：内容与上下文的排序消融", "",
              f"共 {ranking['impressions']:,} 个曝光组、{ranking['candidates']:,} 条完整日志候选；"
              "评估阶段没有采样负例。每项是三个 seed 的指标均值 ± 样本 SD（ddof=1），不是均值的置信区间。", ""]
    stats = ranking["seed_statistics"]
    seed_rows = [[variant, *[f"{number(stats[variant][metric]['mean'])} ± {number(stats[variant][metric]['sample_std'])}"
                            for metric in METRICS]] for variant in VARIANTS]
    lines += [table(["特征方案", "AUC", "MRR", "nDCG@5", "nDCG@10"], seed_rows), ""]
    pair_rows = [[name, *[interval(values[metric]) for metric in METRICS]]
                 for name, values in ranking["paired_user_intervals"].items()]
    lines += [table(["配对比较", "Δ AUC pp [CI]", "Δ MRR pp [CI]", "Δ nDCG@5 pp [CI]", "Δ nDCG@10 pp [CI]"], pair_rows), "",
              "三组区间先对同一曝光组的三个 seed 指标取均值，再按用户聚类 bootstrap 1,000 次（seed=2026）；"
              "它衡量评估用户抽样的不确定性，不混合成 seed 置信区间，也不是三个模型分数集成。", "",
              "## 全部 dev 曝光组", "",
              table(["内部选定模型", "曝光组", "候选曝光", "AUC", "MRR", "nDCG@5", "nDCG@10"],
                    [[ranker_name, f"{full['impressions']:,}", f"{full['candidates']:,}",
                      *[number(full_metrics[metric]["mean"]) for metric in METRICS]]]), "",
              "这里只评估内部验证选定的单一模型。它用于固定日志候选的排序任务，"
              "服务采用的候选及重排方案则由另一组内部目录 Recall 指标选择，两者目标不同。", "",
              "## 真实 localhost HTTP 服务", "",
              f"采样 {service['sampling']['queries']} 个内部请求，warmup 8 个 HTTP 请求后测 3 轮。"
              "离线参考 worker 退出后才启动自有 HTTP 服务；两套模型不同时驻内存。"
              "单 worker、单计算线程，urllib 每请求新建连接；延迟包含 JSON 编解码、连接、服务检索与重排。", ""]
    http_rows = [[value["batch_size"], value["concurrency"], value["measured_http_requests"], value["measured_queries"],
                  number(value["latency_ms"]["p50"], 3), number(value["latency_ms"]["p95"], 3),
                  number(value["successful_queries_per_second"], 2), value["http_errors"],
                  value["offline_http_mismatches"], value["queries_with_output_shortfall"],
                  value["queries_with_retrieval_shortfall"]] for value in service["configurations"]]
    lines += [table(["batch", "并发", "HTTP 请求", "query 数", "p50 ms/批", "p95 ms/批", "成功 query/s",
                     "HTTP 错误", "离线不一致", "输出不足", "候选不足"], http_rows), ""]
    resources = service["resources"]
    check_count = len(service["api_contract_checks"])
    passed = sum(check["passed"] for check in service["api_contract_checks"])
    lines += [f"观测服务进程峰值 RSS **{number(resources['server_peak_rss_bytes'] / 2**20, 2)} MiB**；"
              f"系统最低可用内存 {number(resources['minimum_system_available_bytes'] / 2**20, 2)} MiB。"
              f"每 {resources['sample_interval_seconds']} 秒采样，共 {resources['resource_samples']:,} 次，"
              "覆盖服务健康后的 API 校验和压测，排除启动阶段和客户端；采样可能遗漏瞬时峰值。", "",
              f"API 契约通过 **{passed}/{check_count}**；整体报告 passed=`{str(service['passed']).lower()}`。"
              "检查包括 PAD/OOV、时间覆盖外拒绝、覆盖端点、整点前后 cutoff、非法 top_k/批大小及空历史。"
              "逐请求对比推荐 ID、来源、分数与配置/快照/模型版本。", "",
              "batch=8 的 p95 是整批延迟，不是每条 query 延迟；QPS 按成功 query 计。"
              "并发 4 并不表示使用了分布式推理或四个模型 worker。共享桌面的有限固定负载不能外推生产容量，"
              "不同版本的服务计时也未做同一时段、相同请求的配对性能实验。"
              "RSS 使用经过 PID 和创建时间核验的实际服务子进程，Windows 启动器及客户端不计入。", "",
              "## v1 历史对照（信息预算不等价）", ""]
    old_catalog = evidence["v1-retrieval"]
    historic_rows = [[f"v1 {name}", percent(values["click_micro_recall10"]), percent(values["click_micro_recall100"]),
                      "冻结训练反馈；v1 目录规则"] for name, values in old_catalog["metrics"].items()]
    historic_rows.append([f"v2 {serving_catalog_key}", percent(final_catalog["click_micro_recall10"]),
                          percent(final_catalog["click_micro_recall100"]), "严格延迟目录；历史反馈滚动回放"])
    lines += [table(["目录方法", "R@10", "R@100", "信息条件"], historic_rows), "",
              f"v1 的 {old_catalog['sample_impressions']:,} 请求有 {old_catalog['positive_clicked_items']:,} 个全点击，"
              f"eligible={old_catalog['eligible_positive_items']:,}；v2 eligible={eligible:,}。"
              "主指标始终用各自全部日志点击，不能把不同可用目录的 eligible 比例当成严格相同目标。"
              "这里保留历史数值，不给跨版本伪配对置信区间、不宣称同信息预算下的提升。", ""]
    v1_ranking = evidence["v1-ranking"]
    historic_ranking = []
    for name in ("popularity", "content"):
        if name in v1_ranking["runs"]:
            historic_ranking.append([f"v1 {name}", "单一基线", *[number(v1_ranking["runs"][name]["metrics"][metric]["mean"])
                                                                             for metric in METRICS]])
    for name, value in v1_ranking["seed_summary"].items():
        historic_ranking.append([f"v1 {name}", "3 seed 均值 ± SD",
                                 *[f"{number(value[metric]['mean'])} ± {number(value[metric]['std'])}" for metric in METRICS]])
    historic_ranking.append([f"v2 {ranker_name}", "内部选中单模型", *[number(full_metrics[metric]["mean"]) for metric in METRICS]])
    lines += [table(["完整 dev 日志排序", "统计口径", "AUC", "MRR", "nDCG@5", "nDCG@10"], historic_ranking), "",
              "v1 与 v2 的训练预算、特征和反馈信息不同；以上仅作完整披露。"
              "v2 自身的同样本、同训练预算三种输入消融，才用于判断内容与时间上下文的增量。", "",
              "## 可支持的结论与未支持部分", ""]
    for key, title in (("no_popular", "热度通道"), ("no_content", "内容通道"),
                       ("no_tower", "双塔通道"), ("no_freshness", "首次观察时效窗口")):
        value = catalog["paired_user_intervals"][f"full_minus_{key}_recall100"]
        lines.append(f"- {title}：{verdict(value)}；完整融合减去对应消融的 R@100 差为 {interval(value)} pp。")
    feedback = catalog["paired_user_intervals"]["full_minus_frozen_feedback_recall100"]
    lines.append(f"- 滚动反馈相对冻结训练统计：{verdict(feedback)}，R@100 差 {interval(feedback)} pp；"
                 "该对照同时改变反馈范围与衰减方式，不能单独证明 EMA 或更多反馈的因果贡献。")
    for name, title in (("full_minus_content_only", "在内容特征上加入时间上下文"),
                       ("full_minus_context_only", "在时间上下文上加入个性化内容特征")):
        value = ranking["paired_user_intervals"][name]["ndcg10"]
        lines.append(f"- {title}：{verdict(value)}，12,000 曝光组的 nDCG@10 差 {interval(value)} pp。")
    for variant in VARIANTS:
        value = catalog["paired_user_intervals"][f"selected_rerank_{variant}_minus_no_rerank_recall10"]
        lines.append(f"- `{variant}` 重排选定候选：{verdict(value)}，R@10 差 {interval(value)} pp。")
    lines += ["", "不因某部件被实现就宣称它有用；上述结论保留负结果与跨零区间。"
              "服务选择仍是内部冻结结果，不按这张 dev 表重新挑选。所有区间都是本数据、本抽样和回放假设下的描述，"
              "没有线上 A/B、未曝光文章完整相关性或 propensity；不能声称线上 CTR 提升、因果收益、SOTA 或分布式加速。", "",
              "## 复现与证据摘要", ""]
    def link(path):
        return Path(os.path.relpath(path, Path(output).parent)).as_posix()
    evidence_rows = [[f"[{name}]({link(path)})", sha256(path)] for name, path in paths.items()]
    lines += [table(["聚合证据", "文件 SHA256"], evidence_rows), ""]
    bound = frozen["bindings"]
    hash_rows = [["processed manifest", bound["data_manifest"]], ["temporal manifest", bound["temporal_manifest"]],
                 ["成熟选择骨干参数 state", backbone["selected"]["state_sha256"]],
                 ["实际服务 retriever 文件", backbone["current_backbone"]["checkpoint_sha256"]],
                 ["数据特征", bound["processed_files"]["news_features.npy"]],
                 ["缓存审计 row indices", evidence["query_cache_equivalence"]["row_indices_sha256"]],
                 ["内部候选 row indices", select["window_selection"]["row_indices_sha256"]],
                 ["内部排序 row indices", internal_ranker["validation_row_indices_sha256"]],
                 ["dev catalogue row indices", catalog["row_indices_sha256"]],
                 ["dev ranking sample row indices", ranking["row_indices_sha256"]],
                 ["dev ranking full row indices", full["row_indices_sha256"]],
                 ["HTTP row indices", service["sampling"]["row_sha256"]],
                 ["HTTP model version", service["health"]["model_version"]],
                 ["report generator", sha256(__file__)]]
    lines += [table(["绑定对象", "SHA256"], hash_rows), "",
              table(["执行阶段", "记录耗时（秒）"], [["dev catalogue", number(catalog["seconds"], 3)],
                                                       ["dev ranking sample", number(ranking["seconds"], 3)],
                                                       ["dev ranking full", number(full["seconds"], 3)],
                                                       ["HTTP server phase", number(service["elapsed_seconds"], 3)]]), "",
              "原始历史、用户/请求行与逐候选分数保留在忽略的 artifacts/ 中；公开聚合 JSON 保存哈希。"
              "哈希可以核验文件一致性，不能替代重新下载许可数据和实际运行。模型与数据使用方式见 "
              f"[复现说明]({link(Path('README.md'))})、[v2 协议]({link(Path('docs/V2_PROTOCOL.md'))}) 和 "
              f"[v2 执行顺序]({link(Path('docs/V2_REPRODUCE.md'))})。", "",
              "从仓库根目录按以下阶段复现；时间目录、缓存审计与正式实验拒绝覆盖既有结果。"
              "编码器导出、benchmark 和报告生成可以覆盖输出，重跑前须保留原始证据：", "",
              "```powershell", "python -m newsforge.v2.temporal --data data/processed --output data/temporal_v2",
              "python -m newsforge.v2.select_backbone",
              "# 先确认骨干审计 matches_current_retriever_state=true；完整保护检查见 V2_REPRODUCE.md。",
              "python -m newsforge.v2.retrieval export-encoder",
              "python -m newsforge.v2.ranker --train-only",
              "python -m newsforge.v2.mature_selection",
              "python -m newsforge.v2.verify_cache",
              "python -m newsforge.v2.experiment --phase select",
              "python -m newsforge.v2.experiment --phase catalog",
              "python -m newsforge.v2.experiment --phase ranking",
              "python -m newsforge.v2.experiment --phase ranking-full",
              "python -m newsforge.v2.benchmark", "python -m newsforge.v2.audit",
              "python -m newsforge.v2.report", "```", "",
              "上述命令需要先完成 v1 数据处理、固定双塔及 NumPy 编码器导出；具体参数和依赖以 README 为准。"
              "报告生成器会拒绝缺失文件、缺少预定消融、样本量错误或 selection/data bindings 不一致，"
              "不会把未执行步骤补成结果。", "",
              f"本报告从聚合证据自动生成于 {datetime.now(timezone.utc).isoformat()}；"
              "报告生成时间不是模型训练或 dev 评估时间。", ""]
    if "execution-notes" in evidence:
        lines += ["执行异常与修复记录见 "
                  f"[execution_notes.json]({link(paths['execution-notes'])})。"
                  "目录评估期间观察到一次明显的墙钟停顿，原因未确认；"
                  "日志时间戳与冻结模型未变。目录阶段总耗时包含该停顿，不能用于推断服务性能。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("reports/v2"))
    parser.add_argument("--ranker-selection", type=Path, default=Path("reports/v2/ranker_matured_selection.json"))
    parser.add_argument("--development-ranker-selection", type=Path, default=Path("reports/v2/ranker/selection.json"))
    parser.add_argument("--matured-internal-evaluation", type=Path,
                        default=Path("reports/v2/ranker_matured_internal_evaluation.json"))
    parser.add_argument("--backbone-selection", type=Path, default=Path("reports/v2/backbone_matured_selection.json"))
    parser.add_argument("--v1-retrieval", type=Path, default=Path("reports/retrieval_experiment.json"))
    parser.add_argument("--v1-ranking", type=Path, default=Path("reports/ranking_experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/V2_RESULTS.md"))
    args = parser.parse_args()
    paths = {name: args.input_dir / f"{name}.json" for name in
             ("select", "catalog", "ranking", "ranking-full", "service_benchmark", "query_cache_equivalence")}
    paths.update({"ranker-selection": args.ranker_selection,
                  "ranker-development-selection": args.development_ranker_selection,
                  "ranker-matured-internal-evaluation": args.matured_internal_evaluation,
                  "backbone-matured-selection": args.backbone_selection,
                  "v1-retrieval": args.v1_retrieval, "v1-ranking": args.v1_ranking})
    for name, filename in (("execution-notes", "execution_notes.json"),
                           ("release-integrity", "release_integrity.json")):
        optional_path = args.input_dir / filename
        if optional_path.is_file():
            paths[name] = optional_path
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Complete evidence is required; missing: " + ", ".join(missing))
    evidence = {name: read_json(path) for name, path in paths.items()}
    content = render(evidence, paths, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Generated {args.output} from {len(paths)} verified aggregate evidence files", flush=True)


if __name__ == "__main__":
    main()
