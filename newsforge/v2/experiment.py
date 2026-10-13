"""Internal selection, frozen dev ablations, and complete logged-candidate evaluation.

This is a follow-up on a benchmark observed in v1, not an untouched test set.
Temporal replay deliberately uses only delayed, earlier feedback. Raw per-query
arrays stay under artifacts/; only aggregate evidence belongs in reports/.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numpy.lib.format import open_memmap
from threadpoolctl import threadpool_info, threadpool_limits

from newsforge.store import Store
from newsforge.metrics import impression_metrics, summarize, user_cluster_interval
from .retrieval import CandidateConfig, CandidateGenerator, rerank_order, sha256_file
from .temporal import TemporalStore
from .ranker import CausalRanker, FeatureBuilder

MIN_VALID_TIME = 1573722000  # 2019-11-14 09:00, deterministic source-time coordinate.
MAX_VALID_TIME = 1573772400  # Exclusive 2019-11-14 23:00: mature before the first dev cutoff.
SEEDS = (17, 29, 43)
VARIANTS = ("content_only", "context_only", "full")
LIMITATIONS = [
    "Follow-up on the v1-observed MIND dev benchmark; not a new untouched holdout.",
    "Rolling replay uses previous validation/dev feedback with an hourly snapshot and at least one-hour lag; v1 used frozen training feedback.",
    "First observed timestamp is an availability proxy, not publication time.",
    "Only logged clicks are observed relevance; retrieval recall is not online CTR or causal reward.",
    "Shared CPU desktop timings are not production capacity estimates.",
]


def native_pools():
    return [{key: pool.get(key) for key in ("internal_api", "prefix", "num_threads")}
            for pool in threadpool_info()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sample_rows(store, split, count):
    eligible = np.flatnonzero(store.split == split)
    if split == 1:
        eligible = eligible[(store.timestamps[eligible] >= MIN_VALID_TIME)
                            & (store.timestamps[eligible] < MAX_VALID_TIME)]
    if count < 1:
        return eligible
    return np.sort(np.random.default_rng(2026).choice(eligible, min(count, len(eligible)), replace=False))


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def recall_interval(before_hits, after_hits, denominators, users, repeats=1000):
    """Paired user-cluster bootstrap of the micro-recall difference (not macro)."""
    before, after, denom = map(lambda x: np.asarray(x, dtype=np.float64),
                                (before_hits, after_hits, denominators))
    users = np.asarray(users)
    if not (before.shape == after.shape == denom.shape == users.shape) or before.ndim != 1:
        raise ValueError("matching one-dimensional query arrays required")
    if np.any(denom < 0) or np.any(before < 0) or np.any(after < 0) or denom.sum() <= 0:
        raise ValueError("nonnegative counts and a positive denominator required")
    _, inv = np.unique(users, return_inverse=True)
    n = int(inv.max()) + 1
    delta = np.bincount(inv, weights=after - before, minlength=n)
    counts = np.bincount(inv, weights=denom, minlength=n)
    rng, samples = np.random.default_rng(2026), []
    for _ in range(repeats):
        chosen = rng.integers(n, size=n)
        total = counts[chosen].sum()
        if total:
            samples.append(delta[chosen].sum() / total)
    return {"delta": float((after - before).sum() / denom.sum()),
            "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "resampling_unit": "user", "replicates": len(samples), "seed": 2026}


def recipes(hours):
    full = CandidateConfig(recent_hours=hours)
    return {
        "full": full,
        "no_popular": replace(full, channels=("content", "tower")),
        "no_content": replace(full, channels=("popular", "tower")),
        "no_tower": replace(full, channels=("popular", "content")),
        "popular_only": replace(full, channels=("popular",)),
        "content_only": replace(full, channels=("content",)),
        "tower_only": replace(full, channels=("tower",)),
        "no_freshness": replace(full, recent_hours=None),
        "frozen_feedback": replace(full, freeze_feedback=True),
    }


def catalog_evaluate(store, generator, rows, configs, rankers=None, save_path=None):
    """Every recipe has exactly the same queries, denominator and pool budget."""
    started = time.perf_counter()
    names = list(configs)
    rankers = rankers or {}
    keys = names + [f"selected_rerank_{name}" for name in rankers]
    # hit@10, hit@100, cold hit@100, zero-training-click hit@100, union hit.
    counts = {name: np.zeros((len(rows), 5), dtype=np.int32) for name in keys}
    denominators = np.zeros((len(rows), 5), dtype=np.int32)
    latency = {name: [] for name in names}
    shortfalls = {name: 0 for name in names}
    empty = {name: 0 for name in names}
    route_hits = {name: {r: 0 for r in ("popular", "content", "tower", "fallback")} for name in names}
    available_min, available_max = 2**63 - 1, 0
    cutoff_train = int(store.manifest["train_cutoff_timestamp"])
    execution = np.argsort(store.timestamps[rows], kind="stable")
    for progress, pos in enumerate(execution):
        row = int(rows[pos])
        ids, labels = store.candidates_for(row)
        positive = set(map(int, ids[labels == 1]))
        history, at = store.history[row], int(store.timestamps[row])
        seen = set(map(int, history[history > 0]))
        cut = (at // 3600) * 3600 - 3600
        eligible = {i for i in positive if store.first_seen[i] < cut and i not in seen}
        cold = {i for i in eligible if store.first_seen[i] > cutoff_train}
        zero = {i for i in eligible if store.clicks[i] == 0}
        denominators[pos] = [len(positive), len(eligible), len(cold), len(zero), len(positive & seen)]
        selected_batch = None
        for name, config in configs.items():
            tick = time.perf_counter()
            batch = generator.retrieve(history, at, config=config, reuse_query_routes=True)
            latency[name].append((time.perf_counter() - tick) * 1000)
            if name == "selected":
                selected_batch = batch
            result = set(map(int, batch.ids))
            union = set()
            for route, values in batch.sources.items():
                union.update(values)
                route_hits[name][route] += len(positive.intersection(values))
            counts[name][pos] = [len(positive.intersection(batch.ids[:10])), len(positive & result),
                                len(cold & result), len(zero & result), len(positive & union)]
            shortfalls[name] += int(batch.shortfall > 0)
            empty[name] += int(batch.metadata["empty_usable_history"])
            available_min = min(available_min, batch.metadata["visible_catalogue_count"])
            available_max = max(available_max, batch.metadata["visible_catalogue_count"])
        for name, ranker in rankers.items():
            if selected_batch is None:
                raise ValueError("reranking requires a selected candidate recipe")
            ids = selected_batch.ids
            ordered = ids[rerank_order(ids, ranker.score(history, ids, at))]
            key = f"selected_rerank_{name}"
            counts[key][pos] = counts["selected"][pos]
            counts[key][pos, 0] = len(positive.intersection(ordered[:10]))
        if (progress + 1) % 250 == 0:
            print(f"catalogue {progress + 1}/{len(rows)} ({time.perf_counter()-started:.1f}s)", flush=True)
    totals = denominators.sum(axis=0)
    metrics = {}
    for name, values in counts.items():
        sums = values.sum(axis=0)
        metrics[name] = {
            "click_micro_recall10": float(sums[0] / max(totals[0], 1)),
            "click_micro_recall100": float(sums[1] / max(totals[0], 1)),
            "eligible_click_micro_recall100": float(sums[1] / max(totals[1], 1)),
            "cold_eligible_recall100": float(sums[2] / max(totals[2], 1)),
            "zero_train_click_eligible_recall100": float(sums[3] / max(totals[3], 1)),
            "hits10": int(sums[0]), "hits100": int(sums[1]),
            "route_union_oracle_recall": float(sums[4] / max(totals[0], 1)),
        }
        if name in latency:
            metrics[name].update({"candidate_config": asdict(configs[name]),
                "candidate_ms_p50_p95": np.quantile(latency[name], [.5, .95]).tolist(),
                "shortfall_queries": shortfalls[name], "empty_history_queries": empty[name],
                "route_hits_before_fusion": route_hits[name]})
    comparisons = {}
    for name in names:
        if name != "full" and "full" in counts:
            comparisons[f"full_minus_{name}_recall100"] = recall_interval(
                counts[name][:, 1], counts["full"][:, 1], denominators[:, 0], store.users[rows])
    if "selected" in counts:
        for name in rankers:
            key = f"selected_rerank_{name}"
            comparisons[f"{key}_minus_no_rerank_recall10"] = recall_interval(
                counts["selected"][:, 0], counts[key][:, 0], denominators[:, 0], store.users[rows])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, rows=rows, denominators=denominators, **counts)
    return {"queries": len(rows), "row_indices_sha256": row_hash(rows),
            "query_time_min_max": [int(store.timestamps[rows].min()), int(store.timestamps[rows].max())],
            "query_split_counts": {str(int(value)): int(np.sum(store.split[rows] == value))
                                   for value in np.unique(store.split[rows])},
            "positive_clicked_items": int(totals[0]), "eligible_positive_items": int(totals[1]),
            "cold_eligible_positive_items": int(totals[2]), "zero_train_click_eligible_positive_items": int(totals[3]),
            "repeat_clicks": int(totals[4]), "catalogue_size_min_max": [available_min, available_max],
            "metrics": metrics, "paired_user_intervals": comparisons,
            "execution_schedule": {
                "reuse_current_query_semantic_routes": True,
                "equivalence_report": "reports/v2/query_cache_equivalence.json",
                "candidate_latency_scope": "Marginal time in a shared ablation sweep; routes may be reused from earlier recipes. Not independent strategy or service latency.",
                "service_default_reuses_query_routes": False},
            "seconds": time.perf_counter() - started, "native_thread_pools": native_pools(),
            "limitations": LIMITATIONS}


def ranking_evaluate(store, temporal, rows, models, artifacts):
    """Compute a common full feature matrix once; reuse it across all models."""
    started = time.perf_counter()
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray(store.offsets[rows + 1] - store.offsets[rows], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    feature_path = artifacts / "full_features.npy"
    if feature_path.exists():
        raise FileExistsError(f"Use a fresh evidence directory: {feature_path}")
    features = open_memmap(feature_path, mode="w+", dtype="float32", shape=(int(offsets[-1]), 22))
    builder = FeatureBuilder(store, temporal, variant="full")
    # Length buckets bound padding and retain canonical row order in outputs.
    execution = np.argsort(lengths, kind="stable")
    for begin in range(0, len(rows), 64):
        positions = execution[begin:begin + 64]
        rr = rows[positions]
        candidates, _ = store.padded_candidates(rr)
        matrix = builder.row_features(store, rr, candidates)
        for b, pos in enumerate(positions):
            features[offsets[pos]:offsets[pos + 1]] = matrix[b, :lengths[pos]]
        if (begin // 64) % 100 == 0:
            print(f"ranking features {begin}/{len(rows)}", flush=True)
    features.flush()
    values = {key: open_memmap(artifacts / f"{key}_metrics.npy", mode="w+", dtype="float64",
                              shape=(len(rows), 4)) for key in models}
    scores = {key: open_memmap(artifacts / f"{key}_scores.npy", mode="w+", dtype="float64",
                              shape=(int(offsets[-1]),)) for key in models}
    for key, model in models.items():
        for begin in range(0, len(features), 32768):
            scores[key][begin:begin + 32768] = model.predict_full_features(features[begin:begin + 32768])
        scores[key].flush()
        for pos, row in enumerate(rows):
            _, labels = store.candidates_for(row)
            values[key][pos] = impression_metrics(labels, scores[key][offsets[pos]:offsets[pos + 1]])
        values[key].flush()
        print(f"ranking metrics {key}: {summarize(values[key])}", flush=True)
    metrics = {key: summarize(value) for key, value in values.items()}
    variant_values, seed_stats = {}, {}
    for variant in VARIANTS:
        keys = [f"{variant}-seed{seed}" for seed in SEEDS]
        if all(key in values for key in keys):
            stack = np.stack([values[key] for key in keys])
            variant_values[variant] = stack.mean(axis=0)
            per_seed = np.nanmean(stack, axis=1)
            seed_stats[variant] = {name: {"mean": float(per_seed[:, i].mean()),
                                         "sample_std": float(per_seed[:, i].std(ddof=1))}
                                  for i, name in enumerate(("auc", "mrr", "ndcg5", "ndcg10"))}
    comparisons = {}
    for before, after in (("content_only", "full"), ("context_only", "full"), ("content_only", "context_only")):
        if before in variant_values and after in variant_values:
            comparisons[f"{after}_minus_{before}"] = user_cluster_interval(
                variant_values[before], variant_values[after], store.users[rows], repeats=1000)
    np.save(artifacts / "rows.npy", rows)
    return {"impressions": len(rows), "candidates": int(offsets[-1]), "row_indices_sha256": row_hash(rows),
            "metrics": metrics, "seed_statistics": seed_stats, "paired_user_intervals": comparisons,
            "feature_sha256": sha256_file(feature_path), "feature_columns": 22,
            "seconds": time.perf_counter() - started, "native_thread_pools": native_pools(),
            "limitations": LIMITATIONS,
            "tie_policy": "stable logged candidate order, same as v1 MIND ranking; catalogue ties use article index"}


def model_entries(selection):
    entries = selection["models"]
    return list(entries.values()) if isinstance(entries, dict) else entries


def load_selected_model(entry, store, temporal):
    """Require the model selected internally, not just a self-consistent sidecar."""
    path = Path(entry["model_path"])
    if sha256_file(path) != entry["model_sha256"]:
        raise ValueError("Selected ranker model SHA mismatch")
    model = CausalRanker.load(path, store, temporal)
    # Unpickling loads sklearn's OpenMP runtime after the entry-point limit.
    # Apply the limit again so small HGB predictions do not spawn eight threads.
    threadpool_limits(1)
    if (model.model_version != entry["model_sha256"]
            or model.variant != entry["variant"]
            or model.metadata.get("seed") != entry["seed"]):
        raise ValueError("Selected ranker identity differs from internal selection")
    return model


def load_models(selection, store, temporal, best_only=False):
    entries = list(selection["best_by_variant"].values()) if best_only else model_entries(selection)
    result = {}
    for entry in entries:
        name = entry["variant"] if best_only else f'{entry["variant"]}-seed{entry["seed"]}'
        result[name] = load_selected_model(entry, store, temporal)
    return result


def bindings(args, temporal):
    # The processed manifest declares counts and policies, not array hashes.
    # Freeze every array used by sampling, features, metrics, and retrieval so
    # an unchanged manifest cannot hide a changed evaluation population.
    data_files = (
        "news_features.npy", "news_first_seen.npy", "news_train_clicks.npy",
        "news_train_exposures.npy", "news_ids.json", "all_history.npy",
        "all_candidates.npy", "all_labels.npy", "all_offsets.npy",
        "all_timestamps.npy", "all_users.npy", "all_split.npy",
        "training_cache.json", "negative_pool.npy", "positive_rows.npy",
        "positive_items.npy",
    )
    retrieval_files = {
        "article_embeddings": Path("artifacts/article_embeddings.npy"),
        "retriever": Path("artifacts/retriever.pt"),
        "encoder": Path("artifacts/v2/user_encoder.npz"),
        "encoder_manifest": Path("artifacts/v2/user_encoder.npz.json"),
    }
    selected = json.loads(Path(args.ranker_selection).read_text(encoding="utf-8"))
    rankers = {}
    for entry in model_entries(selected):
        path = Path(entry["model_path"])
        actual = sha256_file(path)
        if actual != entry["model_sha256"]:
            raise ValueError("Selected ranker model SHA mismatch")
        run = f'{entry["variant"]}-seed{entry["seed"]}'
        if run in rankers:
            raise ValueError("Duplicate selected ranker identity")
        rankers[run] = {"model_path": str(path).replace("\\", "/"),
                        "model_sha256": actual,
                        "manifest_sha256": sha256_file(Path(str(path) + ".manifest.json"))}
    for entry in [selected["selected"], *selected["best_by_variant"].values()]:
        run = f'{entry["variant"]}-seed{entry["seed"]}'
        if (run not in rankers or rankers[run]["model_sha256"] != entry["model_sha256"]
                or rankers[run]["model_path"] != str(Path(entry["model_path"])).replace("\\", "/")):
            raise ValueError("Ranker selection entries are inconsistent")
    return {"data_manifest": sha256_file(Path(args.data) / "manifest.json"),
            "temporal_manifest": sha256_file(Path(args.temporal) / "manifest.json"),
            "processed_files": {name: sha256_file(Path(args.data) / name) for name in data_files},
            "retrieval_artifacts": {name: sha256_file(path) for name, path in retrieval_files.items()},
            "ranker_artifacts": rankers,
            "ranker_selection": sha256_file(args.ranker_selection),
            "backbone_matured_selection": sha256_file(Path(args.reports) / "backbone_matured_selection.json"),
            "query_cache_equivalence": sha256_file(Path(args.reports) / "query_cache_equivalence.json"),
            "source": {name: sha256_file(Path(__file__).parent / name)
                       for name in ("experiment.py", "retrieval.py", "temporal.py", "ranker.py",
                                    "verify_cache.py", "mature_selection.py", "select_backbone.py")},
            "metrics_source": sha256_file(Path(__file__).parents[1] / "metrics.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("select", "catalog", "ranking", "ranking-full"))
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--temporal", default="data/temporal_v2")
    parser.add_argument("--ranker-selection", default="reports/v2/ranker_matured_selection.json")
    parser.add_argument("--selection", default="artifacts/v2/selection.json")
    parser.add_argument("--reports", default="reports/v2")
    parser.add_argument("--artifacts", default="artifacts/v2")
    args = parser.parse_args()
    threadpool_limits(1)
    report_path = Path(args.reports) / f"{args.phase}.json"
    if report_path.exists() or (args.phase == "select" and Path(args.selection).exists()):
        raise FileExistsError("Formal evidence already exists. Preserve it; use fresh paths for a new study.")
    store = Store(args.data)
    temporal = TemporalStore(args.temporal, processed_dir=args.data)
    ranker_selection = json.loads(Path(args.ranker_selection).read_text(encoding="utf-8"))
    current_bindings = bindings(args, temporal)
    if args.phase == "select":
        generator = CandidateGenerator.from_paths(args.data, "artifacts", temporal)
        rows = sample_rows(store, 1, 2000)
        windows = {f"window_{hours}": CandidateConfig(recent_hours=hours) for hours in (24, 48, 72)}
        window_result = catalog_evaluate(store, generator, rows, windows)
        best_window = max(windows, key=lambda k: window_result["metrics"][k]["click_micro_recall100"])
        hours = windows[best_window].recent_hours
        configs = recipes(hours)
        candidate_result = catalog_evaluate(store, generator, rows, configs,
                                            save_path=Path(args.artifacts) / "validation_catalog_counts.npz")
        candidates = [key for key in configs if key != "frozen_feedback"]
        chosen = max(candidates, key=lambda k: candidate_result["metrics"][k]["click_micro_recall100"])
        rankers = load_models(ranker_selection, store, temporal, best_only=True)
        rerank_result = catalog_evaluate(store, generator, rows, {"selected": configs[chosen]}, rankers,
                                        save_path=Path(args.artifacts) / "validation_rerank_counts.npz")
        selected_rerank = max(rerank_result["metrics"], key=lambda k: rerank_result["metrics"][k]["click_micro_recall10"])
        use_ranker = selected_rerank != "selected"
        ranker_variant = selected_rerank.replace("selected_rerank_", "") if use_ranker else None
        selected_ranker = ranker_selection["best_by_variant"].get(ranker_variant)
        selection = {"created_utc": datetime.now(timezone.utc).isoformat(),
                     "selection_split": "internal validation only, 2019-11-14 09:00 <= timestamp < 23:00 source coordinate; all labels mature before first dev cutoff",
                     "selection_label_time_bounds": {"minimum_inclusive": MIN_VALID_TIME,
                                                     "maximum_exclusive": MAX_VALID_TIME},
                     "ranking_selected": ranker_selection["selected"],
                     "candidate_selected_name": chosen, "ablation_window_hours": hours,
                     "serving": {"candidate_config": asdict(configs[chosen]),
                                 "ranker_path": selected_ranker["model_path"] if use_ranker else None,
                                 "ranker_identity": ({key: selected_ranker[key] for key in
                                                      ("model_path", "model_sha256", "variant", "seed")}
                                                     if use_ranker else None),
                                 "use_reranker": use_ranker, "encoder_path": "artifacts/v2/user_encoder.npz",
                                 "threads": 1, "default_as_of": None},
                     "bindings": current_bindings, "limitations": LIMITATIONS}
        dump(args.selection, selection)
        dump(report_path, {"window_selection": window_result, "candidate_ablation": candidate_result,
                           "reranker_selection": rerank_result, "frozen_selection": selection})
    else:
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        if selection["bindings"] != current_bindings:
            raise ValueError("Selection/data/model/source binding changed after internal selection")
        if args.phase == "catalog":
            generator = CandidateGenerator.from_paths(args.data, "artifacts", temporal)
            configs = recipes(selection["ablation_window_hours"])
            configs["selected"] = CandidateConfig(**selection["serving"]["candidate_config"])
            result = catalog_evaluate(store, generator, sample_rows(store, 2, 5000), configs,
                                       load_models(ranker_selection, store, temporal, best_only=True),
                                       Path(args.artifacts) / "dev_catalog_counts.npz")
        elif args.phase == "ranking":
            result = ranking_evaluate(store, temporal, sample_rows(store, 2, 12000),
                                      load_models(ranker_selection, store, temporal),
                                      Path(args.artifacts) / "dev_ranking_sample")
        else:
            chosen = selection["ranking_selected"]
            name = f'{chosen["variant"]}-seed{chosen["seed"]}'
            model = load_selected_model(chosen, store, temporal)
            result = ranking_evaluate(store, temporal, sample_rows(store, 2, 0), {name: model},
                                      Path(args.artifacts) / "dev_ranking_full")
        result.update({"phase": args.phase, "selection_sha256": sha256_file(args.selection),
                       "bindings": current_bindings, "selected_before_v2_dev": selection["created_utc"]})
        dump(report_path, result)
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
