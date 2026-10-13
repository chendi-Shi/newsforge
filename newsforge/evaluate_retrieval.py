"""Clicked-item retrieval on a declared observable catalogue, separate from MIND ranking."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .store import Store
from .models import TwoTower, AttentionRanker
from .retrieval import CatalogIndex, build_from_files, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval_experiment.json"))
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    store = Store(args.data)
    store.ensure_training_cache()
    retriever = torch.load(args.artifacts / "retriever.pt", weights_only=True, map_location="cpu")
    model = TwoTower(**retriever["config"])
    model.load_state_dict(retriever["state_dict"])
    model.eval()
    saved_ranker = torch.load(args.artifacts / "reranker.pt", weights_only=True, map_location="cpu")
    ranker = AttentionRanker(**saved_ranker["config"])
    ranker.load_state_dict(saved_ranker["state_dict"])
    ranker.eval()
    cutoff = int(store.manifest["train_cutoff_timestamp"])
    # All observed article vectors may be indexed; each query filters first_seen.
    learned = build_from_files(args.data, args.artifacts / "article_embeddings.npy", args.artifacts / "index_hnsw",
                               default_as_of=cutoff, threads=args.threads,
                               model_version=sha256_file(args.artifacts / "retriever.pt"))
    content = CatalogIndex.build(store.features, store.news_ids, store.first_seen, store.clicks,
                                 default_as_of=cutoff, kind="exact", threads=args.threads,
                                 model_version="training_fitted_tfidf_svd")
    population = np.flatnonzero(store.split == 2)
    rows = np.sort(np.random.default_rng(2026).choice(population, min(args.queries, len(population)), replace=False))
    results = {name: [] for name in ["popularity", "content_exact", "two_tower_hnsw", "two_stage"]}
    total_positive = eligible_positive = repeated_positive = cold_positive = 0
    cold_hits = {name: 0 for name in results}
    full_catalog_sizes = []
    start_time = time.perf_counter()
    with torch.inference_mode():
        for offset, row in enumerate(rows):
            candidates, labels = store.candidates_for(row)
            positives = set(int(x) for x in candidates[labels == 1])
            history_ids = store.history[row]
            seen = set(int(x) for x in history_ids if x)
            as_of = int(store.timestamps[row])
            eligible = {i for i in positives if store.first_seen[i] <= as_of and i not in seen}
            cold = {i for i in eligible if store.first_seen[i] > cutoff}
            total_positive += len(positives)
            eligible_positive += len(eligible)
            repeated_positive += len(positives & seen)
            cold_positive += len(cold)
            full_catalog_sizes.append(int(((store.first_seen <= as_of) & (np.arange(len(store.first_seen)) > 0)).sum()))
            hf, hm = store.history_batch(np.array([row]))
            tensor_hf, tensor_hm = torch.from_numpy(hf), torch.from_numpy(hm)
            if bool(hm.any()):
                raw_query = np.asarray(store.mean_history[row])
                content_items = content.search(raw_query, 100, as_of=as_of, seen=seen).indices
                query = model.encode_user(tensor_hf, tensor_hm)[0].numpy()
                learned_items = learned.search(query, 100, as_of=as_of, seen=seen,
                                               max_overfetch=8192).indices
            else:
                content_items = content.popular(100, as_of=as_of, seen=seen).indices
                learned_items = learned.popular(100, as_of=as_of, seen=seen).indices
            popular = learned.popular(100, as_of=as_of, seen=seen).indices
            if learned_items and bool(hm.any()):
                scores = ranker.score_candidates(tensor_hf, tensor_hm,
                    torch.from_numpy(store.features[np.array(learned_items)[None]])).numpy()[0]
                reranked = np.asarray(learned_items)[np.argsort(-scores, kind="stable")].tolist()
            else:
                reranked = learned_items
            for name, items in [("popularity", popular), ("content_exact", content_items),
                                ("two_tower_hnsw", learned_items), ("two_stage", reranked)]:
                hit10 = len(positives.intersection(items[:10]))
                hit100 = len(positives.intersection(items[:100]))
                results[name].append([hit10, hit100, len(positives), len(eligible)])
                cold_hits[name] += len(cold.intersection(items[:100]))
            if offset % 500 == 0:
                print(f"Retrieval evaluation {offset:,}/{len(rows):,}", flush=True)
    summary = {}
    for name, data in results.items():
        values = np.asarray(data)
        summary[name] = {"click_micro_recall10": float(values[:, 0].sum() / max(total_positive, 1)),
                         "click_micro_recall100": float(values[:, 1].sum() / max(total_positive, 1)),
                         "eligible_click_micro_recall100": float(values[:, 1].sum() / max(eligible_positive, 1)),
                         "cold_eligible_click_recall100": float(cold_hits[name] / max(cold_positive, 1))}
    result = {"query_split": "locked official dev", "seed": 2026, "sample_impressions": len(rows),
              "full_split_impressions": len(population), "positive_clicked_items": total_positive,
              "eligible_positive_items": eligible_positive, "repeat_clicks_excluded_by_service": repeated_positive,
              "cold_eligible_positive_items": cold_positive,
              "catalogue_size_min_max": [min(full_catalog_sizes), max(full_catalog_sizes)],
              "model_selection": "Internal validation only", "candidate_pool_size": 100,
              "max_overfetch": 8192, "cold_start_policy": "train click popularity, no reranking",
              "selected_retriever": retriever["run_id"],
              "selected_ranker": saved_ranker["run_id"], "metrics": summary,
              "seconds": time.perf_counter() - start_time,
              "limitations": ["First observed in a request is a catalogue-availability proxy, not publication time.",
                  "Only logged clicks are relevance evidence; unexposed news feedback is unknown.",
                  "Full-click denominator includes repeated clicks excluded by service; eligible denominator is separate.",
                  "This is not official MIND ranking evaluation or causal online reward.",
                  "Reordering the same top100 cannot change recall100; compare top10 for reranking."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
