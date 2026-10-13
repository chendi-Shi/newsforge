"""Matched-budget exposure-negative experiments over every training click."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from .store import Store
from .models import TwoTower, AttentionRanker
from .evaluate import evaluate, select_validation_rows, encode_catalog
from .metrics import user_cluster_interval, METRICS


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def build_model(variant, feature_dim):
    if variant == "ranker":
        config = {"feature_dim": feature_dim, "hidden": 64, "embedding_dim": 64}
        return AttentionRanker(**config), config, "ranker"
    config = {"feature_dim": feature_dim, "hidden": 128, "embedding_dim": 64,
              "user_pooling": "attention" if variant == "attention_mixed" else "mean"}
    return TwoTower(**config), config, "tower"


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def training_source_digests():
    directory = Path(__file__).resolve().parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ["models.py", "train.py", "store.py", "evaluate.py", "metrics.py"]}


def training_cache_signature(store):
    cache = json.loads((store.path / "training_cache.json").read_text())
    return {key: cache[key] for key in ["cache_version", "sampler", "seed", "pool_size",
                                      "max_chunk_exposures", "data_manifest_sha256", "features_sha256"]}


def choose_negatives(pool_ids, scores, rng, mixed):
    """Two hardest plus two random from a fixed 16-item exposed-negative pool.

    Duplicate IDs in short pools are allowed as explicit repeated samples, as in
    the baseline; no PAD, positive, history-only or unexposed items are inserted.
    """
    count, width = pool_ids.shape
    if mixed:
        hard = np.argsort(-scores, axis=1, kind="stable")[:, :2]
        random_indices = rng.integers(width, size=(count, 2))
        selection = np.concatenate([hard, random_indices], axis=1)
    else:
        selection = rng.integers(width, size=(count, 4))
    return np.take_along_axis(pool_ids, selection, axis=1)


def fit(store, variant, seed, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, config, kind = build_model(variant, store.features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    validation_rows = select_validation_rows(store, args.validation_impressions)
    best = -float("inf")
    winner, best_epoch = None, None
    history = []
    total_steps = 0
    train_seconds = 0.0
    n = len(store.positive_rows)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n)
        running_loss, examples = 0.0, 0
        started = time.perf_counter()
        # Warm-up is identical across random/mixed arms; mining uses only training
        # rows and the current model, after one complete random-negative epoch.
        mixed = variant in {"mean_mixed", "attention_mixed", "ranker"} and epoch > 1
        for start in range(0, n, args.batch_size):
            selected = order[start:start + args.batch_size]
            rows = store.positive_rows[selected]
            positives = store.positive_items[selected]
            pool = np.asarray(store.pool[rows])
            history_features, history_mask = store.history_batch(rows, mean=kind == "tower" and config["user_pooling"] == "mean")
            hf, hm = torch.from_numpy(history_features), torch.from_numpy(history_mask)
            mining_scores = None
            if mixed:
                with torch.no_grad():
                    mining_scores = model(hf, torch.from_numpy(store.features[pool]), hm).numpy()
            negatives = choose_negatives(pool, mining_scores, rng, mixed)
            candidates = np.column_stack([positives, negatives])
            cf = torch.from_numpy(store.features[candidates])
            optimizer.zero_grad(set_to_none=True)
            logits = model(hf, cf, hm)
            loss = criterion(logits, torch.zeros(len(rows), dtype=torch.long))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(rows)
            examples += len(rows)
            total_steps += 1
            if start == 0 or total_steps % 1000 == 0:
                print(json.dumps({"variant": variant, "seed": seed, "epoch": epoch,
                    "examples": examples, "total_examples": n, "loss": running_loss / examples,
                    "examples_per_second": examples / max(time.perf_counter() - started, 1e-6)}), flush=True)
            if args.smoke_steps and total_steps >= args.smoke_steps:
                break
        elapsed = time.perf_counter() - started
        if not args.smoke_steps and examples != n:
            raise RuntimeError("A formal training epoch did not consume every eligible positive pair")
        train_seconds += elapsed
        validation, _ = evaluate(store, model, kind, validation_rows)
        criterion_value = validation["metrics"]["ndcg10"]["mean"]
        history.append({"epoch": epoch, "negative_policy": "two_hard_two_random" if mixed else "four_random",
                        "training_examples": examples, "loss": running_loss / examples,
                        "training_seconds": elapsed, "state_sha256": state_digest(model),
                        "validation": validation})
        print(json.dumps({"variant": variant, "seed": seed, "epoch": epoch, "validation": validation["metrics"]}), flush=True)
        if criterion_value > best:
            best, best_epoch = criterion_value, epoch
            winner = copy.deepcopy(model.state_dict())
        if args.smoke_steps:
            break
    checkpoint = {"state_dict": winner, "config": config, "kind": kind, "variant": variant,
                  "seed": seed, "best_epoch": best_epoch, "validation_ndcg10": best,
                  "run_id": f"{variant}-seed{seed}", "history": history,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "training_seconds": train_seconds,
                  "trained_on_all_eligible_clicks": not bool(args.smoke_steps),
                  "eligible_positive_examples_per_epoch": n,
                  "training_configuration": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "threads": args.threads,
                     "learning_rate": args.learning_rate, "validation_impressions": args.validation_impressions,
                     "negative_pool_size": store.pool.shape[1]},
                  "data_manifest_sha256": hashlib.sha256((store.path / "manifest.json").read_bytes()).hexdigest(),
                  "training_source_sha256": training_source_digests(),
                  "training_cache_signature": training_cache_signature(store),
                  "train_cutoff_timestamp": int(store.manifest["train_cutoff_timestamp"])}
    path = args.artifacts / f"{variant}_seed{seed}.pt"
    torch.save(checkpoint, path)
    write_json(args.output / f"{variant}_seed{seed}_training.json", {k: v for k, v in checkpoint.items() if k != "state_dict"})
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--variants", nargs="+", choices=["mean_random", "mean_mixed", "attention_mixed", "ranker"],
                        default=["mean_random", "mean_mixed", "ranker"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--validation-impressions", type=int, default=12000)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("positive epochs, batch size and threads required")
    if args.smoke_steps and args.artifacts == Path("artifacts"):
        parser.error("smoke runs must use an explicit separate --artifacts directory")
    torch.set_num_threads(args.threads)
    threadpool_limits(args.threads)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    store = Store(args.data)
    store.ensure_training_cache()
    paths = []
    for variant in args.variants:
        for seed in args.seeds:
            path = args.artifacts / f"{variant}_seed{seed}.pt"
            if not args.resume or not path.exists():
                path = fit(store, variant, seed, args)
            else:
                saved = torch.load(path, weights_only=True, map_location="cpu")
                expected = {"epochs": args.epochs, "batch_size": args.batch_size,
                            "threads": args.threads,
                            "learning_rate": args.learning_rate, "validation_impressions": args.validation_impressions,
                            "negative_pool_size": store.pool.shape[1]}
                if saved.get("training_configuration") != expected or saved.get("data_manifest_sha256") != hashlib.sha256((store.path / "manifest.json").read_bytes()).hexdigest():
                    raise ValueError(f"Resume configuration/data mismatch: {path}")
                if saved.get("training_source_sha256") != training_source_digests():
                    raise ValueError(f"Resume training source mismatch: {path}")
                if saved.get("training_cache_signature") != training_cache_signature(store):
                    raise ValueError(f"Resume training cache mismatch: {path}")
            paths.append(path)
    # Select serving artifacts strictly using internal validation, before test.
    checkpoints = [torch.load(path, weights_only=True, map_location="cpu") for path in paths]
    warmup_checks = {}
    for seed in args.seeds:
        pair = {checkpoint["variant"]: checkpoint for checkpoint in checkpoints
                if checkpoint["seed"] == seed and checkpoint["variant"] in {"mean_random", "mean_mixed"}}
        if len(pair) == 2:
            digests = [pair[variant]["history"][0]["state_sha256"] for variant in ["mean_random", "mean_mixed"]]
            warmup_checks[str(seed)] = {"identical": digests[0] == digests[1], "sha256": digests}
            if digests[0] != digests[1]:
                raise RuntimeError("Random/mixed arms did not have identical matched-seed warmup weights")
    write_json(args.output / "matched_warmup_checks.json", warmup_checks)
    for kind, filename in [("tower", "retriever.pt"), ("ranker", "reranker.pt")]:
        options = [checkpoint for checkpoint in checkpoints if checkpoint["kind"] == kind]
        if options:
            winner = max(options, key=lambda checkpoint: checkpoint["validation_ndcg10"])
            torch.save(winner, args.artifacts / filename)
            if kind == "tower":
                model = TwoTower(**winner["config"])
                model.load_state_dict(winner["state_dict"])
                np.save(args.artifacts / "article_embeddings.npy", encode_catalog(store, model))
    if args.skip_test or args.smoke_steps:
        return
    test_rows = np.flatnonzero(store.split == 2)
    np.save(args.artifacts / "evaluation_rows.npy", test_rows)
    reports, matrices = {}, {}
    for baseline in ["popularity", "content"]:
        report, values = evaluate(store, None, baseline, test_rows, output=args.artifacts / f"metrics_{baseline}.npy")
        reports[baseline], matrices[baseline] = report, values
        print(json.dumps({"baseline": baseline, "metrics": report["metrics"]}), flush=True)
    for checkpoint in checkpoints:
        model, _, kind = build_model(checkpoint["variant"], store.features.shape[1])
        model.load_state_dict(checkpoint["state_dict"])
        run_id = checkpoint["run_id"]
        report, values = evaluate(store, model, kind, test_rows, output=args.artifacts / f"metrics_{run_id}.npy")
        reports[run_id], matrices[run_id] = report, values
        print(json.dumps({"run_id": run_id, "metrics": report["metrics"]}), flush=True)
        write_json(args.output / "ranking_partial.json", reports)
    aggregate = {}
    for variant in args.variants:
        run_names = [f"{variant}-seed{seed}" for seed in args.seeds]
        aggregate[variant] = {metric: {"mean": float(np.mean([reports[name]["metrics"][metric]["mean"] for name in run_names])),
            "std": float(np.std([reports[name]["metrics"][metric]["mean"] for name in run_names], ddof=1))
            if len(run_names) > 1 else None} for metric in METRICS}
    comparisons = {}
    if "mean_random" in args.variants:
        base = np.mean([matrices[f"mean_random-seed{seed}"] for seed in args.seeds], axis=0)
        for variant in args.variants:
            if variant != "mean_random":
                values = np.mean([matrices[f"{variant}-seed{seed}"] for seed in args.seeds], axis=0)
                comparisons[variant] = user_cluster_interval(base, values, store.users[test_rows])
    result = {"configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "platform": platform.platform(), "threads": args.threads},
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path("newsforge").glob("*.py"))},
              "runs": reports, "seed_summary": aggregate, "paired_user_bootstrap": comparisons,
              "matched_warmup_checks": warmup_checks,
              "bootstrap_estimand": "paired difference in per-impression metrics averaged over seeds; not score ensemble",
              "evaluation": "Official MIND-large development set used only after all internal-validation selections."}
    write_json(args.output / "ranking_experiment.json", result)


if __name__ == "__main__":
    main()
