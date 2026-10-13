"""Reselect a fixed epoch-2 backbone using only matured internal-validation labels.

The three v1 mean-random checkpoints are eligible only when their loaded state
matches the recorded end of the second full epoch. Original validation scores
are never consulted. This command evaluates complete logged candidate groups
and writes a selection report; it never exports or replaces serving artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

SEEDS = (17, 29, 43)
VALIDATION_IMPRESSIONS = 6000
SAMPLING_SEED = 2026
MIN_VALID_TIME = 1573722000  # 2019-11-14 09:00, deterministic source-time coordinate.
MAX_VALID_TIME = 1573772400  # Exclusive 2019-11-14 23:00; first dev feedback cutoff.


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def select_matured_rows(store, limit=VALIDATION_IMPRESSIONS):
    """Sample from the half-open mature window before reading candidate labels."""
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 1:
        raise ValueError("validation limit must be a positive integer")
    split, timestamps = np.asarray(store.split), np.asarray(store.timestamps)
    if split.ndim != 1 or timestamps.shape != split.shape:
        raise ValueError("split and timestamps must be matching vectors")
    eligible = np.flatnonzero((split == 1) & (timestamps >= MIN_VALID_TIME)
                             & (timestamps < MAX_VALID_TIME))
    if len(eligible) < limit:
        raise ValueError("not enough matured internal-validation impressions for the fixed sample")
    if len(eligible) > limit:
        eligible = np.sort(np.random.default_rng(SAMPLING_SEED).choice(eligible, limit, replace=False))
    return np.asarray(eligible, dtype=np.int64)


def checkpoint_evidence(checkpoint, *, seed, state_sha256, feature_dim,
                        manifest_sha256, source_sha256, cache_signature,
                        train_cutoff_timestamp, eligible_positives):
    """Check actual epoch-2 state identity, not the old validation winner score."""
    if seed not in SEEDS or checkpoint.get("seed") != seed:
        raise ValueError("unexpected backbone seed")
    if checkpoint.get("variant") != "mean_random" or checkpoint.get("kind") != "tower":
        raise ValueError("only mean_random tower checkpoints are eligible")
    expected_config = {"feature_dim": feature_dim, "hidden": 128,
                       "embedding_dim": 64, "user_pooling": "mean"}
    if checkpoint.get("config") != expected_config:
        raise ValueError("backbone configuration mismatch")
    training = checkpoint.get("training_configuration", {})
    if training.get("epochs") != 2 or checkpoint.get("best_epoch") != 2:
        raise ValueError("checkpoint must contain the final state of exactly two fixed epochs")
    if checkpoint.get("trained_on_all_eligible_clicks") is not True:
        raise ValueError("checkpoint is not a complete-data training run")
    if eligible_positives < 1 or checkpoint.get("eligible_positive_examples_per_epoch") != eligible_positives:
        raise ValueError("eligible-positive training population mismatch")
    history = checkpoint.get("history", [])
    if len(history) != 2 or [item.get("epoch") for item in history] != [1, 2]:
        raise ValueError("checkpoint requires both complete epoch records")
    if any(item.get("training_examples") != eligible_positives for item in history):
        raise ValueError("a checkpoint epoch did not consume every eligible training positive")
    if any(item.get("negative_policy") != "four_random" for item in history):
        raise ValueError("checkpoint must use random negatives in both epochs")
    if state_sha256 != history[1].get("state_sha256"):
        raise ValueError("loaded checkpoint state digest does not match epoch 2")
    if checkpoint.get("data_manifest_sha256") != manifest_sha256:
        raise ValueError("checkpoint data manifest mismatch")
    if checkpoint.get("training_source_sha256") != source_sha256:
        raise ValueError("checkpoint training source mismatch")
    if checkpoint.get("training_cache_signature") != cache_signature:
        raise ValueError("checkpoint training cache mismatch")
    if checkpoint.get("train_cutoff_timestamp") != train_cutoff_timestamp:
        raise ValueError("checkpoint training cutoff mismatch")
    if train_cutoff_timestamp >= MIN_VALID_TIME - 3600:
        raise ValueError("training feedback is not mature at the first allowed validation request")
    return {
        "epoch": 2, "state_sha256": state_sha256, "state_matches_epoch2": True,
        "complete_training_epochs": 2, "training_examples_per_epoch": eligible_positives,
        "training_configuration": training,
        "epoch_state_sha256": [item["state_sha256"] for item in history],
    }


def select_winner(candidates):
    """Higher matured-window nDCG@10 wins; exact ties use the lower seed."""
    if sorted(item.get("seed") for item in candidates) != list(SEEDS):
        raise ValueError("selection requires exactly seeds 17, 29 and 43")
    for item in candidates:
        value = item["metrics"]["ndcg10"]["mean"]
        if not np.isfinite(value):
            raise ValueError("selection nDCG@10 must be finite")
    return max(candidates, key=lambda item: (item["metrics"]["ndcg10"]["mean"], -item["seed"]))


def current_backbone_audit(checkpoint, *, checkpoint_path, checkpoint_sha256,
                           state_sha256, selected):
    """Different serialized checkpoint bytes may still hold identical tensors."""
    matches = state_sha256 == selected["state_sha256"]
    return {
        "current_retriever_audit": {
            "checkpoint_path": Path(checkpoint_path).as_posix(),
            "current_file_sha256": checkpoint_sha256,
            "variant": checkpoint["variant"], "seed": checkpoint["seed"],
            "best_epoch": checkpoint["best_epoch"], "state_sha256": state_sha256,
            "selected_state_matches_current": matches,
        },
        "current_backbone": {
            "checkpoint_path": Path(checkpoint_path).as_posix(),
            "checkpoint_sha256": checkpoint_sha256, "state_sha256": state_sha256,
        },
        "matches_current_retriever_state": matches,
    }


def run_selection(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing backbone selection report")
    if args.threads != 1:
        raise ValueError("formal backbone reselection is fixed to one CPU thread")
    # Lazy imports keep boundary/provenance tests independent of Torch startup.
    import torch
    from threadpoolctl import threadpool_limits
    from ..evaluation_scheduler import evaluate_scheduled
    from ..store import Store
    from ..train import build_model, state_digest, training_cache_signature, training_source_digests

    torch.set_num_threads(1)
    data, artifacts = Path(args.data), Path(args.artifacts)
    required_cache = ("training_cache.json", "negative_pool.npy", "positive_rows.npy",
                      "positive_items.npy", "mean_history.npy")
    if any(not (data / name).is_file() for name in required_cache):
        raise FileNotFoundError("Existing verified training caches are required; this command does not build them")
    started = time.perf_counter()
    store = Store(data)
    store.ensure_training_cache()
    rows = select_matured_rows(store)
    manifest_sha = sha256_file(data / "manifest.json")
    source_digests = training_source_digests()
    cache_signature = training_cache_signature(store)
    cutoff = int(store.manifest["train_cutoff_timestamp"])
    candidates, loaded = [], []
    with threadpool_limits(limits=1):
        # Validate every checkpoint before evaluating any labels.
        for seed in SEEDS:
            path = artifacts / f"mean_random_seed{seed}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model, _, _ = build_model("mean_random", store.features.shape[1])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            evidence = checkpoint_evidence(
                checkpoint, seed=seed, state_sha256=state_digest(model),
                feature_dim=store.features.shape[1], manifest_sha256=manifest_sha,
                source_sha256=source_digests, cache_signature=cache_signature,
                train_cutoff_timestamp=cutoff, eligible_positives=len(store.positive_rows),
            )
            loaded.append((seed, path, model, evidence, sha256_file(path)))
        if any(item[3]["training_configuration"] != loaded[0][3]["training_configuration"] for item in loaded):
            raise ValueError("backbone seeds must have the same complete training configuration")
        current_path = artifacts / "retriever.pt"
        current_checkpoint = torch.load(current_path, map_location="cpu", weights_only=True)
        current_model, _, _ = build_model("mean_random", store.features.shape[1])
        current_model.load_state_dict(current_checkpoint["state_dict"], strict=True)
        current_state_sha = state_digest(current_model)
        checkpoint_evidence(
            current_checkpoint, seed=current_checkpoint.get("seed"), state_sha256=current_state_sha,
            feature_dim=store.features.shape[1], manifest_sha256=manifest_sha,
            source_sha256=source_digests, cache_signature=cache_signature,
            train_cutoff_timestamp=cutoff, eligible_positives=len(store.positive_rows),
        )
        current_file_sha = sha256_file(current_path)
        for seed, path, model, evidence, checkpoint_sha in loaded:
            evaluation, values = evaluate_scheduled(store, model, "tower", rows, batch_size=128)
            if values.shape != (VALIDATION_IMPRESSIONS, 4) or not np.isfinite(values[:, 3]).all():
                raise ValueError("full-candidate selection requires finite nDCG@10 for every sampled impression")
            record = {"seed": seed, "checkpoint_path": path.as_posix(),
                      "checkpoint_sha256": checkpoint_sha, **evidence,
                      "metrics": evaluation["metrics"], "full_candidate_evaluation": evaluation}
            candidates.append(record)
            print(json.dumps({"seed": seed, "matured_validation_ndcg10": evaluation["metrics"]["ndcg10"]["mean"]}), flush=True)
    winner = select_winner(candidates)
    result = {
        "schema_version": 1,
        "scope": "matured internal-validation only; no official-dev evaluation; no model fitting or artifact replacement",
        "selection_metric": "impression-macro full-candidate ndcg10",
        "tie_break": "lower seed for exactly equal nDCG@10",
        "seeds": list(SEEDS),
        "validation": {
            "split": 1, "requested_impressions": VALIDATION_IMPRESSIONS,
            "impressions": len(rows), "seed": SAMPLING_SEED,
            "min_timestamp_inclusive": MIN_VALID_TIME, "max_timestamp_exclusive": MAX_VALID_TIME,
            "row_indices_sha256": hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest(),
            "row_index_dtype": "int64", "row_order": "ascending original impression row",
            "candidate_exposures": int((store.offsets[rows + 1] - store.offsets[rows]).sum()),
            "observed_min_timestamp": int(store.timestamps[rows].min()),
            "observed_max_timestamp": int(store.timestamps[rows].max()),
            "feedback_lag_seconds": 3600,
        },
        "selected": {key: winner[key] for key in (
            "seed", "checkpoint_path", "checkpoint_sha256", "state_sha256", "epoch", "metrics")},
        **current_backbone_audit(
            current_checkpoint, checkpoint_path=current_path, checkpoint_sha256=current_file_sha,
            state_sha256=current_state_sha, selected=winner,
        ),
        "candidates": candidates,
        "data_manifest_sha256": manifest_sha,
        "training_source_sha256": source_digests,
        "training_cache_signature": cache_signature,
        "train_cutoff_timestamp": cutoff,
        "source_sha256": sha256_file(__file__),
        "evaluation_scheduler_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "evaluation_scheduler.py"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "threads": 1, "device": "cpu"},
        "seconds": time.perf_counter() - started,
        "limitations": [
            "The old v1 validation criterion chose the saved epoch, but all three actual states must equal the recorded final epoch 2; epoch number is not retuned here.",
            "Unchanged provenance-matched v1 fit uses a fixed two-epoch loop and fixed AdamW learning rate; validation affects checkpoint retention only, with no early stopping or learning-rate scheduling.",
            "Only these three existing fixed-budget seeds are eligible; no additional training or tuning is performed.",
            "The v1 benchmark has already been observed; this is follow-up research, not an untouched benchmark claim.",
            "Time is a deterministic coordinate for the source logs; the source timezone is not asserted.",
            "Selection does not modify retriever.pt, article_embeddings.npy, or a NumPy serving encoder; exports must explicitly bind the selected checkpoint.",
        ],
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(json.dumps({"selected_seed": winner["seed"], "output": output.as_posix()}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/v2/backbone_matured_selection.json"))
    parser.add_argument("--threads", type=int, choices=[1], default=1)
    run_selection(parser.parse_args())


if __name__ == "__main__":
    main()
