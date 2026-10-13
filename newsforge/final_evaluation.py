"""Evaluate completed, preselected experiments with an audited ranker schedule.

This entry point never trains, selects new checkpoints, or rewrites serving
artifacts. All three rankers must pass the fixed internal-validation scheduling
comparison before any official-dev labels are evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from . import evaluation_scheduler
from .evaluate import evaluate, select_validation_rows
from .evaluation_scheduler import _profile, evaluate_scheduled
from .metrics import METRICS, user_cluster_interval
from .store import Store
from .train import build_model, state_digest, training_cache_signature, training_source_digests, write_json


VARIANTS = ["mean_random", "mean_mixed", "ranker"]
SEEDS = [17, 29, 43]
SCHEDULING_VERSION = "stable_candidate_count_v1"
TOLERANCE = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_digest(rows):
    return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()


def _metadata(checkpoint):
    return {key: value for key, value in checkpoint.items() if key != "state_dict"}


def verify_checkpoints(store, artifacts, reports, validation_rows):
    """Check all nine full-epoch records and already-exported serving weights."""
    expected_config = {"epochs": 2, "batch_size": 1024, "threads": 1, "learning_rate": .001,
                       "validation_impressions": 12000, "negative_pool_size": 16}
    source_hashes = training_source_digests()
    cache_signature = training_cache_signature(store)
    manifest_hash = sha256(store.path / "manifest.json")
    eligible = len(store.positive_rows)
    if store.pool.shape[1] != 16 or len(validation_rows) != 12000:
        raise ValueError("final evaluation requires the declared pool16 / validation12000 protocol")
    validation_candidates = int((store.offsets[validation_rows + 1] - store.offsets[validation_rows]).sum())
    checkpoints = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = artifacts / f"{variant}_seed{seed}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model, config, kind = build_model(variant, store.features.shape[1])
            expected_run = f"{variant}-seed{seed}"
            if (checkpoint.get("run_id") != expected_run or checkpoint.get("variant") != variant
                    or checkpoint.get("seed") != seed or checkpoint.get("kind") != kind
                    or checkpoint.get("config") != config
                    or checkpoint.get("training_configuration") != expected_config
                    or checkpoint.get("training_source_sha256") != source_hashes
                    or checkpoint.get("training_cache_signature") != cache_signature
                    or checkpoint.get("data_manifest_sha256") != manifest_hash
                    or checkpoint.get("train_cutoff_timestamp") != int(store.manifest["train_cutoff_timestamp"])
                    or checkpoint.get("trained_on_all_eligible_clicks") is not True
                    or checkpoint.get("eligible_positive_examples_per_epoch") != eligible):
                raise ValueError(f"checkpoint provenance/configuration mismatch: {expected_run}")
            history = checkpoint.get("history", [])
            policies = ["four_random", "four_random" if variant == "mean_random" else "two_hard_two_random"]
            if (len(history) != 2 or [entry["epoch"] for entry in history] != [1, 2]
                    or [entry["negative_policy"] for entry in history] != policies
                    or any(entry["training_examples"] != eligible for entry in history)
                    or any(entry["validation"]["impressions"] != len(validation_rows) for entry in history)
                    or any(entry["validation"]["candidate_exposures"] != validation_candidates for entry in history)):
                raise ValueError(f"incomplete full-epoch training evidence: {expected_run}")
            internal_scores = [entry["validation"]["metrics"]["ndcg10"]["mean"] for entry in history]
            best_epoch = int(np.argmax(internal_scores)) + 1
            if (not np.isfinite(internal_scores).all() or checkpoint["best_epoch"] != best_epoch
                    or not np.isclose(checkpoint["validation_ndcg10"], max(internal_scores), atol=TOLERANCE, rtol=0)):
                raise ValueError(f"saved epoch does not match its original internal selection: {expected_run}")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            if state_digest(model) != history[best_epoch - 1]["state_sha256"]:
                raise ValueError(f"saved weights differ from the recorded selected epoch: {expected_run}")
            training_record = json.loads((reports / f"{variant}_seed{seed}_training.json").read_text(encoding="utf-8"))
            if training_record != _metadata(checkpoint):
                raise ValueError(f"training JSON/checkpoint metadata mismatch: {expected_run}")
            checkpoints.append(checkpoint)
    warmup_checks = {}
    for seed in SEEDS:
        pair = [item for item in checkpoints if item["seed"] == seed and item["variant"] != "ranker"]
        digests = [item["history"][0]["state_sha256"] for item in pair]
        if len(pair) != 2 or digests[0] != digests[1]:
            raise ValueError(f"random/mixed warmup differs for seed {seed}")
        warmup_checks[str(seed)] = {"identical": True, "sha256": digests}
    if json.loads((reports / "matched_warmup_checks.json").read_text(encoding="utf-8")) != warmup_checks:
        raise ValueError("existing matched-warmup report differs from the completed checkpoints")
    serving = {}
    for kind, filename in (("tower", "retriever.pt"), ("ranker", "reranker.pt")):
        exported = torch.load(artifacts / filename, map_location="cpu", weights_only=True)
        # Verify the preexisting export followed train.py's documented rule;
        # do not select, overwrite, or export any checkpoint here.
        expected = max((item for item in checkpoints if item["kind"] == kind),
                       key=lambda item: item["validation_ndcg10"])
        if _metadata(exported) != _metadata(expected):
            raise ValueError(f"preselected serving metadata mismatch: {filename}")
        if (set(exported["state_dict"]) != set(expected["state_dict"])
                or any(not torch.equal(value, expected["state_dict"][name])
                       for name, value in exported["state_dict"].items())):
            raise ValueError(f"preselected serving weights mismatch: {filename}")
        serving[filename] = {"run_id": exported["run_id"], "sha256": sha256(artifacts / filename)}
    embeddings = np.load(artifacts / "article_embeddings.npy", mmap_mode="r", allow_pickle=False)
    try:
        if embeddings.shape != (len(store.news_ids), 64) or not np.isfinite(embeddings).all():
            raise ValueError("preexported article embeddings have invalid shape or values")
    finally:
        embeddings._mmap.close()
    serving["article_embeddings.npy"] = {"sha256": sha256(artifacts / "article_embeddings.npy")}
    return checkpoints, warmup_checks, serving, expected_config


def validation_cache_matches(record, expected):
    """Reuse only a passed full-12000 comparison bound to current identities."""
    try:
        if (any(record.get(key) != value for key, value in expected.items())
                or record.get("metric_equivalence_passed") is not True
                or record.get("per_impression_metric_equivalence_atol") != TOLERANCE
                or record.get("rows") != 12000 or record.get("fixed_validation_rows") != 12000
                or record["environment"]["threads"] != 1
                or record["environment"]["torch"] != torch.__version__
                or record["scheduling"]["strategy"] != "stable_candidate_count"
                or record["scheduling"]["batch_size"] != 32
                or record["scheduling"]["metric_rows_restored_to_original_order"] is not True):
            return False
        for metric in METRICS:
            difference = record["metric_differences"][metric]
            if (difference["same_nan_mask"] is not True or difference["pairs_differing_more_than_1e_12"] != 0
                    or not 0 <= difference["max_absolute_difference"] <= TOLERANCE):
                return False
        return True
    except (KeyError, TypeError):
        return False


def audit_ranker_scheduling(store, checkpoint, rows, artifacts, reports):
    """An internal-only gate; a failed audit aborts before dev is touched."""
    if len(rows) != 12000 or not np.all(store.split[rows] == 1):
        raise ValueError("scheduling audit requires exactly the fixed internal-validation rows")
    path = artifacts / f"ranker_seed{checkpoint['seed']}.pt"
    expected = {"run_id": checkpoint["run_id"], "selected_epoch": checkpoint["best_epoch"],
                "selected_row_indices_sha256": rows_digest(rows), "checkpoint_sha256": sha256(path),
                "data_manifest_sha256": sha256(store.path / "manifest.json"),
                "training_source_sha256": training_source_digests(),
                "scheduler_source_sha256": sha256(evaluation_scheduler.__file__)}
    output = reports / ("evaluation_scheduler_validation_12000.json" if checkpoint["seed"] == 17
                        else f"evaluation_scheduler_ranker_seed{checkpoint['seed']}_validation_12000.json")
    if output.is_file():
        record = json.loads(output.read_text(encoding="utf-8"))
        if validation_cache_matches(record, expected):
            return {"run_id": checkpoint["run_id"], "evidence_file": output.name,
                    "evidence_sha256": sha256(output), "reused": True, "passed": True}
        # Preserve an existing, failed or outdated diagnostic for auditability.
        raise ValueError(f"existing scheduling evidence is failed/stale; review it before proceeding: {output}")
    model, _, kind = build_model("ranker", store.features.shape[1])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    (original_report, original_values), original_cost = _profile(lambda: evaluate(store, model, kind, rows))
    (scheduled_report, scheduled_values), scheduled_cost = _profile(lambda: evaluate_scheduled(store, model, kind, rows))
    differences = {}
    for column, metric in enumerate(METRICS):
        first, second = original_values[:, column], scheduled_values[:, column]
        finite = np.isfinite(first) & np.isfinite(second)
        difference = np.abs(first[finite] - second[finite])
        differences[metric] = {"finite_pairs": int(finite.sum()),
                               "same_nan_mask": bool(np.array_equal(np.isnan(first), np.isnan(second))),
                               "max_absolute_difference": float(difference.max(initial=0)),
                               "pairs_differing_more_than_1e_12": int((difference > TOLERANCE).sum()),
                               "aggregate_mean_difference": (scheduled_report["metrics"][metric]["mean"]
                                                             - original_report["metrics"][metric]["mean"])}
    passed = bool(np.allclose(original_values, scheduled_values, atol=TOLERANCE, rtol=0, equal_nan=True))
    record = {**expected, "scope": "fixed internal validation only; no official-dev label reads",
              "rows": len(rows), "fixed_validation_rows": len(rows), "metric_equivalence_passed": passed,
              "per_impression_metric_equivalence_atol": TOLERANCE, "metric_differences": differences,
              "original": original_cost, "scheduled": scheduled_cost,
              "observed_wall_time_ratio_original_over_scheduled": original_cost["seconds"] / scheduled_cost["seconds"],
              "scheduling": scheduled_report["evaluation_scheduling"],
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "threads": 1},
              "limitations": ["One original-then-scheduled pair; shared desktop load and cache warmth are uncontrolled.",
                              "Float comparison covers each impression's four metrics, not candidate logits."]}
    write_json(output, record)
    if not passed:
        raise ValueError(f"scheduling changed metrics beyond 1e-12 for {checkpoint['run_id']}; official-dev evaluation refused")
    return {"run_id": checkpoint["run_id"], "evidence_file": output.name,
            "evidence_sha256": sha256(output), "reused": False, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--threads", type=int, choices=[1], default=1)
    parser.add_argument("--validate-only", action="store_true",
                        help="check all artifacts and three internal scheduling audits; do not evaluate official dev")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    threadpool_limits(args.threads)
    if not (args.data / "training_cache.json").is_file():
        raise ValueError("existing completed training cache is required; this entry does not build one")
    required = [args.artifacts / f"{variant}_seed{seed}.pt" for variant in VARIANTS for seed in SEEDS]
    required += [args.artifacts / name for name in ("retriever.pt", "reranker.pt", "article_embeddings.npy")]
    if any(not path.is_file() for path in required):
        raise ValueError("all nine completed checkpoints and existing serving exports are required before evaluation")
    store = Store(args.data)
    store.ensure_training_cache()
    validation_rows = select_validation_rows(store, 12000)
    checkpoints, warmup_checks, serving, training_config = verify_checkpoints(
        store, args.artifacts, args.output, validation_rows)
    audits = [audit_ranker_scheduling(store, checkpoint, validation_rows, args.artifacts, args.output)
              for checkpoint in checkpoints if checkpoint["kind"] == "ranker"]
    scheduling = {"version": SCHEDULING_VERSION, "scheduler_source_sha256": sha256(evaluation_scheduler.__file__),
                  "entrypoint_source_sha256": sha256(__file__), "ranker_only": True,
                  "ranker_batch_size": 32, "internal_validation_audits": audits,
                  "per_impression_equivalence_atol": TOLERANCE,
                  "original_metric_row_order_restored": True,
                  "training_and_checkpoint_selection_unchanged": True}
    write_json(args.output / "final_evaluation_preflight.json", {
        "status": "passed", "completed_checkpoint_count": len(checkpoints),
        "data_manifest_sha256": sha256(store.path / "manifest.json"),
        "training_source_sha256": training_source_digests(),
        "training_cache_signature": training_cache_signature(store),
        "serving_artifacts": serving, "evaluation_scheduling": scheduling,
        "official_dev_labels_evaluated_by_preflight": False})
    if args.validate_only:
        print(json.dumps({"status": "preflight_passed", "official_dev_evaluated": False}), flush=True)
        return
    # The dev label-reading path starts only after every provenance and
    # internal-equivalence gate above has passed. Population is never sampled.
    test_rows = np.flatnonzero(store.split == 2)
    if len(test_rows) != store.manifest["official_dev_rows"] or not len(test_rows):
        raise ValueError("official-dev population differs from the completed data manifest")
    np.save(args.artifacts / "evaluation_rows.npy", test_rows)
    reports, matrices = {}, {}
    for baseline in ("popularity", "content"):
        report, values = evaluate(store, None, baseline, test_rows,
                                  output=args.artifacts / f"metrics_{baseline}.npy")
        reports[baseline], matrices[baseline] = report, values
        print(json.dumps({"baseline": baseline, "metrics": report["metrics"]}), flush=True)
    for checkpoint in checkpoints:
        model, _, kind = build_model(checkpoint["variant"], store.features.shape[1])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        run_id = checkpoint["run_id"]
        evaluator = evaluate_scheduled if kind == "ranker" else evaluate
        report, values = evaluator(store, model, kind, test_rows,
                                   output=args.artifacts / f"metrics_{run_id}.npy")
        reports[run_id], matrices[run_id] = report, values
        print(json.dumps({"run_id": run_id, "metrics": report["metrics"], "seconds": report["seconds"]}), flush=True)
        write_json(args.output / "ranking_partial.json", reports)
    aggregate = {}
    for variant in VARIANTS:
        names = [f"{variant}-seed{seed}" for seed in SEEDS]
        aggregate[variant] = {metric: {
            "mean": float(np.mean([reports[name]["metrics"][metric]["mean"] for name in names])),
            "std": float(np.std([reports[name]["metrics"][metric]["mean"] for name in names], ddof=1))}
            for metric in METRICS}
    base = np.mean([matrices[f"mean_random-seed{seed}"] for seed in SEEDS], axis=0)
    comparisons = {}
    for variant in ("mean_mixed", "ranker"):
        values = np.mean([matrices[f"{variant}-seed{seed}"] for seed in SEEDS], axis=0)
        comparisons[variant] = user_cluster_interval(base, values, store.users[test_rows])
    configuration = {"data": str(args.data), "artifacts": str(args.artifacts), "output": str(args.output),
                     "variants": VARIANTS, "seeds": SEEDS, "epochs": training_config["epochs"],
                     "batch_size": training_config["batch_size"], "learning_rate": training_config["learning_rate"],
                     "threads": args.threads, "validation_impressions": training_config["validation_impressions"],
                     "smoke_steps": 0, "skip_test": False, "evaluation_only": True}
    scheduling["original_evaluation_rows_sha256"] = rows_digest(test_rows)
    result = {"configuration": configuration,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "platform": platform.platform(), "threads": args.threads},
              "source_sha256": {f"newsforge/{path.name}": sha256(path)
                                for path in sorted(Path(__file__).parent.glob("*.py"))},
              "runs": reports, "seed_summary": aggregate, "paired_user_bootstrap": comparisons,
              "matched_warmup_checks": warmup_checks,
              "bootstrap_estimand": "paired difference in per-impression metrics averaged over seeds; not score ensemble",
              "evaluation": "Official MIND-large development set used only after all internal-validation selections.",
              "evaluation_scheduling": scheduling, "serving_artifacts": serving}
    write_json(args.output / "ranking_experiment.json", result)


if __name__ == "__main__":
    main()
