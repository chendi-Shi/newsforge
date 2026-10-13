"""Evaluate the preselected GBDT only after bound internal scheduling checks.

Default/--validate-only reads training and internal-validation evidence only.
Official-dev evaluation requires --evaluate-test and never fits a model.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .evaluate import select_validation_rows
from .gbdt import FEATURE_VERSION, _manifest_signature, feature_names
from .gbdt_evaluation_scheduler import evaluate_scheduled_gbdt
from .metrics import METRICS
from .store import Store


EVIDENCE_FILE = "gbdt_evaluation_scheduler_validation_12000.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_digest(rows):
    return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_internal_audit(record, expected):
    """Validate the full paired comparison, including identities and NaN masks."""
    if (any(record.get(key) != value for key, value in expected.items())
            or record.get("rows") != 12000 or record.get("fixed_validation_rows") != 12000
            or record.get("metric_equivalence_passed") is not True
            or record.get("per_impression_metric_equivalence_atol") != 1e-12
            or record["environment"]["threads"] != 1):
        raise ValueError("GBDT internal scheduling audit is stale, incomplete or bound to different artifacts")
    for metric in METRICS:
        difference = record["metric_differences"][metric]
        if (difference["same_nan_mask"] is not True
                or difference["pairs_differing_more_than_1e_12"] != 0
                or not 0 <= difference["max_absolute_difference"] <= 1e-12
                or not 0 < difference["finite_pairs"] <= 12000):
            raise ValueError(f"GBDT scheduling metric equivalence failed: {metric}")
    plan = record["scheduling"]
    if (plan["strategy"] != "stable_candidate_count" or plan["gbdt_only"] is not True
            or plan["batch_size"] != 64 or plan["impressions"] != 12000
            or any(plan[key] is not True for key in
                ("same_population", "same_candidates_within_each_impression", "metric_rows_restored_to_original_order"))
            or not 0 < plan["candidate_exposures"] <= min(plan["original_candidate_slots"], plan["scheduled_candidate_slots"])):
        raise ValueError("GBDT scheduling changed population, candidates, batch size or row order")
    if any(not np.isfinite(record[key]["seconds"]) or record[key]["seconds"] <= 0
           for key in ("original", "scheduled")):
        raise ValueError("GBDT paired scheduling timings are invalid")


def verify_artifacts(store, artifacts, reports):
    """Read-only preflight: preserve the original feature selection and weights."""
    artifacts, reports = Path(artifacts), Path(reports)
    code = Path(__file__).resolve().parent
    training = read_json(reports / "gbdt_training.json")
    selection = read_json(reports / "gbdt_feature_ablation.json")
    cache = read_json(store.path / "training_cache.json")
    manifest_hash = sha256(store.path / "manifest.json")
    feature_hash = sha256(store.path / "news_features.npy")
    configurations = selection["configurations"]
    if set(configurations) != {"similarity", "rich"}:
        raise ValueError("Both original GBDT feature configurations must remain available")
    winner = max(("similarity", "rich"), key=lambda name:
                 configurations[name]["internal_validation"]["metrics"]["ndcg10"]["mean"])
    if (winner != "rich" or selection["selected_feature_mode"] != winner
            or selection["official_development_test_evaluated"] is not False
            or training["feature_mode"] != winner or training["official_development_test_evaluated"] is not False):
        raise ValueError("Canonical GBDT must preserve the internally selected rich model")
    validation_rows = select_validation_rows(store, 12000)
    row_hash = rows_digest(validation_rows)
    if len(validation_rows) != 12000 or selection["shared_validation_row_indices_sha256"] != row_hash:
        raise ValueError("GBDT feature selection uses a different internal-validation population")
    for mode, record in configurations.items():
        if (sha256(artifacts / f"gbdt_{mode}.joblib") != record["preserved_model_sha256"]
                or record["sampling"]["selected_positive_indices_sha256"] != selection["shared_selected_positive_indices_sha256"]
                or not np.array_equal(np.load(artifacts / f"evaluation_rows_gbdt_{mode}_validation.npy"), validation_rows)):
            raise ValueError(f"Original GBDT feature comparison evidence changed: {mode}")
        values = np.load(artifacts / f"metrics_gbdt_{mode}_validation.npy")
        if values.shape != (12000, len(METRICS)):
            raise ValueError("GBDT internal metric matrix shape is inconsistent")
        for column, metric in enumerate(METRICS):
            if not np.isclose(np.nanmean(values[:, column]), record["internal_validation"]["metrics"][metric]["mean"], rtol=0, atol=1e-12):
                raise ValueError("GBDT feature selection scores do not match preserved metric arrays")
    model_hash = sha256(artifacts / "gbdt.joblib")
    if model_hash != configurations[winner]["preserved_model_sha256"]:
        raise ValueError("Canonical GBDT weights differ from the preserved selected model")
    artifact = joblib.load(artifacts / "gbdt.joblib")
    schema = feature_names(store.features.shape[1], int(store.manifest["features"]["text_dimensions"]), winner)
    config = dict(max_iter=80, learning_rate=.1, max_leaf_nodes=31,
                  l2_regularization=1., early_stopping=False, random_state=2026)
    sampling = training["sampling"]
    if (artifact["feature_mode"] != winner or artifact["feature_version"] != FEATURE_VERSION
            or tuple(artifact["feature_names"]) != schema or tuple(training["feature_names"]) != schema
            or training["feature_count"] != len(schema)
            or artifact["manifest_sha256"] != _manifest_signature(store)
            or training["data_manifest_sha256"] != _manifest_signature(store)
            or artifact["config"] != config or training["config"] != config
            or artifact["model"].n_iter_ != 80 or artifact["model"].n_features_in_ != len(schema)
            or any(artifact["model"].get_params()[key] != value for key, value in config.items())
            or artifact["sampling"] != sampling
            or sampling["selected_positive_pairs"] != min(500000, cache["positive_training_examples"])
            or sampling["binary_training_examples"] != sampling["selected_positive_pairs"] * 5
            or sampling["seed"] != 2026 or sampling["negatives_per_positive"] != 4
            or sampling["selected_positive_indices_sha256"] != selection["shared_selected_positive_indices_sha256"]
            or sampling["source_training_cache"] != cache
            or cache["data_manifest_sha256"] != manifest_hash or cache["features_sha256"] != feature_hash
            or artifact["validation_row_indices_sha256"] != row_hash
            or training["validation_row_indices_sha256"] != row_hash
            or training["source_sha256"] != sha256(code / "gbdt.py")):
        raise ValueError("Selected GBDT model/data/features/training budget no longer match their evidence")
    expected = {"checkpoint_sha256": model_hash, "data_manifest_sha256": manifest_hash,
        "selected_row_indices_sha256": row_hash, "feature_mode": winner, "feature_count": len(schema),
        "evaluation_source_sha256": {name: sha256(code / name) for name in ("gbdt.py", "store.py", "evaluate.py", "metrics.py")},
        "scheduler_source_sha256": sha256(code / "gbdt_evaluation_scheduler.py"),
        "order_helper_source_sha256": sha256(code / "evaluation_scheduler.py")}
    evidence_path = reports / EVIDENCE_FILE
    validate_internal_audit(read_json(evidence_path), expected)
    preflight = {"status": "passed", **expected, "features_sha256": feature_hash,
        "selection_evidence_sha256": sha256(reports / "gbdt_feature_ablation.json"),
        "internal_validation_evidence_file": EVIDENCE_FILE,
        "internal_validation_evidence_sha256": sha256(evidence_path),
        "entrypoint_source_sha256": sha256(__file__), "official_dev_evaluated_by_preflight": False}
    return artifact, preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="default: verify artifacts, never evaluate dev")
    mode.add_argument("--evaluate-test", action="store_true", help="explicitly evaluate the full official-dev population once")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--threads", type=int, choices=(1,), default=1)
    args = parser.parse_args()
    store = Store(args.data)
    artifact, preflight = verify_artifacts(store, args.artifacts, args.reports)
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "gbdt_final_evaluation_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if not args.evaluate_test:
        print(json.dumps({"status": "preflight_passed", "official_dev_evaluated": False}), flush=True)
        return
    if (args.reports / "gbdt_evaluation.json").exists() or (args.artifacts / "metrics_gbdt.npy").exists():
        raise ValueError("Existing final GBDT evaluation found; refusing an implicit overwrite")
    rows = np.flatnonzero(store.split == 2)
    if len(rows) != store.manifest["official_dev_rows"] or not len(rows):
        raise ValueError("Official-dev population does not match the data manifest")
    with threadpool_limits(args.threads):
        report, _ = evaluate_scheduled_gbdt(store, artifact, rows, allow_test=True,
            batch_size=64, output=args.artifacts / "metrics_gbdt.npy")
    np.save(args.artifacts / "evaluation_rows_gbdt.npy", rows)
    report.update({"model": "HistGradientBoostingClassifier", "artifact": str(args.artifacts / "gbdt.joblib"),
        "official_development_test_evaluated": True, "threads": args.threads,
        "evaluation_preflight": preflight})
    report["evaluation_scheduling"].update({"original_evaluation_rows_sha256": rows_digest(rows),
        "internal_validation_evidence_file": EVIDENCE_FILE,
        "internal_validation_evidence_sha256": preflight["internal_validation_evidence_sha256"]})
    (args.reports / "gbdt_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["metrics"]), flush=True)


if __name__ == "__main__":
    main()
