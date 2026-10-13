"""Reproduce the fixed 12,000-row internal GBDT scheduling audit.

This command never fits, selects, or evaluates official dev. It preserves the
canonical feature-selected model and refuses to overwrite an existing report.
Use --output to save an additional diagnostic without replacing formal evidence.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from .evaluate import select_validation_rows
from .evaluation_scheduler import _profile
from .gbdt import FEATURE_VERSION, _manifest_signature, evaluate_gbdt, feature_names
from .gbdt_evaluation_scheduler import evaluate_scheduled_gbdt
from .gbdt_final_evaluation import (
    EVIDENCE_FILE, read_json, rows_digest, sha256, validate_internal_audit,
)
from .metrics import METRICS
from .store import Store


def verify_inputs(store, artifacts, reports):
    """Verify source artifacts without requiring the audit this CLI produces.

The final evaluator's verifier additionally requires an existing scheduler
audit. Keep the checks here independent of that file to support a clean run.
The frozen final evaluator repeats these checks before it can read dev labels.
"""
    artifacts, reports = Path(artifacts), Path(reports)
    code = Path(__file__).resolve().parent
    training = read_json(reports / "gbdt_training.json")
    selection = read_json(reports / "gbdt_feature_ablation.json")
    cache = read_json(store.path / "training_cache.json")
    manifest_hash = sha256(store.path / "manifest.json")
    feature_hash = sha256(store.path / "news_features.npy")
    configurations = selection["configurations"]
    if set(configurations) != {"similarity", "rich"}:
        raise ValueError("Both preserved GBDT feature configurations are required")
    winner = max(("similarity", "rich"), key=lambda name:
                 configurations[name]["internal_validation"]["metrics"]["ndcg10"]["mean"])
    if (winner != "rich" or selection["selected_feature_mode"] != winner
            or selection["official_development_test_evaluated"] is not False
            or training["feature_mode"] != winner
            or training["official_development_test_evaluated"] is not False):
        raise ValueError("Canonical GBDT must be the preserved internally selected rich model")
    rows = select_validation_rows(store, 12000)
    row_hash = rows_digest(rows)
    if (len(rows) != 12000 or not np.all(store.split[rows] == 1)
            or selection["shared_validation_row_indices_sha256"] != row_hash):
        raise ValueError("Only the fixed 12,000 internal-validation rows are permitted")
    for mode, record in configurations.items():
        if (sha256(artifacts / f"gbdt_{mode}.joblib") != record["preserved_model_sha256"]
                or record["sampling"]["selected_positive_indices_sha256"]
                != selection["shared_selected_positive_indices_sha256"]
                or not np.array_equal(np.load(artifacts / f"evaluation_rows_gbdt_{mode}_validation.npy"), rows)):
            raise ValueError(f"Preserved GBDT feature comparison changed: {mode}")
        values = np.load(artifacts / f"metrics_gbdt_{mode}_validation.npy")
        if values.shape != (12000, len(METRICS)):
            raise ValueError("GBDT internal metric matrix shape is inconsistent")
        for column, metric in enumerate(METRICS):
            if not np.isclose(np.nanmean(values[:, column]),
                    record["internal_validation"]["metrics"][metric]["mean"], rtol=0, atol=1e-12):
                raise ValueError("GBDT selection scores differ from preserved validation arrays")
    checkpoint = artifacts / "gbdt.joblib"
    model_hash = sha256(checkpoint)
    if model_hash != configurations[winner]["preserved_model_sha256"]:
        raise ValueError("Canonical GBDT differs from the preserved selected weights")
    artifact = joblib.load(checkpoint)
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
        raise ValueError("GBDT model/data/features/training budget no longer match their evidence")
    expected = {"checkpoint_sha256": model_hash, "data_manifest_sha256": manifest_hash,
        "selected_row_indices_sha256": row_hash, "feature_mode": winner, "feature_count": len(schema),
        "evaluation_source_sha256": {name: sha256(code / name)
            for name in ("gbdt.py", "store.py", "evaluate.py", "metrics.py")},
        "scheduler_source_sha256": sha256(code / "gbdt_evaluation_scheduler.py"),
        "order_helper_source_sha256": sha256(code / "evaluation_scheduler.py")}
    # Bind files again after both evaluations so concurrent replacement cannot
    # silently produce evidence against stale input or implementation identities.
    paths = [checkpoint, store.path / "manifest.json", store.path / "news_features.npy",
             store.path / "training_cache.json", reports / "gbdt_training.json",
             reports / "gbdt_feature_ablation.json", Path(__file__).resolve()]
    paths += [code / name for name in expected["evaluation_source_sha256"]]
    paths += [code / "gbdt_evaluation_scheduler.py", code / "evaluation_scheduler.py"]
    bindings = {path: sha256(path) for path in paths}
    provenance = {"features_sha256": feature_hash,
        "selection_evidence_sha256": bindings[reports / "gbdt_feature_ablation.json"],
        "training_report_sha256": bindings[reports / "gbdt_training.json"],
        "training_cache_sha256": bindings[store.path / "training_cache.json"],
        "benchmark_entrypoint_source_sha256": bindings[Path(__file__).resolve()],
        "official_development_test_evaluated": False}
    return artifact, rows, expected, provenance, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--threads", type=int, choices=(1,), default=1)
    parser.add_argument("--output", type=Path,
                        help=f"new output path; default: --reports/{EVIDENCE_FILE}")
    args = parser.parse_args()
    output = args.output if args.output is not None else args.reports / EVIDENCE_FILE
    if output.exists():
        parser.error(f"Refusing to overwrite existing evidence: {output}; use --output with a new path")
    available = psutil.virtual_memory().available
    if available < 1024 * 2**20:
        raise MemoryError(f"GBDT diagnostic requires at least 1 GiB available RAM; current={available}")
    store = Store(args.data)
    artifact, rows, expected, provenance, bindings = verify_inputs(store, args.artifacts, args.reports)
    print(json.dumps({"stage": "provenance_passed", "rows": len(rows),
                      "feature_count": len(artifact["feature_names"]), "official_dev_evaluated": False}), flush=True)
    with threadpool_limits(args.threads):
        (original, original_values), original_cost = _profile(lambda:
            evaluate_gbdt(store, artifact, rows, batch_size=64, allow_test=False))
        print(json.dumps({"stage": "original_complete", **original_cost}), flush=True)
        (scheduled, scheduled_values), scheduled_cost = _profile(lambda:
            evaluate_scheduled_gbdt(store, artifact, rows, batch_size=64, allow_test=False))
    if any(sha256(path) != digest for path, digest in bindings.items()):
        raise ValueError("GBDT benchmark inputs or source changed during evaluation; evidence not written")
    differences = {}
    for column, metric in enumerate(METRICS):
        first, second = original_values[:, column], scheduled_values[:, column]
        finite = np.isfinite(first) & np.isfinite(second)
        difference = np.abs(first[finite] - second[finite])
        differences[metric] = {"finite_pairs": int(finite.sum()),
            "same_nan_mask": bool(np.array_equal(np.isnan(first), np.isnan(second))),
            "max_absolute_difference": float(difference.max(initial=0)),
            "pairs_differing_more_than_1e_12": int((difference > 1e-12).sum()),
            "aggregate_mean_difference": scheduled["metrics"][metric]["mean"] - original["metrics"][metric]["mean"]}
    passed = bool(np.allclose(original_values, scheduled_values, atol=1e-12, rtol=0, equal_nan=True))
    result = {"scope": "fixed internal validation only; original GBDT model/features/evaluator unchanged",
        "rows": len(rows), "fixed_validation_rows": len(rows),
        "row_selection": "first rows in the sorted fixed seed-2026 internal-validation sample",
        **expected, **provenance,
        "per_impression_metric_equivalence_atol": 1e-12, "metric_equivalence_passed": passed,
        "metric_differences": differences, "original": original_cost, "scheduled": scheduled_cost,
        "scheduling": scheduled["evaluation_scheduling"],
        "environment": {"threads": args.threads, "system_available_memory_before": available},
        "limitations": ["One original-then-scheduled pair on a shared desktop; load and cache warmth are uncontrolled.",
            "RSS is sampled in the actual process at 20 ms and includes mapped pages; it is not an exact allocation peak.",
            "Comparison is per-impression metrics, not feature matrices or classifier probabilities.",
            "No official-dev evaluation, model fitting or checkpoint selection occurred."]}
    if passed:
        validate_internal_audit(result, expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a concurrent run from replacing existing evidence.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"stage": "complete", "output": str(output),
        "metric_equivalence_passed": passed, "original_seconds": original_cost["seconds"],
        "scheduled_seconds": scheduled_cost["seconds"], "scheduling": result["scheduling"]}), flush=True)
    if not passed:
        raise SystemExit("GBDT scheduling metrics differ; do not adopt this schedule")


if __name__ == "__main__":
    main()
