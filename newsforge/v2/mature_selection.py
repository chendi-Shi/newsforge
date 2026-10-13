"""Re-select fixed HGB weights using feedback mature before the first dev query.

No fitting or tree changes occur here. The nine original 80-round models were
trained with early_stopping=False on training data alone. This follow-up keeps
their bytes immutable and replaces only the internal validation selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from newsforge.store import Store
from .experiment import (MIN_VALID_TIME, SEEDS, VARIANTS, dump, load_models,
                         model_entries, ranking_evaluate, row_hash)
from .ranker import MODEL_CONFIG, select_models
from .retrieval import sha256_file
from .temporal import TemporalStore


def mature_validation_rows(store, count=6000):
    first_dev = int(store.manifest["splits"]["2"]["min_timestamp"])
    cutoff = first_dev // 3600 * 3600 - 3600
    eligible = np.flatnonzero((store.split == 1) & (store.timestamps >= MIN_VALID_TIME)
                              & (store.timestamps < cutoff))
    if count < 1 or len(eligible) < count:
        raise ValueError("declared mature internal validation budget unavailable")
    rows = np.sort(np.random.default_rng(2026).choice(eligible, count, replace=False))
    selected_times = np.asarray(store.timestamps[rows])
    audit = {
        "validation_min_inclusive": MIN_VALID_TIME,
        "validation_max_exclusive": cutoff,
        "first_dev_timestamp": first_dev,
        "first_dev_snapshot_cutoff": cutoff,
        "sampled_timestamp_min": int(selected_times.min()),
        "sampled_timestamp_max": int(selected_times.max()),
        "sampled_rows": len(rows), "row_indices_sha256": row_hash(rows), "sampling_seed": 2026,
        "all_rows_split1": bool(np.all(store.split[rows] == 1)),
        "all_selection_feedback_strictly_before_first_dev_cutoff": bool(np.all(selected_times < cutoff)),
        "rule": "09:00 <= internal source timestamp < first dev hourly snapshot cutoff (23:00); equality excluded",
    }
    return rows, audit


def checked_entries(source_selection):
    entries = {}
    for entry in model_entries(source_selection):
        run = f'{entry["variant"]}-seed{entry["seed"]}'
        if run in entries:
            raise ValueError("duplicate original model identity")
        entries[run] = entry
    expected = {f"{variant}-seed{seed}" for variant in VARIANTS for seed in SEEDS}
    if set(entries) != expected:
        raise ValueError("mature selection requires exactly the original nine fixed models")
    return entries


def validate_fixed_models(source_selection, models):
    entries = checked_entries(source_selection)
    if set(models) != set(entries):
        raise ValueError("loaded model population differs from original nine weights")
    for run, entry in entries.items():
        ranker = models[run]
        expected_config = {**MODEL_CONFIG, "random_state": entry["seed"]}
        if (ranker.model_version != entry["model_sha256"]
                or ranker.variant != entry["variant"]
                or ranker.metadata.get("seed") != entry["seed"]
                or ranker.metadata.get("config") != expected_config
                or ranker.metadata.get("sampling") != source_selection["sampling"]
                or ranker.model.n_iter_ != 80 or ranker.model.early_stopping is not False):
            raise ValueError(f"model was not an unchanged fixed-budget training-only fit: {run}")
    if (source_selection["sampling"]["positive_pairs"] != 100000
            or source_selection["sampling"]["binary_examples"] != 500000):
        raise ValueError("original declared 100000 positive / 500000 binary-example budget required")
    return entries


def reselect(source_selection, models, evaluation, maturity_audit, before_hashes, after_hashes,
             source_selection_sha256):
    entries = validate_fixed_models(source_selection, models)
    expected_hashes = {run: entry["model_sha256"] for run, entry in entries.items()}
    if before_hashes != expected_hashes or after_hashes != expected_hashes:
        raise ValueError("model bytes changed during mature internal re-selection")
    if (evaluation["impressions"] != maturity_audit["sampled_rows"]
            or evaluation["row_indices_sha256"] != maturity_audit["row_indices_sha256"]
            or set(evaluation["metrics"]) != set(entries)
            or not maturity_audit["all_rows_split1"]
            or not maturity_audit["all_selection_feedback_strictly_before_first_dev_cutoff"]
            or not (maturity_audit["validation_min_inclusive"] <= maturity_audit["sampled_timestamp_min"]
                    <= maturity_audit["sampled_timestamp_max"] < maturity_audit["validation_max_exclusive"])
            or maturity_audit["validation_max_exclusive"] != maturity_audit["first_dev_snapshot_cutoff"]
            or maturity_audit["first_dev_snapshot_cutoff"] != maturity_audit["first_dev_timestamp"] // 3600 * 3600 - 3600):
        raise ValueError("mature evaluation population or timestamp boundary mismatch")
    records = {}
    for run, entry in entries.items():
        ranker = models[run]
        records[run] = {
            "run_id": run, "variant": entry["variant"], "seed": entry["seed"],
            "model_path": entry["model_path"], "model_sha256": entry["model_sha256"],
            "sampling": ranker.metadata["sampling"], "config": ranker.metadata["config"],
            "bindings": ranker.feature_builder.bindings,
            "internal_validation": {"row_indices_sha256": evaluation["row_indices_sha256"],
                                    "candidate_exposures": evaluation["candidates"],
                                    "metrics": evaluation["metrics"][run]},
        }
    selection = select_models(records)
    weight_audit = {
        "fit_invoked": False, "all_weights_unchanged": True,
        "model_sha256_before": before_hashes, "model_sha256_after": after_hashes,
        "fixed_max_iter": 80, "early_stopping": False,
        "source_selection_sha256": source_selection_sha256,
        "validation_role": "metric evaluation and identity selection only; no fitting, early stopping, checkpoint changes or hyperparameter changes",
    }
    selection.update({"maturity_audit": maturity_audit, "weight_reuse_audit": weight_audit,
                      "source_selection_sha256": source_selection_sha256,
                      "selection_protocol": "fixed nine training-only fits, reselected on 6000 internal requests with all selection feedback mature before the first dev query"})
    return selection


def run(data="data/processed", temporal="data/temporal_v2",
        source_selection="reports/v2/ranker/selection.json",
        output="reports/v2/ranker_matured_selection.json",
        evaluation_output="reports/v2/ranker_matured_internal_evaluation.json",
        artifacts="artifacts/v2/matured_internal_ranking"):
    output, evaluation_output, artifacts = map(Path, (output, evaluation_output, artifacts))
    if output.exists() or evaluation_output.exists() or artifacts.exists():
        raise FileExistsError("Mature re-selection requires fresh evidence paths; preserve prior results")
    source_selection = Path(source_selection)
    source_hash = sha256_file(source_selection)
    original = json.loads(source_selection.read_text(encoding="utf-8"))
    entries = checked_entries(original)
    before = {run: sha256_file(entry["model_path"]) for run, entry in entries.items()}
    store = Store(data)
    snapshots = TemporalStore(temporal, processed_dir=data)
    try:
        rows, audit = mature_validation_rows(store)
        with threadpool_limits(1):
            models = load_models(original, store, snapshots)
            validate_fixed_models(original, models)
            evaluation = ranking_evaluate(store, snapshots, rows, models, artifacts)
        after = {run: sha256_file(entry["model_path"]) for run, entry in entries.items()}
        if sha256_file(source_selection) != source_hash:
            raise ValueError("original selection file changed during the audit")
        selection = reselect(original, models, evaluation, audit, before, after, source_hash)
        shared = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "maturity_audit": audit, "weight_reuse_audit": selection["weight_reuse_audit"],
                  "source_selection_sha256": source_hash,
                  "official_dev_evaluated": False,
                  "input_array_sha256": {name: sha256_file(Path(data) / name) for name in
                                         ("manifest.json", "all_split.npy", "all_timestamps.npy")},
                  "source_sha256": {name: sha256_file(Path(__file__).parent / name) for name in
                                    ("mature_selection.py", "experiment.py", "ranker.py")}}
        evaluation.update(shared)
        evaluation["phase"] = "mature_internal_selection"
        dump(evaluation_output, evaluation)
        selection.update(shared)
        selection["mature_internal_evaluation_sha256"] = sha256_file(evaluation_output)
        selection["mature_internal_evaluation_path"] = str(evaluation_output).replace("\\", "/")
        dump(output, selection)
        print(json.dumps({"selected": selection["selected"], "selection": str(output),
                          "maturity_audit": audit, "all_weights_unchanged": True}), flush=True)
        return selection
    finally:
        snapshots.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--temporal", default="data/temporal_v2")
    parser.add_argument("--source-selection", default="reports/v2/ranker/selection.json")
    parser.add_argument("--output", default="reports/v2/ranker_matured_selection.json")
    parser.add_argument("--evaluation-output", default="reports/v2/ranker_matured_internal_evaluation.json")
    parser.add_argument("--artifacts", default="artifacts/v2/matured_internal_ranking")
    args = parser.parse_args()
    run(args.data, args.temporal, args.source_selection, args.output, args.evaluation_output, args.artifacts)


if __name__ == "__main__":
    main()
