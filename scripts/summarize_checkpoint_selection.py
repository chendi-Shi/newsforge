"""Supplement existing v1 evidence with selected epochs; never open model files.

This verifies small retained reports and a three-stage receipt segment, not the
full reproduction or the bytes inside checkpoints. Training JSON and preflight
hashes are bound now; they were not declared neural-train output hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


SEEDS = (17, 29, 43)
VARIANTS = ("mean_random", "mean_mixed", "ranker")
RUNS = tuple((variant, seed, f"{variant}-seed{seed}") for variant in VARIANTS for seed in SEEDS)
STATE = "artifacts/delivery/pipeline"
CONFIG = {"epochs": 2, "batch_size": 1024, "threads": 1, "learning_rate": .001,
          "validation_impressions": 12000, "negative_pool_size": 16}
SOURCES = ("models.py", "train.py", "store.py", "evaluate.py", "metrics.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid SHA-256")
    return value


def score(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            "invalid internal validation nDCG10")
    return value


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes, self.stats = {}, {}

    def read(self, name, expected=None, *, parse=True):
        # Every filename is constructed from this script's fixed allowlists.
        path = self.root / name
        require(path.resolve().is_relative_to(self.root), "evidence escapes workspace")
        before = path.stat()
        data = path.read_bytes()
        identity = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        actual = hashlib.sha256(data).hexdigest()
        require(name not in self.hashes or self.hashes[name] == actual, "evidence changed")
        require(expected is None or actual == sha(expected), "evidence SHA-256 mismatch")
        self.hashes[name], self.stats[name] = actual, identity
        return json.loads(data) if parse else actual

    def finish(self):
        for name, expected in self.stats.items():
            current = (self.root / name).stat()
            require((current.st_size, current.st_mtime_ns, current.st_ctime_ns) == expected,
                    "evidence changed during summary")


def summarize(workspace):
    e = Evidence(workspace)
    workspace_name = f"{STATE}/workspace.json"
    workspace_record = e.read(workspace_name)
    receipts = {}
    for stage, parent, module in (("neural-train", None, "newsforge.train"),
                                  ("backbone-eligibility", "neural-train", "newsforge.delivery.pipeline"),
                                  ("v1-full-ranking", "backbone-eligibility", "newsforge.final_evaluation")):
        directory = f"{STATE}/{stage}"
        require(not (e.root / directory / "failed.json").exists(), "failed stage is present")
        started, completed = e.read(f"{directory}/started.json"), e.read(f"{directory}/success.json")
        require(started.get("status") == "running" and completed.get("status") == "success"
                and completed.get("returncode") == 0 and completed.get("stage") == stage,
                "stage is not completed successfully")
        require(all(completed.get(key) == value for key, value in started.items() if key != "status"),
                "stage start/success identities disagree")
        require(completed.get("workspace_manifest_sha256") == e.hashes[workspace_name]
                and completed.get("argv", [])[1:3] == ["-m", module], "stage identity disagrees")
        if parent:
            require(completed.get("dependencies") == {parent: e.hashes[f"{STATE}/{parent}/success.json"]}
                    and completed.get("inputs") == receipts[parent]["outputs"], "stage receipt chain disagrees")
        receipts[stage] = completed
    neural = receipts["neural-train"]
    checkpoints = {f"artifacts/{variant}_seed{seed}.pt" for variant, seed, _ in RUNS}
    require(set(neural["outputs"]) == checkpoints | {"reports/matched_warmup_checks.json"},
            "unexpected neural output inventory")
    for value in neural["outputs"].values():
        sha(value)
    eligibility_name = "reports/delivery/backbone_eligibility.json"
    require(set(receipts["backbone-eligibility"]["outputs"]) == {eligibility_name}, "unexpected eligibility output")
    eligibility = e.read(eligibility_name, receipts["backbone-eligibility"]["outputs"][eligibility_name])
    ranking_name = "reports/ranking_experiment.json"
    require(set(receipts["v1-full-ranking"]["outputs"]) == {ranking_name}, "unexpected ranking output")
    ranking = e.read(ranking_name, receipts["v1-full-ranking"]["outputs"][ranking_name])
    preflight = e.read("reports/final_evaluation_preflight.json")
    warmup = e.read("reports/matched_warmup_checks.json", neural["outputs"]["reports/matched_warmup_checks.json"])
    require(preflight.get("status") == "passed" and preflight.get("completed_checkpoint_count") == 9
            and preflight.get("official_dev_labels_evaluated_by_preflight") is False, "invalid preflight")
    require(eligibility.get("passed") is True and eligibility.get("official_dev_evaluated") is False,
            "invalid eligibility report")
    manifest_sha = sha(preflight["data_manifest_sha256"])
    require(eligibility["data_manifest_sha256"] == manifest_sha, "manifest references disagree")
    e.read("data/processed/manifest.json", manifest_sha, parse=False)
    source_hashes = preflight["training_source_sha256"]
    require(set(source_hashes) == set(SOURCES), "unexpected training source inventory")
    scheduling = preflight["evaluation_scheduling"]
    require(all(ranking["evaluation_scheduling"].get(key) == value for key, value in scheduling.items()),
            "preflight/final scheduling differs")
    required_sources = {f"newsforge/{name}": sha(source_hashes[name]) for name in SOURCES}
    required_sources.update({"newsforge/final_evaluation.py": sha(scheduling["entrypoint_source_sha256"]),
                             "newsforge/evaluation_scheduler.py": sha(scheduling["scheduler_source_sha256"])})
    for name, expected in required_sources.items():
        require(workspace_record["source_files"].get(name) == expected
                and ranking["source_sha256"].get(name) == expected, "source references disagree")
        e.read(name, expected, parse=False)
    cache = preflight["training_cache_signature"]
    cache_public = {"cache_version": 3, "sampler": "grouped_random_keys_v1", "seed": 20261010,
                    "pool_size": 16, "max_chunk_exposures": 400000, "data_manifest_sha256": manifest_sha,
                    "features_sha256": sha(cache["features_sha256"])}
    require(cache == cache_public, "unexpected training cache signature")
    expected_runs = {run for _, _, run in RUNS}
    require(set(ranking["runs"]) == expected_runs | {"popularity", "content"}
            and ranking["configuration"].get("evaluation_only") is True
            and ranking["configuration"].get("epochs") == 2, "incomplete final ranking report")
    runs, eligible_count, exposure_count = {}, None, None
    for variant, seed, run_id in RUNS:
        name = f"reports/{variant}_seed{seed}_training.json"
        record = e.read(name)
        require((record.get("run_id"), record.get("variant"), record.get("seed")) == (run_id, variant, seed),
                "training run identity disagrees")
        require(record.get("training_configuration") == CONFIG and record.get("trained_on_all_eligible_clicks") is True
                and record.get("data_manifest_sha256") == manifest_sha
                and record.get("training_source_sha256") == source_hashes
                and record.get("training_cache_signature") == cache, "training provenance disagrees")
        count = record["eligible_positive_examples_per_epoch"]
        require(type(count) is int and count > 0 and (eligible_count is None or eligible_count == count),
                "training example count disagrees")
        eligible_count = count
        history = record["history"]
        policies = ["four_random", "four_random" if variant == "mean_random" else "two_hard_two_random"]
        require(len(history) == 2 and [row["epoch"] for row in history] == [1, 2]
                and [row["negative_policy"] for row in history] == policies, "epoch/policy history disagrees")
        compact = []
        for row in history:
            validation = row["validation"]
            exposures = validation["candidate_exposures"]
            require(row["training_examples"] == count and validation["impressions"] == 12000
                    and type(exposures) is int and exposures >= 12000
                    and (exposure_count is None or exposure_count == exposures), "epoch population disagrees")
            exposure_count = exposures
            compact.append({"epoch": row["epoch"], "negative_policy": row["negative_policy"],
                            "training_examples": count, "internal_validation_ndcg10": score(validation["metrics"]["ndcg10"]["mean"]),
                            "state_sha256": sha(row["state_sha256"])})
        selected = max(range(2), key=lambda index: compact[index]["internal_validation_ndcg10"])
        require(type(record["best_epoch"]) is int and record["best_epoch"] == selected + 1
                and score(record["validation_ndcg10"]) == compact[selected]["internal_validation_ndcg10"],
                "selected epoch/validation score disagrees")
        runs[run_id] = {"variant": variant, "seed": seed, "completed_epochs": 2, "selected_epoch": selected + 1,
                        "selected_weights_include_hard_negative_epoch": variant != "mean_random" and selected == 1,
                        "selected_state_sha256": compact[selected]["state_sha256"], "history": compact,
                        "training_json_sha256": e.hashes[name],
                        "checkpoint_sha256_recorded_by_neural_receipt": neural["outputs"][f"artifacts/{variant}_seed{seed}.pt"]}
    for seed in SEEDS:
        random, mixed = runs[f"mean_random-seed{seed}"], runs[f"mean_mixed-seed{seed}"]
        states = [random["history"][0]["state_sha256"], mixed["history"][0]["state_sha256"]]
        require(states[0] == states[1] and warmup[str(seed)] == {"identical": True, "sha256": states}
                and ranking["matched_warmup_checks"][str(seed)] == warmup[str(seed)], "warmup state disagrees")
        candidate = eligibility["candidates"][str(seed)]
        proof = candidate["evidence"]
        require(random["selected_epoch"] == 2 and candidate["state_sha256"] == random["selected_state_sha256"]
                and candidate["checkpoint_sha256"] == random["checkpoint_sha256_recorded_by_neural_receipt"]
                and proof["state_matches_epoch2"] is True and proof["epoch"] == 2
                and proof["state_sha256"] == random["selected_state_sha256"]
                and proof["epoch_state_sha256"] == [row["state_sha256"] for row in random["history"]],
                "eligibility epoch/state disagrees")
    references = scheduling["internal_validation_audits"]
    require(len(references) == 3 and {item["run_id"] for item in references} == {f"ranker-seed{s}" for s in SEEDS},
            "missing ranker scheduling audits")
    for seed in SEEDS:
        run = runs[f"ranker-seed{seed}"]
        filename = "evaluation_scheduler_validation_12000.json" if seed == 17 else f"evaluation_scheduler_ranker_seed{seed}_validation_12000.json"
        reference = next(item for item in references if item["run_id"] == f"ranker-seed{seed}")
        require(reference["evidence_file"] == filename and reference["passed"] is True, "ranker audit reference disagrees")
        audit = e.read(f"reports/{filename}", reference["evidence_sha256"])
        require(audit["run_id"] == f"ranker-seed{seed}" and audit["selected_epoch"] == run["selected_epoch"]
                and audit["checkpoint_sha256"] == run["checkpoint_sha256_recorded_by_neural_receipt"],
                "ranker selected epoch/checkpoint reference disagrees")
    e.finish()
    return {"schema_version": 1, "status": "passed", "scope": "supplemental checkpoint-selection metadata reconciliation",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": e.hashes, "data_manifest_sha256": manifest_sha,
            "training_source_sha256": {name: sha(source_hashes[name]) for name in SOURCES},
            "training_cache_signature": cache_public, "training_configuration": CONFIG,
            "internal_validation_impressions": 12000, "internal_validation_candidate_exposures": exposure_count,
            "runs": runs, "counts": {"completed_two_epochs": 9,
                "selected_epoch1": sum(run["selected_epoch"] == 1 for run in runs.values()),
                "selected_epoch2": sum(run["selected_epoch"] == 2 for run in runs.values()),
                "selected_weights_include_hard_negative_epoch": sum(run["selected_weights_include_hard_negative_epoch"] for run in runs.values())},
            "limitations": [
                "Training JSON and preflight hashes are bound at this supplemental check, not retrospectively added to neural-train's declared output hashes.",
                "No checkpoint bytes are opened or hashed and no predictions are recomputed. Selected state hashes are metadata assertions cross-checked where existing warmup, eligibility or scheduling evidence permits; this is not independent checkpoint verification.",
                "The frozen final-evaluation implementation verifies training JSON against checkpoint metadata and selected-state bytes before dev evaluation. This summary reconciles retained evidence and source hashes; hashes and a passed preflight alone do not independently prove that historical execution.",
                "Only the neural-train to backbone-eligibility to v1-full-ranking receipt segment is reconciled. Full pipeline, raw data, cache arrays and stage logs are not re-audited here; cache hashes are retained references.",
                "Dev comparisons describe training strategies followed by internal-validation checkpoint selection, not a fixed-epoch causal effect of hard negatives. Ranker comparisons additionally change architecture."]}


def write_summary(workspace, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("selection evidence output already exists")
    result = summarize(workspace)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_summary(args.workspace, args.output)
    print(json.dumps({"status": result["status"], "scope": result["scope"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()
