"""Synthetic metadata only: fixtures deliberately contain no checkpoint files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("checkpoint_selection", Path(__file__).resolve().parents[1] / "scripts/summarize_checkpoint_selection.py")
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)


def hash_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Fixture:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.output = self.root / "public/selection.json"
        source = {}
        for name in (*s.SOURCES, "final_evaluation.py", "evaluation_scheduler.py"):
            path = self.root / "newsforge" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic frozen source: " + name, encoding="utf-8")
            source["newsforge/" + name] = digest(path)
        put(self.root, f"{s.STATE}/workspace.json", {"source_files": source, "private": "Z:\\private-user\\path"})
        put(self.root, "data/processed/manifest.json", {"synthetic": True})
        manifest = digest(self.root / "data/processed/manifest.json")
        training_sources = {name: source["newsforge/" + name] for name in s.SOURCES}
        cache = {"cache_version": 3, "sampler": "grouped_random_keys_v1", "seed": 20261010,
                 "pool_size": 16, "max_chunk_exposures": 400000,
                 "data_manifest_sha256": manifest, "features_sha256": hash_text("features reference only")}
        self.records, warmup, candidates, checkpoints, audits = {}, {}, {}, {}, []
        for variant, seed, run_id in s.RUNS:
            best = 2 if variant == "mean_random" or (variant == "mean_mixed" and seed == 29) else 1
            states = [hash_text(f"warmup {seed}" if variant != "ranker" else run_id), hash_text(run_id + " epoch2")]
            history = [{"epoch": epoch, "negative_policy": "two_hard_two_random" if epoch == 2 and variant != "mean_random" else "four_random",
                        "training_examples": 25, "state_sha256": states[epoch - 1],
                        "validation": {"impressions": 12000, "candidate_exposures": 24000,
                                       "metrics": {"ndcg10": {"mean": .6 if epoch == 1 else (.7 if best == 2 else .5)}}}}
                       for epoch in (1, 2)]
            record = {"run_id": run_id, "variant": variant, "seed": seed, "best_epoch": best,
                      "validation_ndcg10": history[best - 1]["validation"]["metrics"]["ndcg10"]["mean"],
                      "history": history, "training_configuration": dict(s.CONFIG),
                      "trained_on_all_eligible_clicks": True, "eligible_positive_examples_per_epoch": 25,
                      "data_manifest_sha256": manifest, "training_source_sha256": training_sources,
                      "training_cache_signature": cache, "private_article_title": "Synthetic secret article"}
            self.records[run_id] = record
            self.write_training(run_id)
            checkpoint_sha = hash_text("checkpoint receipt " + run_id)
            checkpoints[f"artifacts/{variant}_seed{seed}.pt"] = checkpoint_sha
            if variant == "mean_random":
                warmup[str(seed)] = {"identical": True, "sha256": [states[0], states[0]]}
                candidates[str(seed)] = {"state_sha256": states[1], "checkpoint_sha256": checkpoint_sha,
                    "evidence": {"state_matches_epoch2": True, "epoch": 2, "state_sha256": states[1], "epoch_state_sha256": states}}
            if variant == "ranker":
                filename = "evaluation_scheduler_validation_12000.json" if seed == 17 else f"evaluation_scheduler_ranker_seed{seed}_validation_12000.json"
                put(self.root, "reports/" + filename, {"run_id": run_id, "selected_epoch": best, "checkpoint_sha256": checkpoint_sha})
                audits.append({"run_id": run_id, "evidence_file": filename, "evidence_sha256": digest(self.root / "reports" / filename), "passed": True})
        put(self.root, "reports/matched_warmup_checks.json", warmup)
        put(self.root, "reports/delivery/backbone_eligibility.json", {
            "passed": True, "official_dev_evaluated": False, "data_manifest_sha256": manifest, "candidates": candidates})
        scheduling = {"entrypoint_source_sha256": source["newsforge/final_evaluation.py"],
                      "scheduler_source_sha256": source["newsforge/evaluation_scheduler.py"], "internal_validation_audits": audits}
        put(self.root, "reports/final_evaluation_preflight.json", {
            "status": "passed", "completed_checkpoint_count": 9, "official_dev_labels_evaluated_by_preflight": False,
            "data_manifest_sha256": manifest, "training_source_sha256": training_sources,
            "training_cache_signature": cache, "evaluation_scheduling": scheduling})
        put(self.root, "reports/ranking_experiment.json", {
            "runs": {name: {} for name in (*self.records, "popularity", "content")},
            "configuration": {"evaluation_only": True, "epochs": 2, "data": "Z:\\private-user\\dataset"},
            "source_sha256": source, "evaluation_scheduling": scheduling, "matched_warmup_checks": warmup})
        self.outputs = {
            "neural-train": {**checkpoints, "reports/matched_warmup_checks.json": digest(self.root / "reports/matched_warmup_checks.json")},
            "backbone-eligibility": {"reports/delivery/backbone_eligibility.json": digest(self.root / "reports/delivery/backbone_eligibility.json")},
            "v1-full-ranking": {"reports/ranking_experiment.json": digest(self.root / "reports/ranking_experiment.json")}}
        self.refresh_receipts()

    def write_training(self, run_id):
        record = self.records[run_id]
        put(self.root, f"reports/{record['variant']}_seed{record['seed']}_training.json", record)

    def refresh_receipts(self):
        for stage, parent, module in (("neural-train", None, "newsforge.train"),
                                     ("backbone-eligibility", "neural-train", "newsforge.delivery.pipeline"),
                                     ("v1-full-ranking", "backbone-eligibility", "newsforge.final_evaluation")):
            started = {"stage": stage, "status": "running", "argv": ["private-python", "-m", module],
                       "workspace_manifest_sha256": digest(self.root / s.STATE / "workspace.json"),
                       "dependencies": {parent: digest(self.root / s.STATE / parent / "success.json")} if parent else {},
                       "inputs": self.outputs[parent] if parent else {}}
            put(self.root, f"{s.STATE}/{stage}/started.json", started)
            put(self.root, f"{s.STATE}/{stage}/success.json", {
                **started, "status": "success", "returncode": 0, "outputs": self.outputs[stage]})


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def test_selected_epoch1_is_not_reported_as_hard_negative_weights(self):
        result = s.write_summary(self.f.root, self.f.output)
        self.assertEqual(result["counts"], {"completed_two_epochs": 9, "selected_epoch1": 5,
            "selected_epoch2": 4, "selected_weights_include_hard_negative_epoch": 1})
        for variant in ("mean_mixed", "ranker"):
            run = result["runs"][f"{variant}-seed17"]
            self.assertEqual(run["selected_epoch"], 1)
            self.assertFalse(run["selected_weights_include_hard_negative_epoch"])
            self.assertEqual(run["history"][1]["negative_policy"], "two_hard_two_random")
        self.assertIn("not independent checkpoint verification", " ".join(result["limitations"]))
        self.assertFalse(any(self.f.root.rglob("*.pt")))

    def test_wrong_selected_epoch_rejected(self):
        self.f.records["mean_mixed-seed17"]["best_epoch"] = 2
        self.f.write_training("mean_mixed-seed17")
        with self.assertRaisesRegex(ValueError, "selected epoch"):
            s.write_summary(self.f.root, self.f.output)
        self.assertFalse(self.f.output.exists())

    def test_validation_tie_keeps_epoch1_like_strict_trainer_comparison(self):
        record = self.f.records["ranker-seed17"]
        record["history"][1]["validation"]["metrics"]["ndcg10"]["mean"] = record["validation_ndcg10"]
        self.f.write_training("ranker-seed17")
        result = s.summarize(self.f.root)
        self.assertEqual(result["runs"]["ranker-seed17"]["selected_epoch"], 1)
        self.assertFalse(result["runs"]["ranker-seed17"]["selected_weights_include_hard_negative_epoch"])

    def test_state_disagrees_with_bound_eligibility_rejected(self):
        self.f.records["mean_random-seed17"]["history"][1]["state_sha256"] = hash_text("different state")
        self.f.write_training("mean_random-seed17")
        with self.assertRaisesRegex(ValueError, "eligibility epoch/state"):
            s.summarize(self.f.root)

    def test_metadata_output_is_allowlisted(self):
        result = s.write_summary(self.f.root, self.f.output)
        serialized = json.dumps(result)
        for secret in ("private-user", "Synthetic secret article", "private-python", "Z:\\", str(self.f.root)):
            self.assertNotIn(secret, serialized)
        self.assertEqual(set(result["runs"]["mean_random-seed17"]), {
            "variant", "seed", "completed_epochs", "selected_epoch", "selected_weights_include_hard_negative_epoch",
            "selected_state_sha256", "history", "training_json_sha256", "checkpoint_sha256_recorded_by_neural_receipt"})

    def test_output_is_never_overwritten(self):
        self.f.output.parent.mkdir(parents=True)
        self.f.output.write_text("original evidence", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            s.write_summary(self.f.root / "does-not-exist", self.f.output)
        self.assertEqual(self.f.output.read_text(encoding="utf-8"), "original evidence")

    def test_final_report_must_match_success_receipt(self):
        put(self.f.root, "reports/ranking_experiment.json", {"tampered": True})
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            s.summarize(self.f.root)

    def test_running_evaluation_cannot_generate_summary(self):
        (self.f.root / s.STATE / "v1-full-ranking/success.json").unlink()
        with self.assertRaises(FileNotFoundError):
            s.write_summary(self.f.root, self.f.output)
        self.assertFalse(self.f.output.exists())

    def test_nonfinite_validation_rejected(self):
        self.f.records["ranker-seed17"]["history"][1]["validation"]["metrics"]["ndcg10"]["mean"] = float("nan")
        self.f.write_training("ranker-seed17")
        with self.assertRaisesRegex(ValueError, "invalid internal validation"):
            s.summarize(self.f.root)


if __name__ == "__main__":
    unittest.main()
