"""Synthetic selection-boundary and immutable-weight checks; no real dev reads."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from newsforge.v2.experiment import MIN_VALID_TIME, SEEDS, VARIANTS
from newsforge.v2.mature_selection import (checked_entries, mature_validation_rows,
                                         reselect, run, validate_fixed_models)
from newsforge.v2.ranker import MODEL_CONFIG
from tests.support import workspace_temporary_directory


def fixed_models():
    sampling = {"positive_pairs": 100000, "binary_examples": 500000}
    entries, models, metrics, hashes = [], {}, {}, {}
    for variant in VARIANTS:
        for seed in SEEDS:
            name = f"{variant}-seed{seed}"
            entries.append({"run_id": name, "variant": variant, "seed": seed,
                            "model_path": f"artifacts/{name}.joblib", "model_sha256": name})
            models[name] = SimpleNamespace(
                model_version=name, variant=variant,
                metadata={"seed": seed, "config": {**MODEL_CONFIG, "random_state": seed}, "sampling": sampling},
                model=SimpleNamespace(n_iter_=80, early_stopping=False, fit=Mock(side_effect=AssertionError("no fitting"))),
                feature_builder=SimpleNamespace(bindings={"shared": "fixed-inputs"}))
            metrics[name] = {"ndcg10": {"mean": .5, "eligible_impressions": 6000}}
            hashes[name] = name
    audit = {"validation_min_inclusive": MIN_VALID_TIME,
             "validation_max_exclusive": 1573772400,
             "first_dev_timestamp": 1573776000, "first_dev_snapshot_cutoff": 1573772400,
             "sampled_timestamp_min": MIN_VALID_TIME,
             "sampled_timestamp_max": 1573772399,
             "sampled_rows": 6000, "row_indices_sha256": "fixed-rows",
             "all_rows_split1": True, "all_selection_feedback_strictly_before_first_dev_cutoff": True}
    evaluation = {"impressions": 6000, "candidates": 210000,
                  "row_indices_sha256": "fixed-rows", "metrics": metrics}
    return {"models": entries, "sampling": sampling}, models, evaluation, audit, hashes


class MatureSelectionTests(unittest.TestCase):
    def test_sampling_excludes_exact_cutoff_and_early_rows(self):
        cutoff = 1573772400
        timestamps = np.array([MIN_VALID_TIME - 1, MIN_VALID_TIME, MIN_VALID_TIME + 1,
                               cutoff - 1, cutoff, cutoff + 1, cutoff - 2], dtype=np.int64)
        store = SimpleNamespace(split=np.array([1, 1, 1, 1, 1, 1, 2]), timestamps=timestamps,
                                manifest={"splits": {"2": {"min_timestamp": 1573776000}}})
        rows, audit = mature_validation_rows(store, count=3)
        np.testing.assert_array_equal(rows, [1, 2, 3])
        again, repeated = mature_validation_rows(store, count=3)
        np.testing.assert_array_equal(again, rows)
        self.assertEqual(repeated, audit)
        self.assertEqual(audit["sampled_timestamp_max"], cutoff - 1)
        self.assertEqual(audit["first_dev_snapshot_cutoff"], cutoff)
        self.assertTrue(audit["all_rows_split1"])
        with self.assertRaisesRegex(ValueError, "budget"):
            mature_validation_rows(store, count=4)

    def test_reselection_keeps_nine_identities_and_never_fits(self):
        original, models, evaluation, audit, hashes = fixed_models()
        evaluation["metrics"]["full-seed29"]["ndcg10"]["mean"] = .6
        selected = reselect(original, models, evaluation, audit, hashes, hashes, "original-selection-hash")
        self.assertEqual(selected["selected"]["run_id"], "full-seed29")
        self.assertEqual(selected["best_by_variant"]["content_only"]["seed"], 17)
        self.assertEqual(len(selected["models"]), 9)
        self.assertEqual(selected["maturity_audit"], audit)
        self.assertFalse(selected["weight_reuse_audit"]["fit_invoked"])
        self.assertTrue(selected["weight_reuse_audit"]["all_weights_unchanged"])
        self.assertEqual(selected["weight_reuse_audit"]["model_sha256_after"], hashes)
        self.assertEqual(selected["source_selection_sha256"], "original-selection-hash")
        for model in models.values():
            model.model.fit.assert_not_called()

    def test_rejects_early_stopping_changed_rounds_or_training_budget(self):
        for field, replacement in (("early_stopping", True), ("n_iter_", 79)):
            original, models, _, _, _ = fixed_models()
            setattr(models["full-seed17"].model, field, replacement)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "fixed-budget"):
                validate_fixed_models(original, models)
        original, models, _, _, _ = fixed_models()
        original["sampling"]["positive_pairs"] = 99999
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_fixed_models(original, models)

    def test_rejects_mutated_weight_bytes_and_mature_boundary(self):
        original, models, evaluation, audit, hashes = fixed_models()
        for which in ("before", "after"):
            changed = {**hashes, "full-seed17": "changed-bytes"}
            with self.subTest(which=which), self.assertRaisesRegex(ValueError, "bytes changed"):
                reselect(original, models, evaluation, audit,
                         changed if which == "before" else hashes,
                         changed if which == "after" else hashes, "source")
        for field, value in (("sampled_timestamp_max", audit["validation_max_exclusive"]),
                             ("all_rows_split1", False),
                             ("all_selection_feedback_strictly_before_first_dev_cutoff", False),
                             ("first_dev_snapshot_cutoff", audit["first_dev_snapshot_cutoff"] + 3600)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "boundary mismatch"):
                reselect(original, models, evaluation, {**audit, field: value}, hashes, hashes, "source")
        changed = {**evaluation, "row_indices_sha256": "different population"}
        with self.assertRaisesRegex(ValueError, "population"):
            reselect(original, models, changed, audit, hashes, hashes, "source")

    def test_rejects_missing_duplicate_models_and_preserves_existing_evidence(self):
        original, _, _, _, _ = fixed_models()
        changed = copy.deepcopy(original)
        changed["models"].pop()
        with self.assertRaisesRegex(ValueError, "nine"):
            checked_entries(changed)
        changed["models"].append(changed["models"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checked_entries(changed)
        with workspace_temporary_directory(prefix="mature-selection-") as directory:
            path = Path(directory) / "existing.json"
            path.write_text("original evidence", encoding="utf-8")
            with patch("newsforge.v2.mature_selection.Store") as store:
                with self.assertRaises(FileExistsError):
                    run(output=path)
                store.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), "original evidence")


if __name__ == "__main__":
    unittest.main()
