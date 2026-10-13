"""Matured backbone selection boundaries and provenance guards, without dev data."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from newsforge.v2.select_backbone import (
    MAX_VALID_TIME, MIN_VALID_TIME, checkpoint_evidence, current_backbone_audit,
    select_matured_rows, select_winner,
)


class MaturedRowsTests(unittest.TestCase):
    def test_half_open_boundaries_and_internal_split_only(self):
        store = SimpleNamespace(
            split=np.array([1, 1, 1, 1, 2, 0]),
            timestamps=np.array([MIN_VALID_TIME - 1, MIN_VALID_TIME, MAX_VALID_TIME - 1,
                                 MAX_VALID_TIME, MIN_VALID_TIME, MIN_VALID_TIME]),
        )
        np.testing.assert_array_equal(select_matured_rows(store, 2), [1, 2])

    def test_fixed_seed_sample_is_sorted_and_exact(self):
        store = SimpleNamespace(split=np.ones(100, dtype=np.int8),
                                timestamps=np.full(100, MIN_VALID_TIME))
        expected = np.sort(np.random.default_rng(2026).choice(np.arange(100), 12, replace=False))
        actual = select_matured_rows(store, 12)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype("int64"))
        np.testing.assert_array_equal(select_matured_rows(store, 12), actual)

    def test_does_not_silently_reduce_formal_population(self):
        store = SimpleNamespace(split=np.array([1]), timestamps=np.array([MIN_VALID_TIME]))
        with self.assertRaisesRegex(ValueError, "not enough"):
            select_matured_rows(store, 2)
        for value in [0, -1, True, 2.5]:
            with self.subTest(limit=value), self.assertRaises(ValueError):
                select_matured_rows(store, value)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(seed=17, state_sha256="state-two", feature_dim=3,
                             manifest_sha256="manifest", source_sha256={"train.py": "source"},
                             cache_signature={"cache_version": 3},
                             train_cutoff_timestamp=MIN_VALID_TIME - 3601, eligible_positives=20)
        self.checkpoint = {
            "seed": 17, "variant": "mean_random", "kind": "tower",
            "config": {"feature_dim": 3, "hidden": 128, "embedding_dim": 64, "user_pooling": "mean"},
            "training_configuration": {"epochs": 2}, "best_epoch": 2,
            "trained_on_all_eligible_clicks": True, "eligible_positive_examples_per_epoch": 20,
            "history": [{"epoch": 1, "training_examples": 20, "negative_policy": "four_random", "state_sha256": "state-one"},
                        {"epoch": 2, "training_examples": 20, "negative_policy": "four_random", "state_sha256": "state-two"}],
            "data_manifest_sha256": "manifest", "training_source_sha256": {"train.py": "source"},
            "training_cache_signature": {"cache_version": 3},
            "train_cutoff_timestamp": MIN_VALID_TIME - 3601,
            # This obsolete score must have no influence on eligibility/selection.
            "validation_ndcg10": float("nan"),
        }

    def test_accepts_complete_fixed_epoch_state_without_old_validation_score(self):
        result = checkpoint_evidence(self.checkpoint, **self.expected)
        self.assertTrue(result["state_matches_epoch2"])
        self.assertEqual(result["epoch"], 2)
        self.assertNotIn("validation_ndcg10", result)

    def test_rejects_first_epoch_checkpoint_and_wrong_loaded_state(self):
        for key, value in [("best_epoch", 1), ("trained_on_all_eligible_clicks", False),
                           ("eligible_positive_examples_per_epoch", 19), ("variant", "mean_mixed")]:
            changed = copy.deepcopy(self.checkpoint); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint_evidence(changed, **self.expected)
        with self.assertRaisesRegex(ValueError, "state digest"):
            checkpoint_evidence(self.checkpoint, **{**self.expected, "state_sha256": "state-one"})

    def test_rejects_partial_or_changed_training_history(self):
        changes = [lambda c: c["history"].pop(),
                   lambda c: c["history"][1].update(training_examples=19),
                   lambda c: c["history"][1].update(negative_policy="two_hard_two_random"),
                   lambda c: c["training_configuration"].update(epochs=3)]
        for number, change in enumerate(changes):
            changed = copy.deepcopy(self.checkpoint); change(changed)
            with self.subTest(change=number), self.assertRaises(ValueError):
                checkpoint_evidence(changed, **self.expected)

    def test_rejects_changed_data_source_cache_or_unmatured_cutoff(self):
        for key, value in [("data_manifest_sha256", "changed"),
                           ("training_source_sha256", {"train.py": "changed"}),
                           ("training_cache_signature", {"cache_version": 4}),
                           ("train_cutoff_timestamp", MIN_VALID_TIME - 3600)]:
            changed = copy.deepcopy(self.checkpoint); changed[key] = value
            expected = dict(self.expected)
            if key == "train_cutoff_timestamp": expected[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint_evidence(changed, **expected)


class WinnerTests(unittest.TestCase):
    @staticmethod
    def candidates(scores):
        return [{"seed": seed, "metrics": {"ndcg10": {"mean": score}}}
                for seed, score in zip([43, 17, 29], scores)]

    def test_highest_full_candidate_ndcg_wins(self):
        self.assertEqual(select_winner(self.candidates([.2, .3, .4]))["seed"], 29)

    def test_exact_tie_prefers_lower_seed_not_input_order(self):
        self.assertEqual(select_winner(self.candidates([.4, .4, .3]))["seed"], 17)

    def test_requires_three_finite_complete_candidates(self):
        for candidates in [self.candidates([.4, .3]), self.candidates([.4, .3, float("nan")])]:
            with self.assertRaises(ValueError):
                select_winner(candidates)
        candidates = self.candidates([.4, .3, .2]); candidates[0]["seed"] = 17
        with self.assertRaises(ValueError):
            select_winner(candidates)

    def test_current_backbone_matches_tensor_state_not_serialized_file_hash(self):
        checkpoint = {"seed": 43, "variant": "mean_random", "best_epoch": 2}
        selected = {"state_sha256": "same-tensors", "checkpoint_sha256": "original-save"}
        result = current_backbone_audit(
            checkpoint, checkpoint_path="artifacts/retriever.pt",
            checkpoint_sha256="a-different-serialization", state_sha256="same-tensors", selected=selected,
        )
        self.assertTrue(result["matches_current_retriever_state"])
        self.assertTrue(result["current_retriever_audit"]["selected_state_matches_current"])
        self.assertEqual(result["current_backbone"]["checkpoint_sha256"], "a-different-serialization")
        self.assertEqual(result["current_retriever_audit"]["seed"], 43)
        result = current_backbone_audit(
            checkpoint, checkpoint_path="artifacts/retriever.pt",
            checkpoint_sha256="a-different-serialization", state_sha256="different-tensors", selected=selected,
        )
        self.assertFalse(result["matches_current_retriever_state"])
        self.assertFalse(result["current_retriever_audit"]["selected_state_matches_current"])


if __name__ == "__main__":
    unittest.main()
