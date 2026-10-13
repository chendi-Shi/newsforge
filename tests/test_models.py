"""Numerical contracts for retrieval vectors and candidate-wise ranking."""

import unittest

import torch

from newsforge.models import AttentionRanker, TwoTower


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)
        self.history = torch.randn(3, 5, 12)
        self.history[0, 3:] = 0
        self.history[1] = 0
        self.mask = self.history.ne(0).any(-1)
        self.candidates = torch.randn(3, 4, 12)
        self.priors = torch.randn(3, 4, 2)

    def test_tower_shapes_norms_empty_and_padding(self):
        for pooling in ("mean", "attention"):
            with self.subTest(pooling=pooling):
                model = TwoTower(12, embedding_dim=8, hidden=16, user_pooling=pooling)
                users = model.encode_user(self.history, self.mask)
                self.assertEqual(users.shape, (3, 8))
                torch.testing.assert_close(users[[0, 2]].norm(dim=-1), torch.ones(2))
                torch.testing.assert_close(users[1], torch.zeros(8), atol=0, rtol=0)
                padded = torch.cat((self.history, torch.zeros(3, 7, 12)), dim=1)
                padded_mask = torch.cat((self.mask, torch.zeros(3, 7, dtype=torch.bool)), dim=1)
                torch.testing.assert_close(model.encode_user(padded, padded_mask), users)
                # Masked content is unobservable even when its tensor is not zero.
                poisoned = self.history.clone()
                poisoned[~self.mask] = 100
                torch.testing.assert_close(model.encode_user(poisoned, self.mask), users)
                news = model.encode_news(self.candidates)
                self.assertEqual(news.shape, (3, 4, 8))
                torch.testing.assert_close(news.norm(dim=-1), torch.ones(3, 4))
                torch.testing.assert_close(model.encode_news(torch.zeros(2, 12)), torch.zeros(2, 8))
                torch.testing.assert_close(model.encode_user(torch.zeros(3, 0, 12)), torch.zeros(3, 8))

    def test_tower_batch_loop_and_inferred_mask(self):
        for pooling in ("mean", "attention"):
            model = TwoTower(12, 8, 16, pooling)
            scores = model(self.history, self.candidates, self.mask)
            separate = torch.stack([model(self.history, self.candidates[:, c], self.mask)
                                    for c in range(self.candidates.shape[1])], dim=1)
            torch.testing.assert_close(scores, separate)
            torch.testing.assert_close(scores, model(self.history, self.candidates))
            torch.testing.assert_close(scores[0], model(self.history[:1], self.candidates[:1], self.mask[:1])[0])
            torch.testing.assert_close(scores[1], torch.zeros(4))

    def test_ranker_batch_loop_and_padding_invariance(self):
        model = AttentionRanker(12, hidden=16, embedding_dim=8)
        expected = model.score_candidates(self.history, self.mask, self.candidates, self.priors)
        separate = torch.stack([model(self.history, self.candidates[:, c], self.mask, self.priors[:, c])
                                for c in range(self.candidates.shape[1])], dim=1)
        torch.testing.assert_close(expected, separate, atol=1e-6, rtol=1e-5)
        padded = torch.cat((self.history, torch.zeros(3, 7, 12)), dim=1)
        padded_mask = torch.cat((self.mask, torch.zeros(3, 7, dtype=torch.bool)), dim=1)
        torch.testing.assert_close(model.score_candidates(padded, padded_mask, self.candidates, self.priors), expected)
        poisoned = self.history.clone()
        poisoned[~self.mask] = 100
        torch.testing.assert_close(model.score_candidates(poisoned, self.mask, self.candidates, self.priors), expected)
        torch.testing.assert_close(model.score_candidates(self.history[:1], self.mask[:1], self.candidates[:1], self.priors[:1])[0], expected[0])
        torch.testing.assert_close(model.score_candidates(self.history, self.mask, self.candidates),
                                   model.score_candidates(self.history, self.mask, self.candidates, torch.zeros_like(self.priors)))

    def test_empty_history_ranker_is_finite(self):
        model = AttentionRanker(12, 16, 8)
        scores = model(torch.zeros(3, 0, 12), self.candidates)
        self.assertTrue(torch.isfinite(scores).all())
        torch.testing.assert_close(scores, model(torch.zeros(3, 5, 12), self.candidates))
        scores.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_gradient_flow_to_every_trainable_component(self):
        models = [TwoTower(12, 8, 16, "mean"), TwoTower(12, 8, 16, "attention"), AttentionRanker(12, 16, 8)]
        for model in models:
            with self.subTest(model=type(model).__name__, pooling=getattr(model, "user_pooling", None)):
                history = self.history.clone().requires_grad_()
                candidates = self.candidates.clone().requires_grad_()
                scores = model(history, candidates, self.mask)
                (scores * torch.randn_like(scores)).sum().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                self.assertTrue(torch.isfinite(history.grad).all())
                self.assertGreater(candidates.grad.abs().sum().item(), 0)
                torch.testing.assert_close(history.grad[~self.mask], torch.zeros_like(history.grad[~self.mask]), atol=0, rtol=0)

    def test_serialization_and_config_roundtrip(self):
        for cls, config in ((TwoTower, {"feature_dim": 12, "embedding_dim": 8, "hidden": 16, "user_pooling": "attention"}),
                            (AttentionRanker, {"feature_dim": 12, "embedding_dim": 8, "hidden": 16})):
            original = cls(**config)
            restored = cls(**config)
            restored.load_state_dict(original.state_dict())
            torch.testing.assert_close(original(self.history, self.candidates, self.mask),
                                       restored(self.history, self.candidates, self.mask), atol=0, rtol=0)

    def test_shape_validation(self):
        tower = TwoTower(12)
        with self.assertRaises(ValueError):
            tower.encode_user(self.history, self.mask.float())
        with self.assertRaises(ValueError):
            tower.encode_news(torch.randn(4, 13))
        with self.assertRaises(ValueError):
            AttentionRanker(12).score_candidates(self.history, self.mask, self.candidates, torch.zeros(3, 4))


if __name__ == "__main__":
    unittest.main()
