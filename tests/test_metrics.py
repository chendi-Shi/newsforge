import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from newsforge.metrics import impression_metrics, user_cluster_interval


class MetricsTest(unittest.TestCase):
    def test_multiple_positive_mrr_is_not_first_hit(self):
        result = impression_metrics([1, 0, 1], [3., 2., 1.])
        self.assertAlmostEqual(result[1], (1 + 1/3) / 2)

    def test_auc_with_ties_matches_sklearn(self):
        y, p = [0, 1, 1, 0], [.2, .2, .5, .5]
        self.assertEqual(impression_metrics(y, p)[0], roc_auc_score(y, p))

    def test_perfect_and_single_class(self):
        np.testing.assert_allclose(impression_metrics([1, 0, 0], [3, 2, 1]), 1)
        self.assertTrue(np.isnan(impression_metrics([1, 1], [1, 0])[0]))

    def test_identical_models_bootstrap_zero(self):
        metrics = np.arange(16, dtype=float).reshape(4, 4)
        result = user_cluster_interval(metrics, metrics, np.array([1, 1, 2, 3]), repeats=20)
        self.assertEqual(result["auc"]["ci95"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
