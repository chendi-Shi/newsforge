"""GBDT scheduling keeps ragged populations, NaNs, groups and locked test semantics."""

import contextlib
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from newsforge.gbdt import evaluate_gbdt, train_gbdt
from newsforge.gbdt_evaluation_scheduler import evaluate_scheduled_gbdt
from newsforge.gbdt_final_evaluation import validate_internal_audit
from tests.support import workspace_temporary_directory
from tests.test_gbdt import TinyStore


class RaggedStore(TinyStore):
    def __init__(self, path):
        super().__init__(path)
        groups, labels = [], []
        sizes = [4, 1, 3, 2, 4, 2, 3, 1]
        for row in range(len(self.split)):
            size = sizes[row - 40] if 40 <= row < 48 else 4
            groups.append(np.arange(1, size + 1))
            label = np.zeros(size, dtype=np.uint8)
            if row != 47:
                label[0 if size == 1 else row % 2] = 1
            labels.append(label)
        self.candidates = np.concatenate(groups).astype(np.int32)
        self.labels = np.concatenate(labels)
        self.offsets = np.r_[0, np.cumsum([len(group) for group in groups])]

    def padded_candidates(self, rows):
        lengths = self.offsets[rows + 1] - self.offsets[rows]
        ids = np.zeros((len(rows), int(lengths.max())), dtype=np.int32)
        for position, row in enumerate(rows):
            ids[position, :lengths[position]] = self.candidates[self.offsets[row]:self.offsets[row + 1]]
        return ids, np.arange(ids.shape[1])[None, :] < lengths[:, None]


class GBDTSchedulerTests(unittest.TestCase):
    def test_ragged_candidate_metrics_nan_masks_groups_and_original_row_order(self):
        with workspace_temporary_directory('gbdt-scheduler-') as directory:
            root = Path(directory)
            store = RaggedStore(root)
            # The tiny fit supplies a scheduler fixture; ambient RAM is not
            # under test. Production preflight rejection remains tested apart.
            with patch("newsforge.gbdt.psutil.virtual_memory") as memory, \
                    contextlib.redirect_stdout(io.StringIO()):
                memory.return_value.available = 2 * 2**30
                artifact, _ = train_gbdt(store, artifacts=root / 'artifacts', reports=root / 'reports',
                    max_positives=30, threads=1, batch_size=8, validation_impressions=5,
                    feature_mode='rich', run_name='gbdt_rich')
            rows = np.array([46, 40, 47, 42, 41, 45, 43, 44], dtype=np.int64)
            with threadpool_limits(1):
                first, values = evaluate_gbdt(store, artifact, rows, batch_size=2)
                second, reordered = evaluate_scheduled_gbdt(store, artifact, rows, batch_size=2,
                                                            output=root / 'metrics.npy')
                with self.assertRaisesRegex(ValueError, 'evaluate-test'):
                    evaluate_scheduled_gbdt(store, artifact, np.array([48]), batch_size=2)
            np.testing.assert_allclose(values, reordered, atol=1e-12, rtol=0, equal_nan=True)
            np.testing.assert_array_equal(np.isnan(values), np.isnan(reordered))
            np.testing.assert_array_equal(np.load(root / 'metrics.npy'), reordered)
            self.assertEqual(first['candidate_exposures'], second['candidate_exposures'])
            self.assertTrue(second['evaluation_scheduling']['metric_rows_restored_to_original_order'])
            for group, record in first['groups'].items():
                self.assertEqual(record['impressions'], second['groups'][group]['impressions'])
                if record['metrics'] is None:
                    self.assertIsNone(second['groups'][group]['metrics'])
                    continue
                for metric, result in record['metrics'].items():
                    compared = second['groups'][group]['metrics'][metric]
                    self.assertEqual(result['eligible_impressions'], compared['eligible_impressions'])
                    if result['mean'] is None:
                        self.assertIsNone(compared['mean'])
                    else:
                        self.assertAlmostEqual(result['mean'], compared['mean'], places=12)

    def test_audit_requires_bound_full_population_and_metric_equivalence(self):
        expected = {'checkpoint_sha256': 'model', 'scheduler_source_sha256': 'scheduler',
                    'data_manifest_sha256': 'data', 'selected_row_indices_sha256': 'rows'}
        record = {**expected, 'rows': 12000, 'fixed_validation_rows': 12000,
            'metric_equivalence_passed': True, 'per_impression_metric_equivalence_atol': 1e-12,
            'environment': {'threads': 1},
            'metric_differences': {metric: {'same_nan_mask': True, 'pairs_differing_more_than_1e_12': 0,
                'max_absolute_difference': 0., 'finite_pairs': 11999} for metric in ('auc', 'mrr', 'ndcg5', 'ndcg10')},
            'scheduling': {'strategy': 'stable_candidate_count', 'gbdt_only': True, 'batch_size': 64,
                'impressions': 12000, 'candidate_exposures': 100000, 'original_candidate_slots': 200000,
                'scheduled_candidate_slots': 150000, 'same_population': True,
                'same_candidates_within_each_impression': True, 'metric_rows_restored_to_original_order': True},
            'original': {'seconds': 2.}, 'scheduled': {'seconds': 1.}}
        validate_internal_audit(record, expected)
        mutations = [lambda r: r.update(checkpoint_sha256='changed'),
            lambda r: r.update(scheduler_source_sha256='changed'),
            lambda r: r.update(rows=512),
            lambda r: r.update(selected_row_indices_sha256='other-rows'),
            lambda r: r['metric_differences']['auc'].update(same_nan_mask=False),
            lambda r: r['metric_differences']['mrr'].update(max_absolute_difference=2e-12),
            lambda r: r['metric_differences']['ndcg5'].update(max_absolute_difference=float('nan')),
            lambda r: r['scheduling'].update(metric_rows_restored_to_original_order=False)]
        for mutate in mutations:
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_internal_audit(changed, expected)


if __name__ == '__main__':
    unittest.main()
