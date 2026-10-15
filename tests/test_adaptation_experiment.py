"""Synthetic integration checks for matched fitting, decisions and auditing."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import newsforge

_staged = Path(__file__).resolve().parents[1] / "newsforge"
if str(_staged) not in newsforge.__path__:
    newsforge.__path__.insert(0, str(_staged))

import numpy as np

from newsforge.adaptation import common, experiment
from newsforge.adaptation.common import METHODS, MODEL_NAMES, SEEDS
from newsforge.adaptation.replay import build_ema_snapshots
from newsforge.adaptation.scoring import (confirmation_passes, rank_methods,
                                         sample_pairs, select_delta)
from newsforge.external.common import HOUR, START, sha256
from newsforge.v3.residual import PairwiseResidual


def training_cache():
    """Two eligible reads plus three different pair-ineligible requests."""
    queries = np.array([[1, START, 7, 5], [2, START, 8, 2],
                        [3, START, 9, 99], [4, START, 10, 1],
                        [5, START, 11, 1]], dtype=np.int64)
    pools = np.zeros((5, 100), dtype=np.int32)
    pools[0, :16] = np.arange(1, 17)
    pools[1, :3] = [1, 2, 3]
    pools[2, :3] = [1, 2, 3]
    pools[3, :3] = [1, 2, 3]
    pools[4, 0] = 1
    english = np.zeros((5, 100, 13), dtype=np.float32)
    local = np.zeros_like(english)
    for row in range(5):
        for position in range(16):
            english[row, position] = np.linspace(-.3, .4, 13) + position / 30 + row / 10
            local[row, position] = np.linspace(.5, -.2, 13) + position / 25 - row / 12
        positive = np.flatnonzero(pools[row] == queries[row, 3])
        if len(positive):
            english[row, positive[0], 0] += 1
            local[row, positive[0], 4] += 1
    return {'queries': queries, 'pools': pools, 'sizes': np.array([16, 3, 3, 3, 1]),
            'history_lengths': np.array([5, 2, 5, 0, 4]), 'english': english, 'local': local}


class FixedScores:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def predict_raw(self, features):
        if len(features) != len(self.values):
            raise AssertionError("unexpected candidate count")
        return self.values.copy()


def fixed_collections(values):
    collection = [FixedScores(value) for value in values]
    return {name: collection for name in MODEL_NAMES}, collection


def linear_model(variant, seed):
    model = PairwiseResidual(variant)
    rng = np.random.default_rng(seed)
    count = len(model.feature_indices)
    model.mean = rng.uniform(-.05, .05, count)
    model.scale = rng.uniform(.9, 1.2, count)
    model.weights = rng.uniform(-.12, .12, count)
    return model


def linear_collections():
    return ({name: [linear_model('no_recent' if name == 'local_no_recent' else 'full13',
                                  seed + 100 * offset) for seed in SEEDS]
             for offset, name in enumerate(MODEL_NAMES, 1)},
            [linear_model('full13', seed) for seed in SEEDS])


class SamplingAndFittingIntegrationTests(unittest.TestCase):
    def test_shared_seed_pairs_exclude_target_and_account_for_all_requests(self):
        cache = training_cache()
        pairs, counts = sample_pairs(*(cache[key] for key in
            ('queries', 'pools', 'sizes', 'history_lengths')), SEEDS[0])
        repeated, repeated_counts = sample_pairs(*(cache[key] for key in
            ('queries', 'pools', 'sizes', 'history_lengths')), SEEDS[0])
        np.testing.assert_array_equal(pairs, repeated)
        self.assertEqual(counts, repeated_counts)
        self.assertEqual(counts, {'requests': 5, 'empty_history': 1, 'target_outside_pool': 1,
                                 'no_alternatives': 1, 'eligible_requests': 2, 'pairs': 10})
        self.assertEqual(set(pairs[:, 0]), {0, 1})
        for row in (0, 1):
            chosen = pairs[pairs[:, 0] == row]
            self.assertEqual(len(chosen), min(8, cache['sizes'][row] - 1))
            self.assertEqual(len(np.unique(chosen[:, 2])), len(chosen))
            for query_row, positive, alternative in chosen:
                self.assertEqual(cache['pools'][query_row, positive], cache['queries'][query_row, 3])
                self.assertNotEqual(cache['pools'][query_row, alternative], cache['queries'][query_row, 3])
        self.assertTrue(np.all(np.diff(pairs[:, 0]) >= 0))

    def test_actual_fit_uses_one_saved_pair_set_per_seed_for_every_component(self):
        cache = training_cache()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'prepare').mkdir()
            (run / 'manifest.json').write_text('{}', encoding='utf-8')
            (run / 'prepare/receipt.json').write_text('{}', encoding='utf-8')
            with patch.object(experiment, 'check_run', return_value={}), \
                 patch.object(experiment, 'phase_receipt', return_value={}), \
                 patch.object(experiment, 'load_cache', return_value=cache):
                receipt = experiment.fit(run, run, run)
            for seed in SEEDS:
                path = run / f'fit/pairs-seed{seed}.npy'
                pairs = np.load(path, allow_pickle=False)
                expected, _ = sample_pairs(*(cache[key] for key in
                    ('queries', 'pools', 'sizes', 'history_lengths')), seed)
                np.testing.assert_array_equal(pairs, expected)
                for name in MODEL_NAMES:
                    metadata = receipt['models'][f'{name}-seed{seed}']
                    self.assertEqual(metadata['pair_indices_sha256'], sha256(path))
                    self.assertEqual(metadata['requests'], 5)
                    self.assertEqual(metadata['pairs'], 10)
                    self.assertEqual(metadata['fit_metadata']['pairs'], 10)
                    model = PairwiseResidual.load(run / f'fit/models/{name}-seed{seed}.npz')
                    self.assertEqual(model.bindings['pair_indices_sha256'], sha256(path))
                    values = cache['english' if name == 'english' else 'local']
                    positive = values[pairs[:, 0], pairs[:, 1]].astype(np.float64)
                    negative = values[pairs[:, 0], pairs[:, 2]].astype(np.float64)
                    expected_mean = (positive[:, model.feature_indices].mean(axis=0)
                                     + negative[:, model.feature_indices].mean(axis=0)) / 2
                    np.testing.assert_allclose(model.mean, expected_mean, rtol=0, atol=1e-15)
                    self.assertTrue(np.isfinite(model.predict_raw(positive)).all())


class DecisionsAndPredictionTests(unittest.TestCase):
    def test_conservative_delta_ties_and_confirmation_fail_closed(self):
        grid = {str(delta): {'hits10': 42} for delta in (0, 2, 5, 10)}
        self.assertEqual(select_delta(grid), 0)
        grid['2']['hits10'] = grid['5']['hits10'] = 43
        self.assertEqual(select_delta(grid), 2)
        for delta, comparison in [
                (0, {'difference': .2, 'ci95': [.1, .3]}),
                (2, {'difference': 0, 'ci95': [0, .2]}),
                (2, {'difference': .1, 'ci95': [0, .2]}),
                (2, {'difference': .1, 'ci95': [-.01, .2]})]:
            self.assertFalse(confirmation_passes(delta, comparison))
        self.assertTrue(confirmation_passes(2, {'difference': .1, 'ci95': [.01, .2]}))

    def test_clipping_precedes_seed_average_and_no_gate_is_distinct(self):
        features = np.zeros((2, 13), dtype=np.float32)
        models, frozen = fixed_collections(([9, -9], [-2, 2], [-2, 2]))
        orders, scores = rank_methods(features, features, 5, models, frozen, 2, False)
        # Seedwise clipped mean is [-1/3,+1/3], opposite to clipping the raw mean.
        np.testing.assert_allclose(scores[4], [-1 / 3, -2 / 3], rtol=0, atol=1e-15)
        np.testing.assert_allclose(scores[6], [-2 / 3, -1 / 3], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(orders[4], [0, 1])
        np.testing.assert_array_equal(orders[6], [1, 0])
        np.testing.assert_array_equal(scores[7], scores[0])
        np.testing.assert_array_equal(orders[7], orders[0])
        _, promoted = rank_methods(features, features, 5, models, frozen, 2, True)
        np.testing.assert_array_equal(promoted[7], promoted[4])
        self.assertFalse(np.array_equal(scores[1], scores[2]))

    def test_empty_history_and_delta_zero_fallback_preserve_stable_ties(self):
        features = np.zeros((2, 13), dtype=np.float32)
        models, frozen = fixed_collections(([0, 1], [0, 1], [0, 1]))
        orders, scores = rank_methods(features, features, 5, models, frozen, 2, True)
        np.testing.assert_array_equal(scores[4], [0, 0])
        np.testing.assert_array_equal(orders[4], [0, 1])
        orders, scores = rank_methods(features, features, 0, models, frozen, 10, True)
        np.testing.assert_array_equal(orders, np.tile([0, 1], (len(METHODS), 1)))
        np.testing.assert_array_equal(scores, np.tile([0, -1], (len(METHODS), 1)))
        orders, scores = rank_methods(features, features, 5, models, frozen, 0, False)
        for method in (2, 3, 4, 5, 6, 7):
            np.testing.assert_array_equal(scores[method], scores[0])
        self.assertFalse(np.array_equal(scores[1], scores[0]))

    def test_pair_ineligible_requests_remain_in_prediction_and_summary_denominators(self):
        cache = training_cache()
        models, frozen = linear_collections()
        prediction = experiment._predict(cache, models, frozen, 0, False)
        summary = experiment.summarize(cache['queries'], prediction)
        self.assertEqual(summary['requests'], 5)
        for method in METHODS:
            self.assertEqual(summary['methods'][method]['denominator'], 5)
        baseline = summary['methods']['baseline']
        self.assertEqual(baseline['hits10'], 4)
        self.assertEqual(baseline['hit_rate10'], .8)
        self.assertTrue(np.all(prediction['hits10'][2] == 0))  # Target outside pool.
        self.assertTrue(np.all(prediction['hits10'][3] == 1))  # Empty history still evaluated.
        self.assertTrue(np.all(prediction['hits100'] == prediction['hits100'][:, :1]))
        np.testing.assert_array_equal(prediction['hits10'][:, 7], prediction['hits10'][:, 0])

    def test_summary_uint8_losses_are_negative_and_bootstrap_uses_user_clusters(self):
        queries = np.array([[i, START, 7 if i < 2 else 8, 1] for i in range(4)], dtype=np.int64)
        hits = np.zeros((4, len(METHODS)), dtype=np.uint8)
        hits[:2, 0] = 1
        hits[:, 7] = hits[:, 0]
        prediction = {'hits10': hits, 'hits100': np.ones_like(hits),
                      'reciprocal10': hits.astype(np.float64)}
        result = experiment.summarize(queries, prediction)
        comparison = result['comparisons']['adapted_local_minus_baseline']
        self.assertEqual(comparison['difference'], -.5)
        self.assertEqual(comparison['ci95'], [-1., 0.])
        self.assertEqual((comparison['wins'], comparison['losses'], comparison['ties']), (0, 2, 2))
        self.assertEqual(comparison['user_clusters'], 2)
        self.assertEqual(comparison['valid_repetitions'], 1000)
        self.assertEqual(result['comparisons']['deployed_minus_baseline']['ci95'], [0., 0.])


class IndependentAuditIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.run = Path(self.directory.name)
        self.connection = sqlite3.connect(':memory:')
        self.connection.executescript('''
            CREATE TABLE events(id INTEGER PRIMARY KEY,timestamp INTEGER,user INTEGER,
                                item INTEGER,member TEXT,line INTEGER);
            CREATE TABLE counts(hour INTEGER,item INTEGER,n INTEGER,PRIMARY KEY(hour,item));
        ''')
        events = [(item, START + item, 100 + item, item, 'a', item) for item in range(1, 18)]
        events += [(18, START + HOUR - 1, 7, 1, 'b', 1),
                   (19, START + HOUR + 1, 7, 2, 'b', 2)]
        self.connection.executemany('INSERT INTO events VALUES(?,?,?,?,?,?)', events)
        self.connection.execute('INSERT INTO counts SELECT timestamp/3600,item,COUNT(*) FROM events GROUP BY timestamp/3600,item')
        self.first_seen = np.array([-1] + [START + item for item in range(1, 18)], dtype=np.int64)
        rng = np.random.default_rng(291)
        self.english = np.zeros((18, 115), dtype=np.float32)
        self.local = np.zeros_like(self.english)
        for features in (self.english, self.local):
            text = rng.normal(size=(17, 96))
            features[1:, :96] = text / np.linalg.norm(text, axis=1, keepdims=True)
            features[1:, 96] = 1
        patterns = [(START + 3 * HOUR, 7, 3), (START + 3 * HOUR, 999, 17),
                    (START + 4 * HOUR, 7, 1), (START + HOUR, 999, 1)]
        queries = np.array([(1000 + i, *patterns[i % 4]) for i in range(128)], dtype=np.int64)
        self.coverage = {name: np.zeros((18, 4), dtype=np.uint8) for name in ('english', 'local')}
        (self.run / 'prepare').mkdir()
        experiment._cache_phase(self.connection, self.first_seen,
            build_ema_snapshots(self.connection, 18), self.english, self.local, self.coverage,
            queries, self.run / 'prepare/evaluate', lambda *args, **kwargs: None)
        mapped = experiment.load_cache(self.run, 'evaluate')
        self.cache = {name: np.array(value) for name, value in mapped.items()}
        for value in mapped.values():
            value._mmap.close()
        self.models, self.frozen = linear_collections()
        self.predictions = experiment._predict(self.cache, self.models, self.frozen, 5, False)

    def tearDown(self):
        self.connection.close()
        self.directory.cleanup()

    def audit(self, cache=None, predictions=None):
        return experiment._audit(self.connection, self.first_seen, self.english, self.local,
            self.cache if cache is None else cache, self.predictions if predictions is None else predictions,
            self.models, self.frozen, 5, False)

    def test_audit_independently_rebuilds_candidates_features_and_scalar_model_scores(self):
        with patch.object(PairwiseResidual, 'predict_raw', side_effect=AssertionError('audit reused model predictor')):
            result = self.audit()
        self.assertTrue(result['passed'])
        self.assertEqual(result['queries'], 128)
        self.assertLessEqual(result['score_max_absolute_error'], 1e-8)

    def test_audit_rejects_candidate_corruption(self):
        corrupted = {name: value.copy() for name, value in self.cache.items()}
        corrupted['pools'][0, 0] = 17 if corrupted['pools'][0, 0] != 17 else 16
        with self.assertRaisesRegex(ValueError, 'candidate audit mismatch'):
            self.audit(cache=corrupted)

    def test_audit_rejects_repredicted_cached_feature_corruption(self):
        corrupted = {name: value.copy() for name, value in self.cache.items()}
        corrupted['local'][0, 0, 4] += .125
        changed_prediction = experiment._predict(corrupted, self.models, self.frozen, 5, False)
        with self.assertRaisesRegex(ValueError, 'scoring audit mismatch'):
            self.audit(cache=corrupted, predictions=changed_prediction)

    def test_audit_rejects_saved_hit_corruption_even_if_rankings_match(self):
        corrupted = {name: value.copy() for name, value in self.predictions.items()}
        corrupted['hits10'][0, 4] ^= 1
        with self.assertRaisesRegex(ValueError, 'hit audit mismatch'):
            self.audit(predictions=corrupted)


class PhaseBindingTests(unittest.TestCase):
    def test_phase_receipt_rejects_artifact_manifest_and_predecessor_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'manifest.json').write_text('{"synthetic":true}', encoding='utf-8')
            (run / 'prepare').mkdir()
            (run / 'prepare/cache.bin').write_bytes(b'invented cache')
            common.finish_phase(run, 'prepare', ['cache.bin'])
            (run / 'fit').mkdir()
            artifact = run / 'fit/model.bin'
            artifact.write_bytes(b'invented model')
            common.finish_phase(run, 'fit', ['model.bin'],
                prepare_receipt_sha256=sha256(run / 'prepare/receipt.json'))
            self.assertEqual(common.phase_receipt(run, 'fit')['status'], 'passed')
            artifact.write_bytes(b'tampered model')
            with self.assertRaisesRegex(ValueError, 'artifact binding mismatch'):
                common.phase_receipt(run, 'fit')
            artifact.write_bytes(b'invented model')
            predecessor = run / 'prepare/receipt.json'
            original = predecessor.read_bytes()
            predecessor.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                common.phase_receipt(run, 'fit')
            predecessor.write_bytes(original)
            (run / 'manifest.json').write_text('{"synthetic":false}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'manifest or status mismatch'):
                common.phase_receipt(run, 'fit')

    def test_check_run_rejects_source_runtime_and_base_receipt_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, run = root / 'base', root / 'run'
            run.mkdir()
            for phase in ('prepare', 'evaluate'):
                (base / phase).mkdir(parents=True)
                (base / phase / 'receipt.json').write_text('{}', encoding='utf-8')
            old_report = root / 'reports/external/adressa_v1/evaluation.json'
            old_report.parent.mkdir(parents=True)
            old_report.write_text('{"synthetic_old_result":true}', encoding='utf-8')
            manifest = {'config': common.CONFIG, 'sources': {'synthetic.py': 'fixed'},
                        'environment': {'python': 'synthetic'}, 'base_receipts':
                        {phase: sha256(base / phase / 'receipt.json') for phase in ('prepare', 'evaluate')},
                        'existing_external_result_sha256': sha256(old_report)}
            (run / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with patch.object(common, 'source_bindings', return_value=manifest['sources']) as sources, \
                 patch.object(common, 'environment', return_value=manifest['environment']) as runtime:
                self.assertEqual(common.check_run(root, base, run), manifest)
                sources.return_value = {'synthetic.py': 'changed'}
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    common.check_run(root, base, run)
                sources.return_value = manifest['sources']
                runtime.return_value = {'python': 'changed'}
                with self.assertRaisesRegex(ValueError, 'runtime changed'):
                    common.check_run(root, base, run)
                runtime.return_value = manifest['environment']
                old_report.write_text('{"synthetic_old_result":false}', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'original external result changed'):
                    common.check_run(root, base, run)
                old_report.write_text('{"synthetic_old_result":true}', encoding='utf-8')
                (base / 'prepare/receipt.json').write_text('{"changed":true}', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'base receipt changed'):
                    common.check_run(root, base, run)


if __name__ == '__main__':
    unittest.main()
