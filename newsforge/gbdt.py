"""Content/history gradient-boosted-tree ranking baseline, with a locked test split.

The binary objective uses one positive and four same-impression non-clicks. Its
probabilities are sampled-training scores, not calibrated production click rates.
Evaluation always restores every logged candidate and uses MIND macro metrics.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import threading
import time

import joblib
import numpy as np
from numpy.lib.format import open_memmap
import psutil
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .evaluate import select_validation_rows
from .metrics import METRICS, impression_metrics
from .store import Store


FEATURE_NAMES = (
    "history_cosine_mean", "history_cosine_max", "history_cosine_top3_mean",
    "history_cosine_std", "text_cosine_mean", "text_cosine_max",
    "text_cosine_top3_mean", "mean_profile_text_cosine", "mean_profile_text_l1",
    "mean_profile_text_l2", "category_overlap_mean", "category_overlap_max",
    "last_valid_history_text_cosine", "log1p_history_length", "empty_history",
)
FEATURE_VERSION = 2


def feature_names(feature_dim, text_dimensions=96, feature_mode="similarity"):
    if feature_mode == "similarity":
        return FEATURE_NAMES
    if feature_mode != "rich" or not 0 < text_dimensions < feature_dim:
        raise ValueError("feature_mode must be similarity or rich with valid text dimensions")
    width = min(16, text_dimensions)
    categories = feature_dim - text_dimensions
    return FEATURE_NAMES + tuple(f"candidate_svd_{i}" for i in range(width)) + tuple(
        f"mean_history_svd_{i}" for i in range(width)) + tuple(
        f"candidate_category_{i}" for i in range(categories)) + tuple(
        f"mean_history_category_{i}" for i in range(categories))


class MemoryMonitor:
    """Observed process RSS and system availability; sampled peaks are lower bounds."""

    def __enter__(self):
        self.process = psutil.Process()
        self.initial_rss = self.process.memory_info().rss
        self.peak_rss = self.initial_rss
        self.minimum_available = psutil.virtual_memory().available
        self.samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)
            self.samples += 1
            self.stop.wait(.1)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def summary(self):
        return {"initial_rss_bytes": self.initial_rss, "max_observed_rss_bytes": self.peak_rss,
                "minimum_system_available_bytes": self.minimum_available, "samples": self.samples,
                "sampling_interval_seconds": .1, "scope": "whole GBDT process including mapped pages; sampled RSS is not an allocation-only or exact peak measure"}


def memory_preflight(examples, columns, *, memmap=False):
    matrix_bytes = examples * columns * (8 if memmap else 4)
    # sklearn 1.6 HGB validates X as float64. Native float64 memmap avoids a
    # simultaneous float32 matrix plus a full-size float64 conversion.
    conversion_bytes = 0 if memmap else examples * columns * 8
    binned_bytes = examples * columns
    working_reserve = max(512 * 2**20, examples * 40)
    required = matrix_bytes + conversion_bytes + binned_bytes + working_reserve
    available = psutil.virtual_memory().available
    result = {"available_bytes": available, "matrix_bytes": matrix_bytes,
              "estimated_float64_conversion_bytes": conversion_bytes, "estimated_binned_bytes": binned_bytes,
              "working_and_system_reserve_bytes": working_reserve, "required_available_bytes": required,
              "storage": "float64 Fortran-order memmap" if memmap else "float32 in-memory",
              "allowed": available >= required}
    if not result["allowed"]:
        raise MemoryError(f"GBDT memory preflight refused: {json.dumps(result)}")
    return result


def _normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def _history_statistics(similarities, mask):
    """Statistics exclude PAD, including when valid cosine similarities are negative."""
    count = mask.sum(axis=1)[:, None]
    denominator = np.maximum(count, 1)
    valid = mask[:, None, :]
    values = np.where(valid, similarities, 0)
    mean = values.sum(axis=-1) / denominator
    variance = (values * values).sum(axis=-1) / denominator - mean * mean
    masked = np.where(valid, similarities, -np.inf)
    maximum = np.max(masked, axis=-1)
    maximum = np.where(count > 0, maximum, 0)
    k = min(3, similarities.shape[-1])
    top = np.partition(masked, similarities.shape[-1] - k, axis=-1)[..., -k:]
    top_mean = np.where(np.isfinite(top), top, 0).sum(axis=-1) / np.maximum(np.minimum(count, k), 1)
    return mean, maximum, top_mean, np.sqrt(np.maximum(variance, 0))


def pair_features(history_features, candidate_features, history_mask=None, *, text_dimensions=96,
                  feature_mode="similarity"):
    """Return float32 [batch, candidates, features] without labels or ID statistics.

    Inputs are the fixed train-fitted SVD text vector followed by category one-hot
    columns. A candidate has no influence on another candidate's features. The
    largest interaction tensor is [batch, candidates, history], never [B,C,H,D].
    """
    history = np.asarray(history_features, dtype=np.float32)
    candidates = np.asarray(candidate_features, dtype=np.float32)
    if history.ndim != 3 or candidates.ndim != 3:
        raise ValueError("history and candidates must have shapes [B,H,D] and [B,C,D]")
    if history.shape[0] != candidates.shape[0] or history.shape[2] != candidates.shape[2]:
        raise ValueError("history and candidates must share batch size and feature dimension")
    if not 0 < text_dimensions < history.shape[-1]:
        raise ValueError("text_dimensions must leave at least one category column")
    feature_names(history.shape[-1], text_dimensions, feature_mode)
    inferred = np.any(history != 0, axis=-1)
    if history_mask is not None:
        provided = np.asarray(history_mask, dtype=bool)
        if provided.shape != history.shape[:2]:
            raise ValueError("history_mask must have shape [B,H]")
        inferred &= provided
    # A synthetic PAD slot makes zero-length histories follow the same finite path.
    if history.shape[1] == 0:
        history = np.zeros((history.shape[0], 1, history.shape[-1]), dtype=np.float32)
        inferred = np.zeros(history.shape[:2], dtype=bool)
    history = np.where(inferred[..., None], history, 0)
    text_history = history[..., :text_dimensions]
    text_candidates = candidates[..., :text_dimensions]
    full_similarity = np.einsum("bcd,bhd->bch", _normalize(candidates), _normalize(history), optimize=True)
    text_similarity = np.einsum("bcd,bhd->bch", _normalize(text_candidates), _normalize(text_history), optimize=True)
    full_mean, full_max, full_top3, full_std = _history_statistics(full_similarity, inferred)
    text_mean, text_max, text_top3, _ = _history_statistics(text_similarity, inferred)
    count = inferred.sum(axis=1)
    mean_text = text_history.sum(axis=1) / np.maximum(count[:, None], 1)
    profile_cosine = np.einsum("bcd,bd->bc", _normalize(text_candidates), _normalize(mean_text), optimize=True)
    difference = text_candidates - mean_text[:, None, :]
    category_similarity = np.einsum("bcd,bhd->bch", candidates[..., text_dimensions:], history[..., text_dimensions:], optimize=True)
    category_mean, category_max, _, _ = _history_statistics(category_similarity, inferred)
    last_position = np.where(inferred, np.arange(history.shape[1])[None, :], -1).max(axis=1)
    last_similarity = np.take_along_axis(text_similarity, np.maximum(last_position[:, None, None], 0), axis=2)[..., 0]
    last_similarity = np.where(count[:, None] > 0, last_similarity, 0)
    output_shape = candidates.shape[:2]
    output = np.stack((full_mean, full_max, full_top3, full_std, text_mean, text_max,
        text_top3, profile_cosine, np.abs(difference).mean(axis=-1), np.linalg.norm(difference, axis=-1),
        category_mean, category_max, last_similarity,
        np.broadcast_to(np.log1p(count)[:, None], output_shape),
        np.broadcast_to((count == 0)[:, None], output_shape)), axis=-1)
    if feature_mode == "rich":
        width = min(16, text_dimensions)
        mean_category = history[..., text_dimensions:].sum(axis=1) / np.maximum(count[:, None], 1)
        output = np.concatenate((output, text_candidates[..., :width],
            np.broadcast_to(mean_text[:, None, :width], (*output_shape, width)),
            candidates[..., text_dimensions:],
            np.broadcast_to(mean_category[:, None, :], (*output_shape, mean_category.shape[-1]))), axis=-1)
    return output.astype(np.float32, copy=False)


def _text_dimensions(store):
    return int(store.manifest["features"]["text_dimensions"])


def _manifest_signature(store):
    value = json.dumps(store.manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_positive_indices(total, max_positives=500000, seed=2026):
    if total < 1 or max_positives < 1:
        raise ValueError("training requires positives and a positive sample cap")
    if total <= max_positives:
        return np.arange(total, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(total, max_positives, replace=False))


def training_matrix(store, *, max_positives=500000, seed=2026, batch_size=1024,
                    feature_mode="similarity", matrix_path=None):
    """Sample positive pairs uniformly; fetch four negatives from each pair's impression."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    store.ensure_training_cache()
    selected = sample_positive_indices(len(store.positive_rows), max_positives, seed)
    examples = len(selected) * 5
    dimensions = _text_dimensions(store)
    names = feature_names(store.features.shape[1], dimensions, feature_mode)
    matrix = (open_memmap(matrix_path, mode="w+", dtype=np.float64, shape=(examples, len(names)), fortran_order=True)
              if matrix_path is not None else np.empty((examples, len(names)), dtype=np.float32))
    labels = np.tile(np.array([1, 0, 0, 0, 0], dtype=np.uint8), len(selected))
    negative_rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for start in range(0, len(selected), batch_size):
        indices = selected[start:start + batch_size]
        rows = np.asarray(store.positive_rows[indices])
        if np.any(store.split[rows] != 0):
            raise ValueError("positive cache contains a held-out impression")
        pool = np.asarray(store.pool[rows])
        if pool.ndim != 2 or not pool.shape[1] or np.any(pool <= 0):
            raise ValueError("training negative cache must contain non-PAD article IDs")
        columns = negative_rng.integers(0, pool.shape[1], size=(len(rows), 4))
        negatives = np.take_along_axis(pool, columns, axis=1)
        positives = np.asarray(store.positive_items[indices])[:, None]
        if np.any(positives <= 0) or np.any(negatives == positives):
            raise ValueError("negative cache overlaps the positive article or contains PAD")
        candidate_ids = np.concatenate((positives, negatives), axis=1)
        history, mask = store.history_batch(rows)
        features = pair_features(history, store.features[candidate_ids], mask, text_dimensions=dimensions,
                                 feature_mode=feature_mode)
        matrix[start * 5:(start + len(indices)) * 5] = features.reshape(-1, len(names))
        if start and start % (batch_size * 100) == 0:
            print(f"GBDT features: {start:,}/{len(selected):,} positive pairs", flush=True)
    metadata = {
        "eligible_training_positive_pairs": len(store.positive_rows),
        "selected_positive_pairs": len(selected), "binary_training_examples": examples,
        "positive_pair_cap": max_positives, "positive_pairs_subsampled": len(selected) < len(store.positive_rows),
        "positive_pair_selection": "uniform without replacement, sorted cache indices; all when below cap",
        "selected_positive_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "seed": seed, "negatives_per_positive": 4, "negative_sampling_with_replacement": True,
        "negative_policy": "four draws from Store's fixed pool of label=0 articles in the same split-0 impression",
        "feature_mode": feature_mode, "feature_count": len(names), "feature_matrix_dtype": str(matrix.dtype),
        "feature_matrix_storage": "memmap" if matrix_path is not None else "memory",
        "feature_matrix_bytes": matrix.nbytes, "feature_construction_seconds": time.perf_counter() - started,
        "calibration": "sampled 1:4 binary objective; probabilities are ranking scores, not population CTR",
    }
    cache_path = store.path / "training_cache.json"
    if cache_path.exists():
        metadata["source_training_cache"] = json.loads(cache_path.read_text(encoding="utf-8"))
    if matrix_path is not None:
        matrix.flush()
    return matrix, labels, metadata


def _summarize(values):
    # Null, rather than JSON NaN, is explicit for groups without eligible labels.
    return {name: {"mean": float(values[np.isfinite(values[:, i]), i].mean())
                  if np.isfinite(values[:, i]).any() else None,
                  "eligible_impressions": int(np.isfinite(values[:, i]).sum())}
            for i, name in enumerate(METRICS)}


def evaluate_gbdt(store, artifact, rows, *, batch_size=64, output=None, allow_test=False):
    """Score full impressions. Split 2 requires an explicit test unlock."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or batch_size < 1:
        raise ValueError("nonempty one-dimensional rows and positive batch_size required")
    if np.any(rows < 0) or np.any(rows >= len(store.split)):
        raise ValueError("evaluation row is outside Store")
    allowed = (1, 2) if allow_test else (1,)
    if not np.isin(store.split[rows], allowed).all():
        raise ValueError("only internal-validation rows are allowed until --evaluate-test is explicit")
    feature_mode = artifact.get("feature_mode", "similarity")
    expected_names = feature_names(store.features.shape[1], artifact["text_dimensions"], feature_mode)
    if (artifact["feature_version"] not in {1, FEATURE_VERSION}
            or (artifact["feature_version"] == 1 and feature_mode != "similarity")
            or tuple(artifact["feature_names"]) != expected_names):
        raise ValueError("GBDT feature schema mismatch")
    if artifact["manifest_sha256"] != _manifest_signature(store):
        raise ValueError("model was trained against a different processed-data manifest")
    model = artifact["model"]
    started = time.perf_counter()
    values = np.empty((len(rows), len(METRICS)), dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        ids, candidate_mask = store.padded_candidates(batch)
        history, mask = store.history_batch(batch)
        features = pair_features(history, store.features[ids], mask, text_dimensions=artifact["text_dimensions"],
                                 feature_mode=feature_mode)
        scores = np.zeros(candidate_mask.shape, dtype=np.float64)
        # Do not fit or score padding candidates. Every real candidate is retained.
        scores[candidate_mask] = model.predict_proba(features[candidate_mask])[:, 1]
        for offset, row in enumerate(batch):
            _, labels = store.candidates_for(row)
            values[start + offset] = impression_metrics(labels, scores[offset, :len(labels)])
    history_lengths = (store.history[rows] > 0).sum(axis=1)
    group_masks = {
        "empty_history": history_lengths == 0,
        "history_1_to_5": (history_lengths > 0) & (history_lengths <= 5),
        "history_6_to_20": history_lengths > 5,
    }
    for name in ("has_train_unseen_clicked_news", "has_zero_train_click_news"):
        group_masks[name] = np.zeros(len(rows), dtype=bool)
    cutoff = int(store.manifest["train_cutoff_timestamp"])
    for offset, row in enumerate(rows):
        ids, labels = store.candidates_for(row)
        positives = ids[labels == 1]
        group_masks["has_train_unseen_clicked_news"][offset] = np.any(store.first_seen[positives] > cutoff)
        group_masks["has_zero_train_click_news"][offset] = np.any(store.clicks[positives] == 0)
    report = {"impressions": len(rows),
        "candidate_exposures": int((store.offsets[rows + 1] - store.offsets[rows]).sum()),
        "metrics": _summarize(values),
        "groups": {name: {"impressions": int(mask.sum()), "metrics": _summarize(values[mask]) if mask.any() else None}
                   for name, mask in group_masks.items()},
        "seconds": time.perf_counter() - started,
        "protocol": "all logged candidates; impression macro average; no evaluation negative sampling",
        "metric_columns": list(METRICS), "feature_mode": feature_mode, "feature_count": len(expected_names)}
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, values)
    return report, values


def train_gbdt(store, *, artifacts="artifacts", reports="reports", max_positives=500000,
               seed=2026, threads=2, batch_size=1024, validation_impressions=12000,
               feature_mode="similarity", run_name="gbdt"):
    """Fit a fixed 80-round baseline; no official-dev evaluation occurs here."""
    if threads < 1 or validation_impressions < 1:
        raise ValueError("threads and validation_impressions must be positive")
    artifacts, reports = Path(artifacts), Path(reports)
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    if run_name not in {"gbdt", "gbdt_similarity", "gbdt_rich"}:
        raise ValueError("unsupported GBDT run name")
    started = time.perf_counter()
    names = feature_names(store.features.shape[1], _text_dimensions(store), feature_mode)
    store.ensure_training_cache()
    examples = min(max_positives, len(store.positive_rows)) * 5
    preflight = memory_preflight(examples, len(names), memmap=feature_mode == "rich")
    print(json.dumps({"gbdt_memory_preflight": preflight, "feature_mode": feature_mode, "feature_count": len(names)}), flush=True)
    config = dict(max_iter=80, learning_rate=0.1, max_leaf_nodes=31,
                  l2_regularization=1.0, early_stopping=False, random_state=seed)
    validation_rows = select_validation_rows(store, validation_impressions)
    if not len(validation_rows):
        raise ValueError("no internal-validation impressions available")
    matrix_path = artifacts / f".{run_name}_training_features.npy" if feature_mode == "rich" else None
    with threadpool_limits(limits=threads), MemoryMonitor() as memory:
        matrix, labels, sampling = training_matrix(store, max_positives=max_positives,
                                                   seed=seed, batch_size=batch_size,
                                                   feature_mode=feature_mode, matrix_path=matrix_path)
        fit_started = time.perf_counter()
        model = HistGradientBoostingClassifier(**config)
        try:
            model.fit(matrix, labels)
        finally:
            if isinstance(matrix, np.memmap):
                matrix._mmap.close()
            del matrix, labels
            if matrix_path is not None:
                matrix_path.unlink(missing_ok=True)
        fit_seconds = time.perf_counter() - fit_started
        artifact = {"model": model, "model_name": "hist_gradient_boosting",
                    "feature_names": list(names), "feature_version": FEATURE_VERSION, "feature_mode": feature_mode,
                    "text_dimensions": _text_dimensions(store), "manifest_sha256": _manifest_signature(store),
                    "config": config, "sampling": sampling}
        validation, _ = evaluate_gbdt(store, artifact, validation_rows,
                                      output=artifacts / f"metrics_{run_name}_validation.npy")
    artifact["internal_validation"] = validation
    artifact["validation_row_indices_sha256"] = hashlib.sha256(validation_rows.astype("<i8").tobytes()).hexdigest()
    joblib.dump(artifact, artifacts / f"{run_name}.joblib", compress=3)
    np.save(artifacts / f"evaluation_rows_{run_name}_validation.npy", validation_rows)
    report = {"model": "HistGradientBoostingClassifier", "feature_names": list(names),
        "feature_mode": feature_mode, "feature_count": len(names), "memory_preflight": preflight,
        "memory_observed": memory.summary(),
        "feature_version": FEATURE_VERSION, "config": config, "sampling": sampling,
        "internal_validation": validation, "validation_row_indices_sha256": artifact["validation_row_indices_sha256"],
        "selection": "fixed 80 rounds, learning rate 0.1, 31 leaves; no hyperparameter search or early stopping",
        "official_development_test_evaluated": False, "fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started, "threads": threads,
        "data_manifest_sha256": artifact["manifest_sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "platform": platform.platform()},
        "scope": "strong handcrafted-feature tree baseline; popularity/content are separate sanity baselines",
        "feature_policy": "only fixed train-fitted text/category features and provided past history; no IDs, labels, or CTR target encoding"}
    (reports / f"{run_name}_training.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"gbdt_training": report["sampling"], "internal_validation": validation["metrics"],
                      "fit_seconds": fit_seconds}), flush=True)
    return artifact, report


def assert_test_locked(artifacts, reports):
    markers = [Path(artifacts) / name for name in ("evaluation_rows.npy", "metrics_gbdt.npy")]
    markers += [Path(reports) / name for name in ("ranking_experiment.json", "gbdt_evaluation.json")]
    if any(path.exists() for path in markers):
        raise ValueError("GBDT feature amendment requires official dev to remain locked")


def _files_for_run(artifacts, reports, name):
    return [Path(artifacts) / f"{name}.joblib",
            Path(artifacts) / f"metrics_{name}_validation.npy",
            Path(artifacts) / f"evaluation_rows_{name}_validation.npy",
            Path(reports) / f"{name}_training.json"]


def preserve_similarity(artifacts, reports):
    """Keep the original model and evidence byte-for-byte before the amendment."""
    assert_test_locked(artifacts, reports)
    destination = _files_for_run(artifacts, reports, "gbdt_similarity")
    if all(path.exists() for path in destination):
        report = json.loads(destination[-1].read_text(encoding="utf-8"))
        if report.get("feature_mode", "similarity") != "similarity" or len(report["feature_names"]) != 15:
            raise ValueError("preserved GBDT is not the original similarity baseline")
        return
    source = _files_for_run(artifacts, reports, "gbdt")
    if not all(path.exists() for path in source):
        raise ValueError("complete original GBDT model and validation evidence required before rich training")
    report = json.loads(source[-1].read_text(encoding="utf-8"))
    if report.get("feature_mode", "similarity") != "similarity" or len(report["feature_names"]) != 15:
        raise ValueError("refusing to label a non-similarity model as the original baseline")
    for old, new in zip(source, destination):
        if new.exists() and new.read_bytes() != old.read_bytes():
            raise ValueError(f"refusing to replace preserved evidence: {new}")
        shutil.copy2(old, new)
    old_log, new_log = Path(reports) / "gbdt_training.log", Path(reports) / "gbdt_similarity_training.log"
    if old_log.exists() and not new_log.exists():
        shutil.copy2(old_log, new_log)


def select_feature_ablation(artifacts, reports):
    """Use the same internal validation rows, with no official-dev label access."""
    artifacts, reports = Path(artifacts), Path(reports)
    assert_test_locked(artifacts, reports)
    records, models, row_ids = {}, {}, {}
    for mode in ("similarity", "rich"):
        paths = _files_for_run(artifacts, reports, f"gbdt_{mode}")
        if not all(path.exists() for path in paths):
            raise ValueError(f"complete {mode} evidence required")
        models[mode] = joblib.load(paths[0])
        record = json.loads(paths[-1].read_text(encoding="utf-8"))
        row_ids[mode] = np.load(paths[2])
        values = np.load(paths[1])
        row_hash = hashlib.sha256(row_ids[mode].astype("<i8").tobytes()).hexdigest()
        if (record.get("official_development_test_evaluated")
                or record.get("feature_mode", "similarity") != mode
                or record["validation_row_indices_sha256"] != row_hash
                or models[mode]["validation_row_indices_sha256"] != row_hash
                or values.shape != (len(row_ids[mode]), len(METRICS))
                or record["internal_validation"]["impressions"] != len(row_ids[mode])):
            raise ValueError("feature ablation evidence or validation population mismatch")
        observed = _summarize(values)
        for metric in METRICS:
            expected = record["internal_validation"]["metrics"][metric]["mean"]
            if expected is None or not np.isclose(observed[metric]["mean"], expected, rtol=0, atol=1e-12):
                raise ValueError("saved metric matrix differs from the validation report")
        records[mode] = record
    first, second = records["similarity"], records["rich"]
    sampling_keys = ("eligible_training_positive_pairs", "selected_positive_pairs", "binary_training_examples",
                     "selected_positive_indices_sha256", "seed", "negatives_per_positive", "source_training_cache")
    if (first["config"] != second["config"] or first["data_manifest_sha256"] != second["data_manifest_sha256"]
            or first["threads"] != second["threads"]
            or any(first["sampling"].get(key) != second["sampling"].get(key) for key in sampling_keys)
            or not np.array_equal(row_ids["similarity"], row_ids["rich"])):
        raise ValueError("feature comparisons require identical samples, estimator budget, data and validation rows")
    winner = max(("similarity", "rich"), key=lambda mode: records[mode]["internal_validation"]["metrics"]["ndcg10"]["mean"])
    comparison = {
        "amendment_stage": "added during internal validation while official dev remained locked; not part of the original feature protocol",
        "reason": "give the tree baseline access to additional raw content information also available to neural models",
        "original_protocol": "similarity-only 15 features, fixed 80 rounds; original model/report/validation arrays retained byte-for-byte",
        "selection_metric": "internal-validation impression-macro ndcg10", "tie_break": "similarity",
        "selected_feature_mode": winner, "official_development_test_evaluated": False,
        "shared_validation_row_indices_sha256": first["validation_row_indices_sha256"],
        "shared_selected_positive_indices_sha256": first["sampling"]["selected_positive_indices_sha256"],
        "configurations": {mode: {"feature_count": len(record["feature_names"]),
            "internal_validation": record["internal_validation"], "config": record["config"],
            "sampling": record["sampling"], "threads": record["threads"], "fit_seconds": record["fit_seconds"],
            "total_seconds": record["total_seconds"], "memory_observed": record.get("memory_observed"),
            "preserved_training_report": f"gbdt_{mode}_training.json",
            "preserved_model_sha256": hashlib.sha256((artifacts / f"gbdt_{mode}.joblib").read_bytes()).hexdigest()}
            for mode, record in records.items()},
    }
    selected_paths = _files_for_run(artifacts, reports, f"gbdt_{winner}")
    canonical = _files_for_run(artifacts, reports, "gbdt")
    for source, destination in zip(selected_paths[:-1], canonical[:-1]):
        shutil.copy2(source, destination)
    selected_report = dict(records[winner])
    selected_report.update(feature_mode=winner, feature_count=len(selected_report["feature_names"]),
        selection="two feature configurations selected by the same internal validation ndcg10; fixed 80 rounds and remaining hyperparameters; official dev locked",
        feature_selection={"report": "gbdt_feature_ablation.json", "selected_feature_mode": winner,
                           "amendment_stage": comparison["amendment_stage"]})
    canonical[-1].write_text(json.dumps(selected_report, indent=2, allow_nan=False), encoding="utf-8")
    (reports / "gbdt_feature_ablation.json").write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"gbdt_selected_feature_mode": winner, "validation_ndcg10": {
        mode: record["internal_validation"]["metrics"]["ndcg10"]["mean"] for mode, record in records.items()}}), flush=True)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true", help="default: fit and validate without opening test")
    mode.add_argument("--evaluate-test", action="store_true", help="explicitly score official dev using the saved artifact; never fit")
    mode.add_argument("--select-feature-ablation", action="store_true", help="choose similarity/rich using saved internal validation only")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--max-positives", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024, help="feature construction batch size, measured in positive pairs")
    parser.add_argument("--validation-impressions", type=int, default=12000)
    parser.add_argument("--feature-mode", choices=("similarity", "rich"), default="similarity")
    parser.add_argument("--run-name", choices=("gbdt", "gbdt_similarity", "gbdt_rich"))
    args = parser.parse_args()
    if min(args.max_positives, args.threads, args.batch_size, args.validation_impressions) < 1:
        parser.error("sample cap, threads, batch size and validation size must be positive")
    if args.select_feature_ablation:
        select_feature_ablation(args.artifacts, args.reports)
        return
    store = Store(args.data)
    if not args.evaluate_test:
        if args.feature_mode == "rich":
            preserve_similarity(args.artifacts, args.reports)
        run_name = args.run_name or ("gbdt_rich" if args.feature_mode == "rich" else "gbdt")
        train_gbdt(store, artifacts=args.artifacts, reports=args.reports,
                   max_positives=args.max_positives, seed=args.seed, threads=args.threads,
                   batch_size=args.batch_size, validation_impressions=args.validation_impressions,
                   feature_mode=args.feature_mode, run_name=run_name)
        return
    artifact = joblib.load(args.artifacts / "gbdt.joblib")
    rows = np.flatnonzero(store.split == 2)
    with threadpool_limits(limits=args.threads):
        report, _ = evaluate_gbdt(store, artifact, rows, allow_test=True,
                                  output=args.artifacts / "metrics_gbdt.npy")
    np.save(args.artifacts / "evaluation_rows_gbdt.npy", rows)
    args.reports.mkdir(parents=True, exist_ok=True)
    report.update({"model": "HistGradientBoostingClassifier", "artifact": str(args.artifacts / "gbdt.joblib"),
                   "official_development_test_evaluated": True, "threads": args.threads})
    (args.reports / "gbdt_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["metrics"]), flush=True)


if __name__ == "__main__":
    main()
