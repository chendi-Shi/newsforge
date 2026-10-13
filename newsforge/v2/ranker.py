"""Small CPU tree rankers with causal, delayed-feedback context.

The 15 NumPy content features reproduce this project's v1 gbdt.pair_features
implementation. They are kept here to avoid its indirect PyTorch import.
All scores are from a sampled 1:4 objective, not calibrated population CTR.
The CLI trains/selects on internal data only; it has no official-dev mode.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time

import joblib
import numpy as np
from numpy.lib.format import open_memmap
import psutil
from threadpoolctl import threadpool_limits

from ..metrics import METRICS, impression_metrics, summarize
from ..store import Store


CONTENT_NAMES = (
    "history_cosine_mean", "history_cosine_max", "history_cosine_top3_mean",
    "history_cosine_std", "text_cosine_mean", "text_cosine_max",
    "text_cosine_top3_mean", "mean_profile_text_cosine", "mean_profile_text_l1",
    "mean_profile_text_l2", "category_overlap_mean", "category_overlap_max",
    "last_valid_history_text_cosine", "log1p_history_length", "empty_history",
)
CONTEXT_NAMES = (
    "ema_log1p_clicks", "ema_log1p_exposures", "ema_smoothed_ctr",
    "log1p_observed_age_hours", "known_before_feedback_cutoff",
    "no_past_exposure", "past_global_click_prior",
)
FULL_FEATURE_NAMES = CONTENT_NAMES + CONTEXT_NAMES
VARIANTS = ("content_only", "context_only", "full")
SEEDS = (17, 29, 43)
FEATURE_VERSION = "causal_ranker_v2_1"
INTERNAL_MIN_TIMESTAMP = 1573722000  # 2019-11-14 09:00:00 UTC coordinate.
SAMPLING_SEED = 2026
MODEL_CONFIG = dict(max_iter=80, max_leaf_nodes=15, max_bins=63,
                    l2_regularization=1., learning_rate=.1, early_stopping=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def feature_indices(variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown feature variant: {variant}")
    return tuple(range(15)) if variant == "content_only" else (
        tuple(range(15, 22)) if variant == "context_only" else tuple(range(22)))


def _normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def _history_statistics(similarities, mask):
    count = mask.sum(axis=1)[:, None]
    denominator = np.maximum(count, 1)
    valid = mask[:, None, :]
    values = np.where(valid, similarities, 0)
    mean = values.sum(axis=-1) / denominator
    variance = (values * values).sum(axis=-1) / denominator - mean * mean
    masked = np.where(valid, similarities, -np.inf)
    maximum = np.where(count > 0, np.max(masked, axis=-1), 0)
    k = min(3, similarities.shape[-1])
    top = np.partition(masked, similarities.shape[-1] - k, axis=-1)[..., -k:]
    top_mean = np.where(np.isfinite(top), top, 0).sum(axis=-1) / np.maximum(np.minimum(count, k), 1)
    return mean, maximum, top_mean, np.sqrt(np.maximum(variance, 0))


def content_features(history_features, candidate_features, history_mask=None, *, text_dimensions=96):
    """v1's fixed 15-feature NumPy definition, without importing its trainer."""
    history = np.asarray(history_features, dtype=np.float32)
    candidates = np.asarray(candidate_features, dtype=np.float32)
    if history.ndim != 3 or candidates.ndim != 3:
        raise ValueError("history/candidates must have shape [B,H,D]/[B,C,D]")
    if history.shape[0] != candidates.shape[0] or history.shape[2] != candidates.shape[2]:
        raise ValueError("history/candidates must share batch and feature dimensions")
    if not 0 < text_dimensions < history.shape[-1]:
        raise ValueError("text_dimensions must leave category columns")
    inferred = np.any(history != 0, axis=-1)
    if history_mask is not None:
        provided = np.asarray(history_mask, dtype=bool)
        if provided.shape != history.shape[:2]:
            raise ValueError("history mask shape mismatch")
        inferred &= provided
    if history.shape[1] == 0:
        history = np.zeros((history.shape[0], 1, history.shape[-1]), dtype=np.float32)
        inferred = np.zeros(history.shape[:2], dtype=bool)
    history = np.where(inferred[..., None], history, 0)
    text_history, text_candidates = history[..., :text_dimensions], candidates[..., :text_dimensions]
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
    shape = candidates.shape[:2]
    output = np.stack((full_mean, full_max, full_top3, full_std, text_mean, text_max,
        text_top3, profile_cosine, np.abs(difference).mean(axis=-1), np.linalg.norm(difference, axis=-1),
        category_mean, category_max, last_similarity,
        np.broadcast_to(np.log1p(count)[:, None], shape),
        np.broadcast_to((count == 0)[:, None], shape)), axis=-1)
    return output.astype(np.float32, copy=False)


class FeatureBuilder:
    """Same feature calculation for training, evaluation and service requests."""

    def __init__(self, store, temporal, variant="full"):
        self.store, self.temporal, self.variant = store, temporal, variant
        self.feature_indices = feature_indices(variant)
        self.feature_names = tuple(FULL_FEATURE_NAMES[i] for i in self.feature_indices)
        self.text_dimensions = int(store.manifest["features"]["text_dimensions"])
        self._priors = {}
        self.bindings = {
            "data_manifest_sha256": sha256(store.path / "manifest.json"),
            "news_features_sha256": sha256(store.path / "news_features.npy"),
            "news_first_seen_sha256": sha256(store.path / "news_first_seen.npy"),
            "training_cache_sha256": sha256(store.path / "training_cache.json"),
            "temporal_manifest_sha256": sha256(temporal.path / "manifest.json"),
            "temporal_policy_hash": temporal.policy_hash,
            "temporal_arrays": temporal.manifest["arrays"],
            "source_sha256": sha256(__file__),
        }
        if (temporal.manifest["data_manifest_sha256"] != self.bindings["data_manifest_sha256"]
                or temporal.version != self.bindings["temporal_manifest_sha256"]):
            raise ValueError("temporal snapshots and processed data do not match")

    def _ids(self, values, ndim):
        values = np.asarray(values)
        if values.ndim != ndim or (values.size and not np.issubdtype(values.dtype, np.integer)):
            raise ValueError("news IDs must be an integer array of the requested shape")
        values = values.astype(np.int64, copy=False)
        if np.any(values < 0) or np.any(values >= len(self.store.features)):
            raise ValueError("news ID outside the catalogue")
        return values

    def _context(self, candidates, timestamps):
        output = np.empty((*candidates.shape, 7), dtype=np.float32)
        hours = timestamps // 3600
        for hour in np.unique(hours):
            positions = np.flatnonzero(hours == hour)
            snapshot = self.temporal.snapshot(int(timestamps[positions[0]]))
            if snapshot.as_of_start != int(hour) * 3600 or snapshot.cutoff != snapshot.as_of_start - 3600:
                raise ValueError("snapshot violates the declared hourly one-hour feedback delay")
            key = (int(snapshot.index), int(snapshot.cutoff))
            if key not in self._priors:
                click_sum = float(snapshot.clicks.sum(dtype=np.float64))
                exposure_sum = float(snapshot.exposures.sum(dtype=np.float64))
                if not np.isfinite([click_sum, exposure_sum]).all() or not 0 <= click_sum <= exposure_sum + 1e-5:
                    raise ValueError("invalid EMA totals")
                self._priors[key] = click_sum / max(exposure_sum, 1.)
            prior = self._priors[key]
            ids = candidates[positions]
            first_seen = self.store.first_seen[ids]
            known = (ids > 0) & (first_seen < snapshot.cutoff)
            clicks = np.where(known, snapshot.clicks[ids], 0).astype(np.float64)
            exposures = np.where(known, snapshot.exposures[ids], 0).astype(np.float64)
            if (not np.isfinite(clicks).all() or not np.isfinite(exposures).all()
                    or np.any(clicks < 0) or np.any(exposures < 0)
                    or np.any(clicks > exposures + 1e-5)):
                raise ValueError("invalid item EMA values")
            # Substitute before subtracting to avoid MAX_INT PAD/unknown overflow.
            observed = np.where(known, first_seen, snapshot.as_of_start)
            age = np.maximum((snapshot.as_of_start - observed) / 3600., 0)
            output[positions] = np.stack((np.log1p(clicks), np.log1p(exposures),
                (clicks + 20. * prior) / (exposures + 20.), np.log1p(age),
                known.astype(np.float64), (exposures == 0).astype(np.float64),
                np.full(ids.shape, prior)), axis=-1)
        output[candidates == 0] = 0
        return output

    def full_features(self, history_ids, candidate_ids, as_of):
        histories, candidates = self._ids(history_ids, 2), self._ids(candidate_ids, 2)
        times = np.asarray(as_of)
        if (times.ndim != 1 or not np.issubdtype(times.dtype, np.integer)
                or histories.shape[0] != candidates.shape[0] or len(times) != len(histories)):
            raise ValueError("batch/time shape mismatch")
        content = content_features(self.store.features[histories], self.store.features[candidates],
            histories > 0, text_dimensions=self.text_dimensions)
        result = np.concatenate((content, self._context(candidates, times)), axis=-1)
        result[candidates == 0] = 0
        return result

    def row_features(self, store, rows, candidate_ids):
        if store is not self.store:
            raise ValueError("feature builder belongs to a different Store")
        rows = np.asarray(rows, dtype=np.int64)
        return self.full_features(store.history[rows], candidate_ids, store.timestamps[rows])[..., self.feature_indices]

    def candidate_features(self, history_ids, candidate_ids, as_of):
        history = self._ids(history_ids, 1)
        candidates = self._ids(candidate_ids, 1)
        full = self.full_features(history[None], candidates[None], np.array([as_of], dtype=np.int64))[0]
        return full[:, self.feature_indices]


class CausalRanker:
    def __init__(self, model, feature_builder, metadata=None, manifest=None):
        self.model, self.feature_builder = model, feature_builder
        self.variant = feature_builder.variant
        self.feature_indices = feature_builder.feature_indices
        self.feature_names = feature_builder.feature_names
        self.metadata = metadata or {}
        self.manifest = manifest or {}
        self.model_version = self.manifest.get("model_sha256")

    def predict_full_features(self, full_matrix_22):
        values = np.asarray(full_matrix_22)
        if values.ndim < 2 or values.shape[-1] != 22 or not np.isfinite(values).all():
            raise ValueError("expected finite features [...,22]")
        shape = values.shape[:-1]
        if not values.size:
            return np.empty(shape, dtype=np.float64)
        selected = values[..., slice(self.feature_indices[0], self.feature_indices[-1] + 1)]
        return self.model.predict_proba(selected.reshape(-1, len(self.feature_names)))[:, 1].reshape(shape)

    def score(self, history_ids, candidate_ids, as_of):
        history = self.feature_builder._ids(history_ids, 1)
        candidates = self.feature_builder._ids(candidate_ids, 1)
        if not len(candidates):
            return np.empty(0, dtype=np.float64)
        features = self.feature_builder.full_features(history[None], candidates[None], np.array([as_of], dtype=np.int64))
        return self.predict_full_features(features)[0]

    def save(self, path):
        path = Path(path)
        sidecar = Path(str(path) + ".manifest.json")
        if path.exists() or sidecar.exists():
            raise FileExistsError("refusing to overwrite a ranker artifact")
        path.parent.mkdir(parents=True, exist_ok=True)
        common = {"feature_version": FEATURE_VERSION, "variant": self.variant,
            "full_feature_names": list(FULL_FEATURE_NAMES), "feature_names": list(self.feature_names),
            "feature_schema_sha256": json_hash(FULL_FEATURE_NAMES),
            "bindings": self.feature_builder.bindings, "metadata": self.metadata,
            "score_semantics": "sampled 1:4 binary ranking score; not calibrated population CTR"}
        with path.open("xb") as stream:
            joblib.dump({"model": self.model, **common}, stream)
        self.manifest = {**common, "model_sha256": sha256(path)}
        with sidecar.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(self.manifest, indent=2, allow_nan=False))
        self.model_version = self.manifest["model_sha256"]

    @classmethod
    def load(cls, path, store, temporal):
        path = Path(path)
        manifest = read_json(str(path) + ".manifest.json")
        if manifest["model_sha256"] != sha256(path):
            raise ValueError("ranker model SHA mismatch")
        builder = FeatureBuilder(store, temporal, manifest["variant"])
        if (manifest["bindings"] != builder.bindings or manifest["feature_version"] != FEATURE_VERSION
                or manifest["feature_schema_sha256"] != json_hash(FULL_FEATURE_NAMES)
                or manifest["full_feature_names"] != list(FULL_FEATURE_NAMES)
                or manifest["feature_names"] != list(builder.feature_names)):
            raise ValueError("ranker data, snapshot, policy or feature binding mismatch")
        artifact = joblib.load(path)
        if any(artifact[key] != value for key, value in manifest.items() if key != "model_sha256"):
            raise ValueError("ranker manifest differs from artifact metadata")
        if artifact["model"].n_features_in_ != len(builder.feature_names):
            raise ValueError("ranker estimator feature count mismatch")
        return cls(artifact["model"], builder, artifact["metadata"], manifest)


@dataclass
class TrainingSamples:
    rows: np.ndarray
    candidate_ids: np.ndarray
    labels: np.ndarray
    metadata: dict


def training_samples(store, max_positives=100000):
    if not (store.path / "training_cache.json").is_file():
        raise ValueError("existing v1 training cache required; v2 never builds one")
    store.ensure_training_cache()  # Existing-marker branch only: reads mmap arrays.
    total = len(store.positive_rows)
    if total < max_positives or max_positives < 1:
        raise ValueError("declared positive budget unavailable; refusing to silently shrink it")
    indices = np.sort(np.random.default_rng(SAMPLING_SEED).choice(total, max_positives, replace=False))
    rows = np.asarray(store.positive_rows[indices], dtype=np.int64)
    positives = np.asarray(store.positive_items[indices], dtype=np.int64)
    if np.any(store.split[rows] != 0):
        raise ValueError("training cache contains held-out rows")
    pool = np.asarray(store.pool[rows])
    if pool.ndim != 2 or not pool.shape[1] or np.any(pool <= 0):
        raise ValueError("invalid same-exposure negative pool")
    columns = np.random.default_rng(SAMPLING_SEED).integers(pool.shape[1], size=(len(rows), 4))
    negatives = np.take_along_axis(pool, columns, axis=1)
    if np.any(positives <= 0) or np.any(negatives == positives[:, None]):
        raise ValueError("negative pool overlaps positives or PAD")
    candidates = np.concatenate((positives[:, None], negatives), axis=1)
    labels = np.tile(np.array([1, 0, 0, 0, 0], dtype=np.uint8), len(rows))
    metadata = {"positive_pairs": len(rows), "binary_examples": len(labels), "sampling_seed": SAMPLING_SEED,
        "positive_indices_sha256": array_hash(indices.astype("<i8")),
        "training_rows_sha256": array_hash(rows.astype("<i8")),
        "candidate_ids_sha256": array_hash(candidates.astype("<i8")), "labels_sha256": array_hash(labels),
        "negative_policy": "four uniform replacement draws from fixed same-exposure nonclick pool",
        "sampling_bias": "positive-pair sampling overweights multi-click impressions; 1:4 scores are not population CTR",
        "source_training_cache": read_json(store.path / "training_cache.json")}
    return TrainingSamples(rows, candidates, labels, metadata)


def memory_preflight(examples, columns=22):
    available = psutil.virtual_memory().available
    matrix = int(examples) * columns * 8
    bins = int(examples) * columns
    reserve = 896 * 2**20
    record = {"available_bytes": available, "float64_memmap_bytes": matrix,
        "estimated_binned_bytes": bins, "working_and_system_reserve_bytes": reserve,
        "required_available_bytes": matrix + bins + reserve,
        "allowed": available >= matrix + bins + reserve}
    if not record["allowed"]:
        raise MemoryError(f"insufficient memory for declared budget: {json.dumps(record)}")
    return record


class MemoryMonitor:
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.minimum_available = psutil.virtual_memory().available
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(.1):
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()

    def summary(self):
        return {"pid": self.process.pid, "max_observed_rss_bytes": self.peak,
            "minimum_system_available_bytes": self.minimum_available, "sampling_interval_seconds": .1,
            "scope": "actual process RSS including mapped pages; sampled peak only"}


def build_training_matrix(builder, samples, path, batch_size=256):
    if not 1 <= batch_size <= 256 or builder.variant != "full":
        raise ValueError("full 22-feature builder and batch size 1..256 required")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    matrix = open_memmap(path, mode="w+", dtype=np.float64,
                         shape=(len(samples.labels), 22), fortran_order=True)
    # Process nearby hours together but write back to the unchanged sampled order.
    order = np.argsort(builder.store.timestamps[samples.rows] // 3600, kind="stable")
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        values = builder.row_features(builder.store, samples.rows[indices], samples.candidate_ids[indices])
        positions = (indices[:, None] * 5 + np.arange(5)[None]).ravel()
        matrix[positions] = values.reshape(-1, 22)
    matrix.flush()
    return matrix


def validation_rows(store, limit=6000):
    rows = np.flatnonzero((store.split == 1) & (store.timestamps >= INTERNAL_MIN_TIMESTAMP))
    if len(rows) < limit or limit < 1:
        raise ValueError("declared internal validation budget unavailable")
    return np.sort(np.random.default_rng(SAMPLING_SEED).choice(rows, limit, replace=False))


def build_validation_cache(builder, rows, path, batch_size=32):
    rows = np.asarray(rows, dtype=np.int64)
    if builder.variant != "full" or np.any(builder.store.split[rows] != 1):
        raise ValueError("validation cache is internal-only and uses all 22 features")
    lengths = np.asarray(builder.store.offsets[rows + 1] - builder.store.offsets[rows], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    if Path(path).exists():
        raise FileExistsError(path)
    features = open_memmap(path, mode="w+", dtype=np.float32, shape=(int(offsets[-1]), 22))
    hours = builder.store.timestamps[rows] // 3600
    order = np.lexsort((lengths, hours))
    for start in range(0, len(order), batch_size):
        positions = order[start:start + batch_size]
        batch = rows[positions]
        ids, mask = builder.store.padded_candidates(batch)
        values = builder.row_features(builder.store, batch, ids)
        for local, position in enumerate(positions):
            features[offsets[position]:offsets[position + 1]] = values[local, mask[local]]
    features.flush()
    return features, offsets


def evaluate_internal(ranker, store, rows, full_features, offsets, output=None):
    rows = np.asarray(rows, dtype=np.int64)
    if (np.any(store.split[rows] != 1) or np.any(store.timestamps[rows] < INTERNAL_MIN_TIMESTAMP)
            or len(offsets) != len(rows) + 1 or offsets[-1] != len(full_features)):
        raise ValueError("only the declared internal validation population is allowed")
    scores = np.empty(len(full_features), dtype=np.float64)
    for start in range(0, len(scores), 4096):
        scores[start:start + 4096] = ranker.predict_full_features(full_features[start:start + 4096])
    values = np.empty((len(rows), 4), dtype=np.float64)
    for position, row in enumerate(rows):
        _, labels = store.candidates_for(row)
        values[position] = impression_metrics(labels, scores[offsets[position]:offsets[position + 1]])
    if output is not None:
        np.save(output, values)
    return {"impressions": len(rows), "candidate_exposures": int(offsets[-1]),
        "row_indices_sha256": array_hash(rows.astype("<i8")), "metrics": summarize(values),
        "metric_columns": list(METRICS), "all_logged_candidates": True,
        "official_dev_evaluated": False}, values


def select_models(records):
    expected = {f"{variant}-seed{seed}" for variant in VARIANTS for seed in SEEDS}
    if set(records) != expected:
        raise ValueError("selection requires all three feature variants and all three seeds")
    first = records["content_only-seed17"]
    for run, record in records.items():
        if (record["sampling"] != first["sampling"] or record["bindings"] != first["bindings"]
                or record["variant"] not in VARIANTS or record["seed"] not in SEEDS
                or record["config"] != {**MODEL_CONFIG, "random_state": record["seed"]}
                or record["internal_validation"]["row_indices_sha256"] != first["internal_validation"]["row_indices_sha256"]
                or record["internal_validation"]["candidate_exposures"] != first["internal_validation"]["candidate_exposures"]
                or not np.isfinite(record["internal_validation"]["metrics"]["ndcg10"]["mean"])):
            raise ValueError(f"ablation budgets/populations differ: {run}")
    def quality(run):
        return records[run]["internal_validation"]["metrics"]["ndcg10"]["mean"]
    best = {variant: max((f"{variant}-seed{seed}" for seed in SEEDS), key=quality) for variant in VARIANTS}
    winner = max(VARIANTS, key=lambda variant: quality(best[variant]))
    best_records = {variant: {"run_id": run, "variant": variant, "seed": records[run]["seed"],
        "model_path": records[run]["model_path"], "model_sha256": records[run]["model_sha256"],
        "internal_ndcg10": quality(run)} for variant, run in best.items()}
    return {"criterion": "internal impression-macro ndcg10", "tie_break": "variant content_only/context_only/full, then seed ascending",
        "models": [{"run_id": run, "variant": records[run]["variant"], "seed": records[run]["seed"],
            "model_path": records[run]["model_path"], "model_sha256": records[run]["model_sha256"],
            "metrics": records[run]["internal_validation"]["metrics"]}
            for variant in VARIANTS for run in (f"{variant}-seed{seed}" for seed in SEEDS)],
        "best_by_variant": best_records, "selected": best_records[winner],
        "best_variant": winner, "best_run_id": best[winner],
        "sampling": first["sampling"], "validation_row_indices_sha256": first["internal_validation"]["row_indices_sha256"],
        "official_dev_used_for_selection": False,
        "study_status": "v2 retrospective benchmark: v1 official-dev results were already observed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true", help="default: fit and select using internal validation only")
    mode.add_argument("--validation-only", action="store_true", help="load existing models; no fitting")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--temporal", type=Path, default=Path("data/temporal_v2"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/v2/ranker"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v2/ranker"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--max-positives", type=int, choices=(100000,), default=100000)
    parser.add_argument("--validation-impressions", type=int, choices=(6000,), default=6000)
    parser.add_argument("--threads", type=int, choices=(1,), default=1)
    parser.add_argument("--resume", action="store_true", help="reuse only matching completed artifacts")
    args = parser.parse_args()
    # Check before importing sklearn or opening feature matrices. No budget shrink.
    preflight = memory_preflight(args.max_positives * 5)
    from .temporal import TemporalStore
    from sklearn.ensemble import HistGradientBoostingClassifier
    store = Store(args.data)
    temporal = TemporalStore(args.temporal, processed_dir=args.data)
    full_builder = FeatureBuilder(store, temporal)
    samples = training_samples(store, args.max_positives)
    rows = validation_rows(store, args.validation_impressions)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    matrix_path = args.artifacts / ".training_features.npy"
    validation_path = args.artifacts / ".validation_features.npy"
    matrix = validation = None
    try:
        with threadpool_limits(1), MemoryMonitor() as memory:
            started = time.perf_counter()
            if not args.validation_only:
                matrix = build_training_matrix(full_builder, samples, matrix_path)
            construction_seconds = time.perf_counter() - started
            validation, offsets = build_validation_cache(full_builder, rows, validation_path)
            np.save(args.artifacts / "validation_rows.npy", rows)
            records = {}
            for variant in args.variants:
                builder = FeatureBuilder(store, temporal, variant)
                indices = builder.feature_indices
                for seed in args.seeds:
                    run = f"{variant}-seed{seed}"
                    path = args.artifacts / f"{run}.joblib"
                    if args.validation_only or (args.resume and path.exists()):
                        ranker = CausalRanker.load(path, store, temporal)
                        if ranker.metadata["sampling"] != samples.metadata or ranker.metadata["seed"] != seed:
                            raise ValueError("existing artifact belongs to a different sample/seed")
                        fit_seconds = ranker.metadata["fit_seconds"]
                    else:
                        if path.exists():
                            raise FileExistsError(f"{path}; use --resume for matching completed models")
                        fit_started = time.perf_counter()
                        model = HistGradientBoostingClassifier(**MODEL_CONFIG, random_state=seed)
                        # Consecutive Fortran-order columns remain a float64 view.
                        model.fit(matrix[:, indices[0]:indices[-1] + 1], samples.labels)
                        fit_seconds = time.perf_counter() - fit_started
                        ranker = CausalRanker(model, builder, {"run_id": run, "seed": seed,
                            "config": {**MODEL_CONFIG, "random_state": seed}, "sampling": samples.metadata,
                            "fit_seconds": fit_seconds, "threads": 1})
                        ranker.save(path)
                    evaluation, _ = evaluate_internal(ranker, store, rows, validation, offsets,
                        args.artifacts / f"metrics_{run}_validation.npy")
                    record = {"run_id": run, "variant": variant, "seed": seed,
                        "feature_names": list(builder.feature_names), "feature_count": len(indices),
                        "model_path": str(path).replace("\\", "/"),
                        "model_sha256": ranker.model_version, "bindings": builder.bindings,
                        "config": {**MODEL_CONFIG, "random_state": seed}, "sampling": samples.metadata,
                        "fit_seconds": fit_seconds, "internal_validation": evaluation, "threads": 1,
                        "official_dev_evaluated": False}
                    write_json(args.reports / f"{run}_training.json", record)
                    records[run] = record
                    print(json.dumps({"run_id": run, "validation": evaluation["metrics"], "fit_seconds": fit_seconds}), flush=True)
            for variant in VARIANTS:
                for seed in SEEDS:
                    run = f"{variant}-seed{seed}"
                    path = args.reports / f"{run}_training.json"
                    if run not in records and path.exists():
                        record = read_json(path)
                        if record["bindings"] != full_builder.bindings or record["model_sha256"] != sha256(args.artifacts / f"{run}.joblib"):
                            raise ValueError("existing ablation report is stale")
                        records[run] = record
            if len(records) == len(VARIANTS) * len(SEEDS):
                write_json(args.reports / "selection.json", select_models(records))
            write_json(args.reports / "resource_report.json", {"preflight": preflight,
                "observed": memory.summary(), "feature_construction_seconds": construction_seconds,
                "training_matrix_bytes": 0 if matrix is None else matrix.nbytes,
                "validation_feature_bytes": validation.nbytes, "source_sha256": sha256(__file__),
                "official_dev_evaluated": False})
    finally:
        for array, path in ((matrix, matrix_path), (validation, validation_path)):
            if array is not None:
                array._mmap.close()
                path.unlink()


if __name__ == "__main__":
    main()
