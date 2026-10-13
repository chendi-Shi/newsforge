"""A deterministic, bounded pairwise residual above a fixed popularity order.

Only the supplied fit pairs estimate normalization or coefficients. Every pair
has equal loss weight; callers own exposure sampling and temporal eligibility.
The score is a ranking adjustment, never a calibrated click probability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


FEATURE_NAMES = (
    "history_cosine_mean", "history_cosine_max", "history_cosine_top3_mean",
    "history_cosine_std", "text_cosine_mean", "text_cosine_max",
    "text_cosine_top3_mean", "mean_profile_text_cosine", "mean_profile_text_l1",
    "mean_profile_text_l2", "category_overlap_mean", "category_overlap_max",
    "last_valid_history_text_cosine",
)
VARIANTS = {
    "full13": tuple(range(13)),
    "no_category": tuple(i for i in range(13) if i not in (10, 11)),
    "no_recent": tuple(range(12)),
}
FORMAT_VERSION = 1
FEATURE_VERSION = "v2_content_first13_v3_pairwise_residual_1"
L2 = .01
MAX_ITERATIONS = 200


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value):
    # A serialized copy prevents a caller from silently mutating a binding.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _feature_matrix(value, *, nonempty=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] != 13:
        raise ValueError("features must be a real numeric [rows, 13] matrix")
    values = np.asarray(raw, dtype=np.float64)
    if (nonempty and len(values) == 0) or not np.isfinite(values).all():
        raise ValueError("features must be finite, with nonempty fit pairs")
    return values


def _loss_gradient(weights, differences):
    """Mean pairwise logistic loss + .5 * L2 * ||w||^2 and its gradient."""
    margins = differences @ weights
    loss = np.logaddexp(0., -margins).mean() + .5 * L2 * np.dot(weights, weights)
    gradient = -(differences.T @ expit(-margins)) / len(differences) + L2 * weights
    return float(loss), gradient


class PairwiseResidual:
    """Linear pairwise preference model with fit-only feature normalization."""

    def __init__(self, variant="full13"):
        if not isinstance(variant, str) or variant not in VARIANTS:
            raise ValueError("unknown residual feature variant")
        self.variant = variant
        self.feature_indices = VARIANTS[variant]
        self.feature_names = tuple(FEATURE_NAMES[i] for i in self.feature_indices)
        self.bindings = {}
        self.fit_metadata = {}
        self.mean = self.scale = self.weights = None

    def _require_fitted(self):
        if self.weights is None:
            raise ValueError("residual model has not been fitted")

    def fit(self, positive_features, negative_features, *, bindings=None):
        """Fit matching [N,13] positive/negative pairs, with equal pair weight.

        The loss has no intercept: a query-constant intercept cancels between
        paired items. Normalization is estimated from both sides of these fit
        pairs only. Repeated items keep their sampling multiplicity.
        """
        if self.weights is not None:
            raise ValueError("create a fresh residual model instead of refitting")
        positive = _feature_matrix(positive_features, nonempty=True)
        negative = _feature_matrix(negative_features, nonempty=True)
        if positive.shape != negative.shape:
            raise ValueError("positive and negative feature shapes must match")
        if bindings is not None and not isinstance(bindings, dict):
            raise ValueError("training bindings must be a JSON object")
        fit_bindings = _json_copy(bindings or {})
        positive, negative = positive[:, self.feature_indices], negative[:, self.feature_indices]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean = positive.mean(axis=0) * .5 + negative.mean(axis=0) * .5
            variance = ((positive - mean) ** 2).mean(axis=0) * .5
            variance += ((negative - mean) ** 2).mean(axis=0) * .5
            scale = np.sqrt(variance)
            scale = np.where(scale > 0., scale, 1.)
            differences = (positive - negative) / scale
        if not all(np.isfinite(value).all() for value in (mean, scale, differences)):
            raise ValueError("fit normalization overflowed or produced nonfinite values")
        result = minimize(_loss_gradient, np.zeros(len(self.feature_indices)),
                          args=(differences,), method="L-BFGS-B", jac=True,
                          options={"maxiter": MAX_ITERATIONS, "ftol": 1e-12, "gtol": 1e-8})
        if not result.success:
            raise RuntimeError(f"residual optimizer failed (status={result.status}): {result.message}")
        weights = np.asarray(result.x, dtype=np.float64)
        loss, gradient = _loss_gradient(weights, differences)
        if (weights.shape != mean.shape or not np.isfinite(weights).all()
                or not np.isfinite(loss) or not np.isfinite(gradient).all()):
            raise RuntimeError("residual optimizer returned invalid coefficients or objective")
        self.mean, self.scale, self.weights = mean.copy(), scale.copy(), weights.copy()
        self.bindings = fit_bindings
        self.fit_metadata = {
            "pairs": len(positive), "pair_weighting": "equal weight per supplied pair",
            "normalization": "population mean/SD of the supplied positive and negative fit rows; constant scale=1",
            "objective": "mean(logaddexp(0,-((positive-negative)/scale)@w)) + 0.5*l2*sum(w*w)",
            "l2": L2, "max_iterations": MAX_ITERATIONS, "optimizer": "L-BFGS-B",
            "optimizer_success": True, "optimizer_status": int(result.status),
            "optimizer_message": str(result.message), "iterations": int(result.nit),
            "function_evaluations": int(result.nfev), "loss": loss,
            "gradient_max_absolute": float(np.max(np.abs(gradient), initial=0.)),
            "model_randomness": "none; randomness belongs to caller's pair sampling",
        }
        return self

    def predict_raw(self, features):
        """Return standardized-feature linear scores without clipping or gating."""
        self._require_fitted()
        values = _feature_matrix(features)[:, self.feature_indices]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw = ((values - self.mean) / self.scale) @ self.weights
        if not np.isfinite(raw).all():
            raise ValueError("prediction produced nonfinite residuals")
        return raw

    def score(self, features, *, history_length, gate=True):
        """Clip to [-1,1], optionally gate by n/(n+5); an empty history is zero."""
        if (not isinstance(history_length, (int, np.integer))
                or isinstance(history_length, (bool, np.bool_)) or history_length < 0):
            raise ValueError("history_length must be a nonnegative integer")
        if not isinstance(gate, (bool, np.bool_)):
            raise ValueError("gate must be boolean")
        values = np.clip(self.predict_raw(features), -1., 1.)
        if history_length == 0:
            return np.zeros_like(values)
        if gate:
            values *= float(history_length) / (float(history_length) + 5.)
        return values

    def rank(self, features, *, history_length, delta, gate=True):
        """Return (order, scores), with scores indexed by the original item rank.

        A later item cannot overtake an earlier item whose original rank is
        at least 2*delta away: clipped residual differences are at most two,
        and exact score ties keep the original order. This bounds pairwise
        inversion distance, not every item's final rank displacement.
        """
        if (not isinstance(delta, (int, float, np.integer, np.floating))
                or isinstance(delta, (bool, np.bool_)) or not np.isfinite(delta) or delta < 0):
            raise ValueError("delta must be a finite nonnegative number")
        residual = self.score(features, history_length=history_length, gate=gate)
        original_rank = np.arange(len(residual), dtype=np.int64)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = -original_rank.astype(np.float64) + float(delta) * residual
        if not np.isfinite(scores).all():
            raise ValueError("rank scores must be finite")
        order = np.lexsort((original_rank, -scores))
        return order, scores

    def save(self, path):
        """Save non-pickle arrays and a hash-bound JSON sidecar; never overwrite."""
        self._require_fitted()
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("residual model path must end in .npz")
        sidecar = path.with_suffix(path.suffix + ".json")
        if path.exists() or sidecar.exists():
            raise FileExistsError("residual model or sidecar already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez(stream, mean=self.mean, scale=self.scale, weights=self.weights)
        metadata = {
            "format_version": FORMAT_VERSION, "feature_version": FEATURE_VERSION,
            "variant": self.variant, "input_feature_names": list(FEATURE_NAMES),
            "feature_indices": list(self.feature_indices), "feature_names": list(self.feature_names),
            "npz_sha256": _sha256(path), "source_sha256": _sha256(__file__),
            "fit_metadata": self.fit_metadata, "bindings": self.bindings,
            "score_semantics": "bounded personal ranking residual; not calibrated CTR",
        }
        with sidecar.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))
        return metadata

    @classmethod
    def load(cls, path, *, expected_bindings=None):
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("residual model path must end in .npz")
        sidecar = path.with_suffix(path.suffix + ".json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if (not isinstance(metadata, dict) or metadata.get("format_version") != FORMAT_VERSION
                or metadata.get("feature_version") != FEATURE_VERSION
                or metadata.get("input_feature_names") != list(FEATURE_NAMES)):
            raise ValueError("residual model feature schema or format mismatch")
        model = cls(metadata.get("variant"))
        if (metadata.get("feature_indices") != list(model.feature_indices)
                or metadata.get("feature_names") != list(model.feature_names)):
            raise ValueError("residual model feature variant mismatch")
        if metadata.get("npz_sha256") != _sha256(path):
            raise ValueError("residual model array SHA256 mismatch")
        if metadata.get("source_sha256") != _sha256(__file__):
            raise ValueError("residual model implementation SHA256 mismatch")
        bindings = metadata.get("bindings")
        if not isinstance(bindings, dict):
            raise ValueError("residual model lacks training bindings")
        if expected_bindings is not None:
            if not isinstance(expected_bindings, dict) or bindings != _json_copy(expected_bindings):
                raise ValueError("residual model training bindings mismatch")
        fit_metadata = metadata.get("fit_metadata")
        if (not isinstance(fit_metadata, dict) or fit_metadata.get("optimizer_success") is not True
                or fit_metadata.get("l2") != L2 or fit_metadata.get("max_iterations") != MAX_ITERATIONS
                or fit_metadata.get("pair_weighting") != "equal weight per supplied pair"
                or not isinstance(fit_metadata.get("pairs"), int) or fit_metadata["pairs"] < 1):
            raise ValueError("residual model fit metadata mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"mean", "scale", "weights"}:
                raise ValueError("unexpected residual model arrays")
            values = {name: np.asarray(arrays[name]) for name in arrays.files}
            if any(value.dtype != np.dtype("float64") or value.shape != (len(model.feature_indices),)
                   or not np.isfinite(value).all() for value in values.values()):
                raise ValueError("invalid residual model array shape, dtype or values")
            if np.any(values["scale"] <= 0):
                raise ValueError("residual model scales must be positive")
            model.mean, model.scale, model.weights = (values[key].copy() for key in ("mean", "scale", "weights"))
        model.fit_metadata, model.bindings = _json_copy(fit_metadata), _json_copy(bindings)
        return model
