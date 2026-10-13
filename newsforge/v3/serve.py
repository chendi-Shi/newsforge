"""CPU service for the confirmed v3 residual or its exact popularity fallback."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from newsforge.v2.ranker import content_features
from newsforge.v2.retrieval import CHANNEL_BITS, canonical_hash, sha256_file
from newsforge.v2.serve import (
    V2Engine, V2ServiceConfig, V2RecommendRequest, V2BatchRequest,
    create_app as create_v2_app, verify_generator_selection,
)
from newsforge.v2.temporal import TemporalStore
from .fast_retrieval import FastCandidateGenerator
from .residual import PairwiseResidual


SEEDS = (17, 29, 43)
DEPLOYMENT_AVAILABLE_AS_OF = 1573776000


def runtime_sources():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "v3" / name for name in ("serve.py", "residual.py", "fast_retrieval.py")]
    paths += [root / "v2" / name for name in ("ranker.py", "retrieval.py", "serve.py", "temporal.py")]
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def rank_ensemble(models, features, *, history_length, delta):
    """Clip each seed before averaging; gate once; exact ties keep input rank."""
    if (not isinstance(history_length, (int, np.integer)) or isinstance(history_length, (bool, np.bool_))
            or history_length < 0):
        raise ValueError("history_length must be a nonnegative integer")
    if (not isinstance(delta, (int, float, np.integer, np.floating)) or isinstance(delta, (bool, np.bool_))
            or not np.isfinite(delta) or delta < 0):
        raise ValueError("delta must be a finite nonnegative number")
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 13 or not np.isfinite(features).all():
        raise ValueError("residual features must be finite [candidates,13]")
    original = np.arange(len(features), dtype=np.int64)
    scores = -original.astype(np.float64)
    if history_length and delta and len(features):
        if len(models) != 3:
            raise ValueError("the fixed residual ensemble requires exactly three seeds")
        clipped = np.asarray([model.score(features, history_length=int(history_length), gate=False)
                              for model in models], dtype=np.float64)
        if clipped.shape != (3, len(features)) or not np.isfinite(clipped).all() or np.any(np.abs(clipped) > 1.):
            raise ValueError("residual ensemble returned invalid clipped scores")
        residual = clipped.mean(axis=0)
        gate = float(history_length) / (float(history_length) + 5.)
        scores += float(delta) * gate * residual
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite residual rank score")
    return np.lexsort((original, -scores)), scores


def _model_content_hash(model):
    digest = hashlib.sha256()
    digest.update(model.variant.encode())
    for array in (model.mean, model.scale, model.weights):
        values = np.asarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


class V3Engine(V2Engine):
    def __init__(self, generator, news_ids, config, *, models=(), delta=0.,
                 deployment_gate_passed=False, text_dimensions=96,
                 evidence_bindings=None, deployment_available_as_of=DEPLOYMENT_AVAILABLE_AS_OF):
        if config.use_reranker or config.candidate_config.channels != ("popular",):
            raise ValueError("v3 requires the frozen popularity-only candidate recipe without the v2 ranker")
        if not isinstance(deployment_gate_passed, bool):
            raise ValueError("deployment gate must be a boolean")
        if (not isinstance(delta, (int, float)) or isinstance(delta, bool)
                or not np.isfinite(delta) or delta < 0):
            raise ValueError("selected delta must be finite and nonnegative")
        if not isinstance(text_dimensions, int) or isinstance(text_dimensions, bool) or not 0 < text_dimensions < generator.features.shape[1]:
            raise ValueError("text_dimensions must leave category features")
        if (not isinstance(deployment_available_as_of, int) or isinstance(deployment_available_as_of, bool)
                or deployment_available_as_of < 0):
            raise ValueError("deployment availability must be a nonnegative integer timestamp")
        self.delta, self.deployment_gate_passed = float(delta), deployment_gate_passed
        self.residual_enabled = deployment_gate_passed and delta > 0
        self.models = tuple(models)
        if self.residual_enabled and (len(self.models) != 3 or any(model.variant != "full13" for model in self.models)):
            raise ValueError("confirmed deployment needs the three full13 residual models")
        for model in self.models:
            for array in (model.mean, model.scale, model.weights):
                if array is None:
                    raise ValueError("residual model must be fitted")
                array.setflags(write=False)
        self.text_dimensions = text_dimensions
        super().__init__(generator, news_ids, config)
        if self.residual_enabled:
            self.minimum_as_of = max(self.minimum_as_of, deployment_available_as_of)
            try:
                self._validate_as_of(self.default_as_of)
            except BaseException:
                self._thread_limiter.restore_original_limits()
                raise
        self.bindings = {**self.bindings, "api_version": "v3", "v3_runtime_sources": runtime_sources(),
                         "v2_ranker_enabled": False, "reranker_enabled": self.residual_enabled,
                         "v3_evidence": json.loads(json.dumps(evidence_bindings or {}, allow_nan=False)),
                         "deployment_gate_passed": deployment_gate_passed,
                         "residual_enabled": self.residual_enabled, "delta": self.delta,
                         "text_dimensions": text_dimensions,
                         "deployment_available_as_of": deployment_available_as_of,
                         "residual_model_content_sha256": [_model_content_hash(model) for model in self.models],
                         "residual_ensemble": "mean of three seed-wise clips; n/(n+5) gate; stable original-rank tie"}
        self.model_version = canonical_hash(self.bindings)

    def health(self):
        health = super().health()
        health.update({"api_version": "v3", "reranker_enabled": self.residual_enabled,
                       "deployment_gate_passed": self.deployment_gate_passed,
                       "selected_delta": self.delta, "residual_seed_count": len(self.models),
                       "residual_score_type": "bounded preference adjustment, not a click probability",
                       "fast_retrieval_cache": self.generator.cache_info()})
        return health

    def _recommend(self, request):
        as_of = self.default_as_of if request.as_of is None else int(request.as_of)
        self._validate_as_of(as_of)
        history = []
        for item in request.history:
            index = self.id_to_index.get(item)
            if index is None:
                raise ValueError("history contains PAD/OOV or unknown article IDs")
            history.append(index)
        if request.top_k > self.config.candidate_config.pool_size:
            raise ValueError("top_k exceeds the selected candidate pool size")
        candidates = self.generator.retrieve(history, as_of, config=self.config.candidate_config)
        order, scores = np.arange(len(candidates.ids)), None
        history_length = candidates.metadata["history_length_used"]
        residual_applied = bool(self.residual_enabled and history_length and len(candidates.ids))
        if residual_applied:
            hf = np.asarray(self.generator.features[history[-20:]], dtype=np.float32)
            candidate_features = np.asarray(self.generator.features[candidates.ids], dtype=np.float32)
            features = content_features(hf[None], candidate_features[None],
                                        text_dimensions=self.text_dimensions)[0, :, :13]
            order, scores = rank_ensemble(self.models, features, history_length=history_length, delta=self.delta)
        items = []
        for position in order[:request.top_k]:
            item = int(candidates.ids[position])
            source_mask = int(candidates.source_mask[position])
            items.append({"item_id": self.news_ids[item],
                          "retrieval_score": float(candidates.scores[position]),
                          "rerank_score": None if scores is None else float(scores[position]),
                          "sources": [name for name, bit in CHANNEL_BITS.items() if source_mask & bit]})
        return {"items": items, "model_version": self.model_version,
                "candidate_config_version": self.config.candidate_config.version,
                "snapshot_version": self.temporal.version, "snapshot_index": candidates.snapshot_index,
                "as_of": as_of, "snapshot_cutoff": candidates.cutoff,
                "candidate_count": len(candidates.ids), "requested_top_k": request.top_k,
                "shortfall": max(0, request.top_k - len(items)), "retrieval_shortfall": candidates.shortfall,
                "history_length_used": history_length, "route_counts": candidates.metadata["route_counts"],
                "fallback_count": candidates.metadata["fallback_count"],
                "zero_query_channels": candidates.metadata["zero_query_channels"],
                "strategy": "popular_then_confirmed_bounded_residual" if residual_applied else "causal_rrf",
                "retrieval_score_type": "equal_weight_rrf; fallback items have zero fusion score",
                "api_version": "v3", "deployment_gate_passed": self.deployment_gate_passed,
                "residual_applied": residual_applied, "selected_delta": self.delta,
                "rerank_score_type": "negative original rank plus bounded gated preference; not CTR" if scores is not None else None}


def create_app(engine):
    app = create_v2_app(engine)
    app.title = "NewsForge Confirmed Residual Recommender"
    app.version = "3.0"
    return app


def load_release(selection_path="artifacts/v3/selection.json", confirm_path="reports/v3/confirm.json",
                 fit_report_path="reports/v3/fit.json", v2_selection_path="artifacts/v2/selection.json"):
    """Verify the selected models against their frozen fit and confirmation."""
    def read(path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("release evidence must be a JSON object")
        return value

    selection, confirm, fit = map(read, (selection_path, confirm_path, fit_report_path))
    tune_path = Path(confirm_path).with_name("tune.json")
    tune = read(tune_path)
    artifact_root = Path(selection_path).parent
    tune_cache_path, confirm_cache_path = (artifact_root / phase / "cache.json" for phase in ("tune", "confirm"))
    selection_hash = sha256_file(selection_path)
    try:
        if (selection["fit_report_sha256"] != sha256_file(fit_report_path)
                or selection["v2_selection_sha256"] != sha256_file(v2_selection_path)
                or confirm["selection_sha256"] != selection_hash
                or selection["sources"] != fit["sources"]
                or selection["models"] != fit["models"]):
            raise ValueError("selection, confirmation and fit report bindings differ")
        gate = confirm["deployment_gate_passed"]
        if not isinstance(gate, bool):
            raise ValueError("deployment gate must be a boolean")
        if (confirm.get("phase") != "confirm" or confirm["delta"] != selection["delta"]
                or confirm["bindings"]["fit_report_sha256"] != selection["fit_report_sha256"]
                or confirm["bindings"]["selection_sha256"] != selection_hash
                or confirm["bindings"]["sources"] != selection["sources"]):
            raise ValueError("confirmation evidence does not match selected recipe")
        if (confirm["tune_report_sha256"] != sha256_file(tune_path)
                or tune["phase"] != "tune" or tune["selected_delta"] != selection["delta"]
                or tune["bindings"]["fit_report_sha256"] != selection["fit_report_sha256"]
                or tune["bindings"]["sources"] != selection["sources"]
                or tune["cache_manifest_sha256"] != selection["tune_cache_manifest_sha256"]
                or sha256_file(tune_cache_path) != tune["cache_manifest_sha256"]
                or sha256_file(confirm_cache_path) != confirm["cache_manifest_sha256"]
                or read(tune_cache_path)["bindings"] != tune["bindings"]
                or read(confirm_cache_path)["bindings"] != confirm["bindings"]):
            raise ValueError("incomplete or inconsistent tune/confirmation cache evidence")
        if (isinstance(selection["delta"], bool) or selection["delta"] not in (0, 2, 5, 10)
                or selection.get("primary_recipe") != "full13+gate"):
            raise ValueError("selected recipe differs from the predefined residual grid")
        required_sources = {"newsforge/v2/ranker.py", "newsforge/v3/residual.py", "newsforge/v3/fast_retrieval.py"}
        if not required_sources.issubset(selection["sources"]):
            raise ValueError("release lacks critical feature/model/generator source bindings")
        interval = confirm["metrics"]["full13"]["versus_baseline"]
        delta_r10, ci = interval["delta"], interval["ci95"]
        if (not isinstance(ci, list) or len(ci) != 2 or not np.isfinite([delta_r10, *ci]).all()
                or ci[0] > ci[1] or gate != bool(delta_r10 > 0 and ci[0] > 0)):
            raise ValueError("deployment gate disagrees with its prescribed confirmation interval")
        if confirm.get("deployment_available_as_of", DEPLOYMENT_AVAILABLE_AS_OF) != DEPLOYMENT_AVAILABLE_AS_OF:
            raise ValueError("deployment availability differs from declared confirmation maturity")
        if set(selection["models"]) != {"full13", "no_category", "no_recent"}:
            raise ValueError("selected model variants differ from fixed experiment")
        model_files = {}
        for variant, paths in selection["models"].items():
            if set(paths) != {str(seed) for seed in SEEDS}:
                raise ValueError("selected ensemble seeds differ from the fixed three seeds")
            for seed in SEEDS:
                record = fit["runs"][str(seed)]["models"][variant]
                if paths[str(seed)] != record["path"]:
                    raise ValueError("selected model path differs from fit report")
                model_files[record["path"]] = record["npz_sha256"]
                model_files[record["path"] + ".json"] = record["sidecar_sha256"]
        if model_files != selection["model_files"]:
            raise ValueError("selected model SHA256 inventory differs from fit report")
        for path, expected in {**selection["sources"], **model_files}.items():
            if sha256_file(path) != expected:
                raise ValueError(f"release source/model SHA256 mismatch: {path}")
        models = []
        for seed in SEEDS:
            run = fit["runs"][str(seed)]
            record = run["models"]["full13"]
            expected_bindings = {"sources": fit["sources"], "input_files": fit["input_files"],
                                 "fit_rows": fit["fit_rows"], "seed": seed,
                                 "pairs_sha256": run["pairs_sha256"],
                                 "pair_features_sha256": run["pair_features_sha256"],
                                 "environment": fit["environment"]}
            model = PairwiseResidual.load(record["path"], expected_bindings=expected_bindings)
            if model.variant != "full13" or model.fit_metadata != record["fit_metadata"]:
                raise ValueError("deployed residual differs from fitted full13 model")
            models.append(model)
    except KeyError as error:
        raise ValueError(f"incomplete v3 release evidence: {error}") from error
    evidence = {"selection_sha256": selection_hash, "confirmation_sha256": sha256_file(confirm_path),
                "tune_report_sha256": sha256_file(tune_path),
                "tune_cache_manifest_sha256": sha256_file(tune_cache_path),
                "confirm_cache_manifest_sha256": sha256_file(confirm_cache_path),
                "fit_report_sha256": selection["fit_report_sha256"],
                "v2_selection_sha256": selection["v2_selection_sha256"],
                "model_files_sha256": model_files, "fit_sources_sha256": selection["sources"]}
    return selection, gate, tuple(models), evidence


def load_engine(processed_dir="data/processed", artifact_dir="artifacts", temporal_dir="data/temporal_v2",
                selection_path="artifacts/v3/selection.json", confirm_path="reports/v3/confirm.json",
                v2_selection_path="artifacts/v2/selection.json", fit_report_path="reports/v3/fit.json"):
    selection, gate, models, evidence = load_release(selection_path, confirm_path, fit_report_path,
                                                    v2_selection_path)
    config = V2ServiceConfig.from_selection(v2_selection_path)
    processed = Path(processed_dir)
    temporal, generator = None, None
    try:
        temporal = TemporalStore(temporal_dir, processed_dir=processed, verify=True)
        generator = FastCandidateGenerator.from_paths(processed, artifact_dir, temporal,
                                                      encoder_path=config.encoder_path, config=config.candidate_config)
        verify_generator_selection(config, generator)
        manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
        news_ids = json.loads((processed / "news_ids.json").read_text(encoding="utf-8"))
        return V3Engine(generator, news_ids, config, models=models, delta=selection["delta"],
                        deployment_gate_passed=gate,
                        text_dimensions=int(manifest["features"]["text_dimensions"]), evidence_bindings=evidence)
    except BaseException:
        if generator is not None:
            generator.close()
        if temporal is not None:
            temporal.close()
        raise


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--temporal-dir", default="data/temporal_v2")
    parser.add_argument("--selection", default="artifacts/v3/selection.json")
    parser.add_argument("--confirm", default="reports/v3/confirm.json")
    parser.add_argument("--v2-selection", default="artifacts/v2/selection.json")
    parser.add_argument("--fit-report", default="reports/v3/fit.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    engine = load_engine(args.processed_dir, args.artifact_dir, args.temporal_dir, args.selection,
                         args.confirm, args.v2_selection, args.fit_report)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, access_log=False, workers=1)


if __name__ == "__main__":
    main()
