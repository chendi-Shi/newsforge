"""Local v2 HTTP service sharing the causal candidate/ranking implementation."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_info, threadpool_limits

from .retrieval import (CHANNEL_BITS, CandidateConfig, CandidateGenerator,
                        canonical_hash, rerank_order, sha256_file)


@dataclass(frozen=True)
class V2ServiceConfig:
    candidate_config: CandidateConfig
    ranker_path: str | None = None
    use_reranker: bool = True
    encoder_path: str = "artifacts/v2/user_encoder.npz"
    default_as_of: int | None = None
    threads: int = 1
    selection_sha256: str = "injected"
    ranker_identity: dict | None = None
    frozen_bindings: dict | None = None

    @classmethod
    def from_selection(cls, path="artifacts/v2/selection.json"):
        path = Path(path)
        selection = json.loads(path.read_text(encoding="utf-8"))
        value = selection["serving"]
        allowed = {"candidate_config", "ranker_path", "ranker_identity", "use_reranker", "encoder_path", "default_as_of", "threads"}
        if set(value) - allowed:
            raise ValueError("unknown v2 serving configuration fields")
        value = {**value, "candidate_config": CandidateConfig(**value["candidate_config"]),
                 "selection_sha256": sha256_file(path), "frozen_bindings": selection.get("bindings")}
        result = cls(**value)
        if (not isinstance(result.use_reranker, bool) or result.threads != 1
                or isinstance(result.threads, bool) or result.use_reranker and not result.ranker_path):
            raise ValueError("v2 serving requires one thread and an explicitly selected ranker when enabled")
        return result


class V2RecommendRequest(BaseModel):
    history: list[str] = Field(default_factory=list, max_length=200)
    as_of: int | None = Field(default=None, ge=0, lt=int(np.iinfo(np.int64).max))
    top_k: int = Field(default=10, ge=1, le=50)


class V2BatchRequest(BaseModel):
    requests: list[V2RecommendRequest] = Field(min_length=1, max_length=32)


class V2Engine:
    def __init__(self, generator, news_ids, config, *, ranker=None, ranker_manifest_sha256=None):
        if len(news_ids) != len(generator.features) or len(set(news_ids)) != len(news_ids):
            raise ValueError("news ID catalogue shape/order mismatch")
        if config.use_reranker and ranker is None:
            raise ValueError("selected reranker is missing")
        if config.threads != 1:
            raise ValueError("v2 service protocol requires one CPU thread")
        self.generator, self.news_ids, self.config, self.ranker = generator, list(news_ids), config, ranker
        self.id_to_index = {value: index for index, value in enumerate(news_ids) if index > 0}
        self.temporal = generator.temporal
        cutoffs = [generator.encoder.metadata.get("train_cutoff_timestamp")]
        if config.use_reranker or config.candidate_config.freeze_feedback:
            cutoffs.append(self.temporal.manifest.get("train_cutoff_timestamp"))
        known_cutoffs = [int(value) for value in cutoffs if value is not None]
        training_cutoff = max(known_cutoffs) if known_cutoffs else None
        # Strictly cutoff > last training event, including an exact hour tie.
        model_available = 0 if training_cutoff is None else (int(training_cutoff) // 3600 + 2) * 3600
        self.minimum_as_of = max(int(self.temporal.manifest["min_as_of"]), model_available)
        self.maximum_as_of = int(self.temporal.manifest["max_as_of"])
        self.default_as_of = self.maximum_as_of if config.default_as_of is None else int(config.default_as_of)
        self._validate_as_of(self.default_as_of)
        self._lock, self._closed = threading.Lock(), False
        self._thread_limiter = threadpool_limits(limits=config.threads)
        self.bindings = {"candidate_generator_version": generator.version,
                         "candidate_inputs": generator.bindings,
                         "candidate_config_sha256": config.candidate_config.version,
                         "selection_sha256": config.selection_sha256,
                         "temporal_manifest_sha256": self.temporal.version,
                         "temporal_policy_sha256": self.temporal.policy_hash,
                         "ranker_sha256": getattr(ranker, "model_version", None),
                         "ranker_manifest_sha256": ranker_manifest_sha256,
                         "reranker_enabled": config.use_reranker,
                         "service_source_sha256": sha256_file(__file__),
                         "runtime_source_sha256": {name: sha256_file(Path(__file__).parent / name)
                                                   for name in ("retrieval.py", "serve.py", "temporal.py", "ranker.py")
                                                   if (Path(__file__).parent / name).is_file()}}
        self.model_version = canonical_hash(self.bindings)

    def _validate_as_of(self, as_of):
        if not self.minimum_as_of <= as_of <= self.maximum_as_of:
            raise ValueError(f"as_of outside supported range [{self.minimum_as_of}, {self.maximum_as_of}]")
        if self.config.candidate_config.freeze_feedback:
            self.temporal.frozen_snapshot(as_of)
        else:
            self.temporal.snapshot(as_of)

    def health(self):
        return {"status": "closed" if self._closed else "ok", "api_version": "v2",
                "server_pid": os.getpid(), "model_version": self.model_version,
                "candidate_config": asdict(self.config.candidate_config), "bindings": self.bindings,
                "catalogue_rows": len(self.news_ids), "default_as_of": self.default_as_of,
                "minimum_as_of": self.minimum_as_of, "maximum_as_of": self.maximum_as_of,
                "timestamp_policy": "strict first_seen < hourly snapshot cutoff; no future snapshot fallback",
                "device": "cpu", "threads": self.config.threads,
                "native_thread_pools": [{key: pool.get(key) for key in ("internal_api", "prefix", "num_threads")}
                                        for pool in threadpool_info()],
                "reranker_enabled": self.config.use_reranker}

    def _recommend(self, request):
        as_of = self.default_as_of if request.as_of is None else int(request.as_of)
        self._validate_as_of(as_of)
        history = []
        for item in request.history:
            numeric = self.id_to_index.get(item)
            if numeric is None:
                raise ValueError("history contains PAD/OOV or unknown article IDs")
            history.append(numeric)
        if request.top_k > self.config.candidate_config.pool_size:
            raise ValueError("top_k exceeds the selected candidate pool size")
        candidates = self.generator.retrieve(history, as_of, config=self.config.candidate_config)
        if self.config.use_reranker and len(candidates.ids):
            # Every supplied ID is excluded by retrieval, but both learned
            # components use the same last-20 history as offline evaluation.
            scores = np.asarray(self.ranker.score(history[-20:], candidates.ids, as_of))
            order = rerank_order(candidates.ids, scores)
        else:
            scores = None
            order = np.arange(len(candidates.ids))
        items = []
        for position in order[:request.top_k]:
            item = int(candidates.ids[position])
            source_mask = int(candidates.source_mask[position])
            items.append({"item_id": self.news_ids[item], "retrieval_score": float(candidates.scores[position]),
                          "rerank_score": None if scores is None else float(scores[position]),
                          "sources": [name for name, bit in CHANNEL_BITS.items() if source_mask & bit]})
        return {"items": items, "model_version": self.model_version,
                "candidate_config_version": self.config.candidate_config.version,
                "snapshot_version": self.temporal.version, "snapshot_index": candidates.snapshot_index,
                "as_of": as_of, "snapshot_cutoff": candidates.cutoff,
                "candidate_count": len(candidates.ids), "requested_top_k": request.top_k,
                "shortfall": max(0, request.top_k - len(items)), "retrieval_shortfall": candidates.shortfall,
                "history_length_used": candidates.metadata["history_length_used"],
                "route_counts": candidates.metadata["route_counts"],
                "fallback_count": candidates.metadata["fallback_count"],
                "zero_query_channels": candidates.metadata["zero_query_channels"],
                "strategy": "causal_rrf_then_selected_ranker" if scores is not None else "causal_rrf",
                "retrieval_score_type": "equal_weight_rrf; fallback items have zero fusion score"}

    def recommend_batch(self, requests):
        if not 1 <= len(requests) <= 32:
            raise ValueError("batch size must be between 1 and 32")
        with self._lock:
            if self._closed:
                raise RuntimeError("service is closed")
            return [self._recommend(request) for request in requests]

    def recommend(self, request):
        return self.recommend_batch([request])[0]

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self.generator.close()
                close_temporal = getattr(self.temporal, "close", None)
                if close_temporal is not None:
                    close_temporal()
                self._thread_limiter.restore_original_limits()


def verify_generator_selection(config, generator):
    """Reject self-consistent replacement artifacts after internal selection."""
    frozen = config.frozen_bindings
    if not frozen:
        raise ValueError("frozen internal-selection bindings are required to load a service")
    try:
        processed, artifacts = frozen["processed_files"], frozen["retrieval_artifacts"]
        expected = {"features": processed["news_features.npy"],
                    "first_seen": processed["news_first_seen.npy"],
                    "train_clicks": processed["news_train_clicks.npy"],
                    "train_exposures": processed["news_train_exposures.npy"],
                    "news_ids": processed["news_ids.json"],
                    "embeddings": artifacts["article_embeddings"], "retriever": artifacts["retriever"],
                    "encoder": artifacts["encoder"], "encoder_manifest": artifacts["encoder_manifest"],
                    "data_manifest": frozen["data_manifest"], "temporal_manifest": frozen["temporal_manifest"]}
        if generator.bindings != expected:
            raise ValueError("candidate inputs differ from frozen internal selection")
        for name in ("retrieval.py", "temporal.py", "ranker.py"):
            if sha256_file(Path(__file__).parent / name) != frozen["source"][name]:
                raise ValueError("runtime source differs from frozen internal selection")
    except KeyError as error:
        raise ValueError("incomplete frozen internal-selection bindings") from error


def verify_ranker_selection(config, ranker, manifest_sha256):
    identity = config.ranker_identity
    if not identity or not config.frozen_bindings:
        raise ValueError("selected ranker identity and bindings are required")
    try:
        run = f'{identity["variant"]}-seed{identity["seed"]}'
        frozen = config.frozen_bindings["ranker_artifacts"][run]
        if (str(Path(config.ranker_path)).replace("\\", "/") != frozen["model_path"]
                or str(Path(identity["model_path"])).replace("\\", "/") != frozen["model_path"]
                or identity["model_sha256"] != frozen["model_sha256"]
                or ranker.model_version != frozen["model_sha256"]
                or manifest_sha256 != frozen["manifest_sha256"]
                or ranker.variant != identity["variant"]
                or ranker.metadata.get("seed") != identity["seed"]):
            raise ValueError("ranker differs from frozen internal selection")
    except KeyError as error:
        raise ValueError("incomplete frozen ranker identity") from error


def load_engine(processed_dir="data/processed", artifact_dir="artifacts", temporal_dir="data/temporal_v2",
                selection_path="artifacts/v2/selection.json"):
    from .temporal import TemporalStore
    config = V2ServiceConfig.from_selection(selection_path)
    processed_dir = Path(processed_dir)
    temporal = TemporalStore(temporal_dir, processed_dir=processed_dir, verify=True)
    generator = None
    ranker, ranker_manifest_sha256 = None, None
    try:
        generator = CandidateGenerator.from_paths(processed_dir, artifact_dir, temporal,
                                                   encoder_path=config.encoder_path, config=config.candidate_config)
        verify_generator_selection(config, generator)
        news_ids = json.loads((processed_dir / "news_ids.json").read_text(encoding="utf-8"))
        if config.use_reranker:
            from .ranker import CausalRanker
            store = SimpleNamespace(path=processed_dir, features=generator.features,
                                    first_seen=generator.first_seen, clicks=generator.train_clicks,
                                    exposures=generator.train_exposures, news_ids=news_ids,
                                    manifest=json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8")))
            ranker = CausalRanker.load(config.ranker_path, store, temporal)
            ranker_manifest_sha256 = sha256_file(str(config.ranker_path) + ".manifest.json")
            verify_ranker_selection(config, ranker, ranker_manifest_sha256)
        return V2Engine(generator, news_ids, config, ranker=ranker,
                        ranker_manifest_sha256=ranker_manifest_sha256)
    except BaseException:
        if generator is not None:
            generator.close()
        temporal.close()
        raise


def create_app(engine):
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            engine.close()
    app = FastAPI(title="NewsForge Causal Recommender", version="2.0", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    def health():
        return engine.health()

    @app.post("/recommend")
    def recommend(request: V2RecommendRequest):
        try:
            return engine.recommend(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/recommend_batch")
    def recommend_batch(request: V2BatchRequest):
        try:
            return {"results": engine.recommend_batch(request.requests), "batch_size": len(request.requests)}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--temporal-dir", default="data/temporal_v2")
    parser.add_argument("--selection", default="artifacts/v2/selection.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    engine = load_engine(args.processed_dir, args.artifact_dir, args.temporal_dir, args.selection)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, access_log=False, workers=1)


if __name__ == "__main__":
    main()
