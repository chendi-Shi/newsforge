"""Local FastAPI news retrieval and batched candidate-aware reranking service.

Run ``python -m newsforge.serve --help`` after training and building an index.
The API takes article IDs, never an internal user ID or raw user record.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field, model_validator
import torch

from newsforge.retrieval import CatalogIndex, RESERVED_IDS, sha256_file


class RecommendRequest(BaseModel):
    history: list[str] = Field(default_factory=list, max_length=200)
    top_k: int = Field(default=10, ge=1, le=50)
    as_of: int | None = Field(default=None, ge=0, lt=int(np.iinfo(np.int64).max))
    candidate_pool_size: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def check_pool(self):
        if self.candidate_pool_size < self.top_k:
            raise ValueError("candidate_pool_size must be >= top_k")
        return self


class RecommendBatchRequest(BaseModel):
    requests: list[RecommendRequest] = Field(min_length=1, max_length=32)


class RecommendationItem(BaseModel):
    item_id: str
    retrieval_score: float
    rerank_score: float | None


class RecommendResponse(BaseModel):
    items: list[RecommendationItem]
    model_version: str
    index_version: str
    as_of: int
    history_length_used: int
    candidate_count: int
    requested_top_k: int
    shortfall: int
    retrieval_shortfall: int
    retrieval_examined: int
    retrieval_exhausted: bool
    strategy: str
    retrieval_score_type: str


class RecommendBatchResponse(BaseModel):
    results: list[RecommendResponse]
    batch_size: int


class RecommendationEngine:
    """CPU inference with batch encoding and bounded candidate scoring chunks.

    Model interfaces:
      retriever.encode_user(history_features[B,H,D], history_mask[B,H]) -> [B,E]
      ranker.score_candidates(history_features, history_mask, candidates[B,C,D]) -> [B,C]

    No label-derived candidate priors are sent to the ranker. Training-click
    popularity is used only for explicitly empty histories. Shared models are
    protected by a lock so per-request thread pools do not oversubscribe CPU.
    The /recommend_batch route amortizes model calls across multiple requests.
    """

    def __init__(self, catalogue: CatalogIndex, news_features: np.ndarray, retriever,
                 ranker, *, model_version: str, max_history: int = 20,
                 candidate_batch_size: int = 256, max_overfetch: int = 8192,
                 torch_threads: int = 1, model_train_cutoff: int | None = None):
        if news_features.ndim != 2 or len(news_features) != len(catalogue.news_ids):
            raise ValueError("news_features/catalogue row mismatch")
        if max_history < 1 or candidate_batch_size < 1 or max_overfetch < 1 or torch_threads < 1:
            raise ValueError("invalid inference batch/history/thread configuration")
        if model_train_cutoff is not None:
            if (not isinstance(model_train_cutoff, int) or isinstance(model_train_cutoff, bool)
                    or not 0 <= model_train_cutoff < np.iinfo(np.int64).max):
                raise ValueError("model_train_cutoff must be a known nonnegative Unix timestamp")
            if catalogue.default_as_of < model_train_cutoff:
                raise ValueError("index default_as_of predates the model training cutoff")
            # Persisted project indexes use the training snapshot as their
            # default cutoff, including the popularity counts in that snapshot.
            model_train_cutoff = max(model_train_cutoff, catalogue.default_as_of)
        self.catalogue = catalogue
        self.news_features = news_features
        self.retriever = retriever.to("cpu").eval()
        self.ranker = ranker.to("cpu").eval()
        self.model_version = model_version
        self.model_train_cutoff = model_train_cutoff
        self.max_history = max_history
        self.candidate_batch_size = candidate_batch_size
        self.max_overfetch = max_overfetch
        self.torch_threads = torch_threads
        self._lock = threading.Lock()
        self._closed = False
        torch.set_num_threads(torch_threads)

    def health(self):
        return {"status": "closed" if self._closed else "ok", "model_version": self.model_version,
                "server_pid": os.getpid(),
                "index_version": self.catalogue.version,
                "catalogue_items": self.catalogue.index.ntotal,
                "catalogue_rows": len(self.catalogue.news_ids),
                "index_kind": self.catalogue.manifest["kind"],
                "embedding_dimension": self.catalogue.index.d,
                "default_as_of": self.catalogue.default_as_of,
                "model_train_cutoff": self.model_train_cutoff,
                "device": "cpu", "torch_threads": self.torch_threads,
                "faiss_threads": self.catalogue.threads,
                "hnsw_m": self.catalogue.manifest.get("hnsw_m"),
                "ef_search": self.catalogue.manifest.get("ef_search"),
                "max_overfetch": self.max_overfetch,
                "max_history": self.max_history,
                "candidate_scoring_batch_size": self.candidate_batch_size}

    def _history(self, request: RecommendRequest):
        cutoff = self.catalogue._cutoff(request.as_of)
        # Time-filtering articles alone cannot undo future information already
        # contained in trained weights or the fixed training-popularity table.
        if self.model_train_cutoff is not None and cutoff < self.model_train_cutoff:
            raise ValueError(f"as_of must be >= model_train_cutoff ({self.model_train_cutoff})")
        numeric, usable = [], []
        for value in request.history:
            if value.upper() in RESERVED_IDS or value not in self.catalogue.id_to_index:
                raise ValueError("history contains PAD/OOV or unknown article IDs")
            item = self.catalogue.id_to_index[value]
            if self.catalogue.first_seen[item] > cutoff:
                raise ValueError("history contains an article first observed after as_of")
            numeric.append(item)
            # Known IDs can lack usable text features. They remain excluded
            # from recommendations, but cannot create a nonzero user vector.
            if np.any(self.news_features[item] != 0):
                usable.append(item)
        # Exclude every seen item; only the latest H events enter the encoder.
        return usable[-self.max_history:], set(numeric), cutoff

    @torch.inference_mode()
    def recommend_batch(self, requests: list[RecommendRequest]):
        if not 1 <= len(requests) <= 32:
            raise ValueError("batch size must be between 1 and 32")
        with self._lock:
            if self._closed:
                raise RuntimeError("recommendation engine is closed")
            return self._recommend_batch(requests)

    def close(self):
        """Release an owned feature memmap, including its Windows file handle."""
        with self._lock:
            if not self._closed:
                self._closed = True
                mapping = getattr(self.news_features, "_mmap", None)
                if mapping is not None:
                    mapping.close()

    def _recommend_batch(self, requests: list[RecommendRequest]):
        parsed = [self._history(request) for request in requests]
        history_ids = np.zeros((len(requests), self.max_history), dtype=np.int64)
        for row, (history, _, _) in enumerate(parsed):
            history_ids[row, :len(history)] = history
        history_mask = torch.from_numpy(history_ids != 0)
        history_features = torch.from_numpy(np.ascontiguousarray(self.news_features[history_ids], dtype=np.float32))
        active_rows = [i for i, (history, _, _) in enumerate(parsed) if history]
        vectors = {}
        if active_rows:
            encoded = self.retriever.encode_user(history_features[active_rows], history_mask[active_rows])
            encoded = encoded.detach().cpu().numpy()
            if encoded.shape != (len(active_rows), self.catalogue.index.d) or not np.isfinite(encoded).all():
                raise RuntimeError("retriever returned invalid user embeddings")
            vectors = dict(zip(active_rows, encoded))
        candidates = []
        for row, (request, (history, seen, cutoff)) in enumerate(zip(requests, parsed)):
            if history:
                found = self.catalogue.search(vectors[row], request.candidate_pool_size,
                                               as_of=cutoff, seen=seen,
                                               max_overfetch=self.max_overfetch)
            else:
                found = self.catalogue.popular(request.candidate_pool_size, as_of=cutoff, seen=seen)
            candidates.append(found)
        scores = {}
        scoring_rows = [row for row in active_rows if candidates[row].indices]
        if scoring_rows:
            max_candidates = max(len(candidates[row].indices) for row in scoring_rows)
            all_scores = np.full((len(scoring_rows), max_candidates), -np.inf, dtype=np.float32)
            for start in range(0, max_candidates, self.candidate_batch_size):
                width = min(self.candidate_batch_size, max_candidates - start)
                candidate_ids = np.zeros((len(scoring_rows), width), dtype=np.int64)
                for position, row in enumerate(scoring_rows):
                    part = candidates[row].indices[start:start + width]
                    candidate_ids[position, :len(part)] = part
                features = torch.from_numpy(np.ascontiguousarray(self.news_features[candidate_ids], dtype=np.float32))
                logits = self.ranker.score_candidates(history_features[scoring_rows],
                                                       history_mask[scoring_rows], features)
                logits = logits.detach().cpu().numpy()
                if logits.shape != candidate_ids.shape or not np.isfinite(logits).all():
                    raise RuntimeError("reranker returned invalid candidate scores")
                all_scores[:, start:start + width] = np.where(candidate_ids != 0, logits, -np.inf)
            for position, row in enumerate(scoring_rows):
                scores[row] = all_scores[position, :len(candidates[row].indices)]
        responses = []
        for row, (request, found, (history, _, cutoff)) in enumerate(zip(requests, candidates, parsed)):
            if history and found.indices:
                ordering = sorted(range(len(found.indices)), key=lambda j: (-float(scores[row][j]), found.indices[j]))
                strategy, score_type = "two_tower_then_candidate_attention", "cosine_inner_product"
            else:
                ordering = list(range(len(found.indices)))
                strategy = "train_popularity_cold_start" if not history else "two_tower_then_candidate_attention"
                score_type = "log1p_training_click_count" if not history else "cosine_inner_product"
            items = [RecommendationItem(item_id=self.catalogue.news_ids[found.indices[j]],
                                        retrieval_score=found.scores[j],
                                        rerank_score=float(scores[row][j]) if history else None)
                     for j in ordering[:request.top_k]]
            responses.append(RecommendResponse(
                items=items, model_version=self.model_version, index_version=self.catalogue.version,
                as_of=cutoff, history_length_used=len(history), candidate_count=len(found.indices),
                requested_top_k=request.top_k, shortfall=max(0, request.top_k - len(items)),
                retrieval_shortfall=found.shortfall, retrieval_examined=found.examined,
                retrieval_exhausted=found.exhausted, strategy=strategy, retrieval_score_type=score_type))
        return responses

    def recommend(self, request: RecommendRequest):
        return self.recommend_batch([request])[0]

    def warmup(self, iterations: int = 2):
        self.catalogue.warmup(iterations)
        visible = self.catalogue.indexed_ids[self.catalogue.first_seen[self.catalogue.indexed_ids]
                                            <= self.catalogue.default_as_of]
        if len(visible):
            example = RecommendRequest(history=[self.catalogue.news_ids[int(visible[0])]],
                                       top_k=1, candidate_pool_size=min(100, len(visible)))
            for _ in range(iterations):
                self.recommend(example)


def load_engine(processed_dir: str | Path, artifact_dir: str | Path, index_dir: str | Path,
                *, torch_threads=1, faiss_threads=1, max_history=20, candidate_batch_size=256,
                max_overfetch=8192):
    from newsforge.models import TwoTower, AttentionRanker
    processed_dir, artifact_dir = Path(processed_dir), Path(artifact_dir)
    retriever_path, ranker_path = artifact_dir / "retriever.pt", artifact_dir / "reranker.pt"
    retriever_checkpoint = torch.load(retriever_path, map_location="cpu", weights_only=True)
    ranker_checkpoint = torch.load(ranker_path, map_location="cpu", weights_only=True)
    training_cutoffs = []
    for name, checkpoint in (("retriever", retriever_checkpoint), ("reranker", ranker_checkpoint)):
        cutoff = checkpoint.get("train_cutoff_timestamp")
        if (not isinstance(cutoff, int) or isinstance(cutoff, bool)
                or not 0 <= cutoff < np.iinfo(np.int64).max):
            raise ValueError(f"{name} checkpoint requires a valid train_cutoff_timestamp")
        training_cutoffs.append(cutoff)
    # Both models contribute to each personalized response, so the newer
    # training cutoff governs the complete serving bundle.
    model_train_cutoff = max(training_cutoffs)
    retriever = TwoTower(**retriever_checkpoint["config"])
    retriever.load_state_dict(retriever_checkpoint["state_dict"], strict=True)
    ranker = AttentionRanker(**ranker_checkpoint["config"])
    ranker.load_state_dict(ranker_checkpoint["state_dict"], strict=True)
    hashes = {"retriever": sha256_file(retriever_path), "reranker": sha256_file(ranker_path)}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    catalogue = CatalogIndex.load(index_dir, threads=faiss_threads)
    if catalogue.default_as_of < model_train_cutoff:
        raise ValueError("index default_as_of predates the model training cutoff")
    expected_model = catalogue.manifest.get("model_version", "unknown")
    # A run name is not a weight identity: an overwritten checkpoint can keep
    # the same run_id. Serving requires a cryptographic checkpoint binding.
    accepted_versions = {hashes["retriever"], version}
    if expected_model not in accepted_versions:
        raise ValueError("index/retriever model-version mismatch")
    feature_path = processed_dir / "news_features.npy"
    ids_path = processed_dir / "news_ids.json"
    for source_name, path in (("news_features", feature_path), ("news_ids", ids_path)):
        expected = catalogue.manifest.get("source_hashes", {}).get(source_name)
        if expected is not None and sha256_file(path) != expected:
            raise ValueError(f"processed {source_name}/index source hash mismatch")
    if json.loads(ids_path.read_text(encoding="utf-8")) != catalogue.news_ids:
        raise ValueError("processed/index article-ID order mismatch")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if features.shape[1] != retriever_checkpoint["config"]["feature_dim"]:
        raise ValueError("feature dimension/model configuration mismatch")
    return RecommendationEngine(catalogue, features, retriever, ranker, model_version=version,
                                 max_history=max_history, candidate_batch_size=candidate_batch_size,
                                 max_overfetch=max_overfetch, torch_threads=torch_threads,
                                 model_train_cutoff=model_train_cutoff)


def create_app(engine: RecommendationEngine, *, warmup_iterations=2) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        try:
            engine.warmup(warmup_iterations)
            yield
        finally:
            engine.close()

    app = FastAPI(title="NewsForge Research Recommender", version="1.0", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    def health():
        return engine.health()

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(request: RecommendRequest):
        try:
            return engine.recommend(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/recommend_batch", response_model=RecommendBatchResponse)
    def recommend_batch(request: RecommendBatchRequest):
        try:
            results = engine.recommend_batch(request.requests)
            return RecommendBatchResponse(results=results, batch_size=len(results))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--index-dir", default="artifacts/index_hnsw")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--candidate-batch-size", type=int, default=256)
    parser.add_argument("--max-overfetch", type=int, default=8192)
    args = parser.parse_args()
    engine = load_engine(args.processed_dir, args.artifact_dir, args.index_dir,
                         torch_threads=args.torch_threads, faiss_threads=args.faiss_threads,
                         max_history=args.max_history, candidate_batch_size=args.candidate_batch_size,
                         max_overfetch=args.max_overfetch)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, access_log=False, workers=1)


if __name__ == "__main__":
    main()
