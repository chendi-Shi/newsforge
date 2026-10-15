"""Read-only localhost HTTP replay of the confirmed Adressa adaptation recipe."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

from newsforge.external.common import DAY, END, HOUR, START
from newsforge.external.evaluate import frozen_features
from newsforge.external.prepare import connect_readonly
from .common import MODEL_NAMES, SEEDS, check_run, phase_receipt, read_json, sha256
from .replay import replay_query
from .scoring import confirmation_passes, ensemble, load_adapted_models, select_delta


EARLIEST_TIMESTAMP = START + 4 * DAY


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    user: int = Field(ge=1)
    timestamp: int = Field(ge=EARLIEST_TIMESTAMP, lt=END)
    top_k: int = Field(default=10, ge=1, le=100)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    requests: list[RecommendRequest] = Field(min_length=1, max_length=32)


def validate_release(run, receipts):
    """Recompute the prescribed gate and check every adjacent phase binding."""
    run = Path(run)
    selection = read_json(run / "tune/selection.json")
    decision = read_json(run / "confirm/decision.json")
    if (receipts["fit"]["prepare_receipt_sha256"] != sha256(run / "prepare/receipt.json")
            or selection["fit_receipt_sha256"] != sha256(run / "fit/receipt.json")
            or selection["prepare_receipt_sha256"] != sha256(run / "prepare/receipt.json")
            or selection != receipts["tune"]["selection"]
            or decision != receipts["confirm"]["decision"]
            or decision["selection_sha256"] != sha256(run / "tune/selection.json")
            or receipts["confirm"]["selection_sha256"] != decision["selection_sha256"]
            or receipts["evaluate"]["confirm_receipt_sha256"] != sha256(run / "confirm/receipt.json")):
        raise ValueError("inconsistent adaptation phase bindings")
    delta = decision["delta"]
    if (type(delta) is not int or delta not in (0, 2, 5, 10)
            or delta != selection["delta"] or delta != select_delta(selection["grid"])
            or selection["primary_recipe"] != "adapted_local"):
        raise ValueError("decision does not match the fixed tune selection")
    comparison = decision["primary_comparison"]
    interval = comparison["ci95"]
    if (not isinstance(interval, list) or len(interval) != 2
            or not np.isfinite([comparison["difference"], *interval]).all()
            or interval[0] > interval[1]):
        raise ValueError("invalid confirmation interval")
    count = decision["requests"]
    baseline, adapted = decision["baseline_hits10"], decision["adapted_local_hits10"]
    if (type(count) is not int or count < 1 or type(baseline) is not int or type(adapted) is not int
            or not 0 <= baseline <= count or not 0 <= adapted <= count
            or abs(comparison["difference"] - (adapted - baseline) / count) > 1e-12):
        raise ValueError("confirmation difference does not match its hit counts")
    gate = decision["gate_passed"]
    if type(gate) is not bool or gate != confirmation_passes(delta, comparison):
        raise ValueError("deployment gate disagrees with the prescribed confirmation rule")
    if (decision["deployed_recipe"] != ("adapted_local" if gate else "baseline")
            or type(decision["earliest_service_timestamp"]) is not int
            or decision["earliest_service_timestamp"] != EARLIEST_TIMESTAMP):
        raise ValueError("invalid deployed recipe or retrospective service boundary")
    return decision


class Engine:
    def __init__(self, repo, base, run):
        self.repo, self.base, self.run = map(Path, (repo, base, run))
        self.manifest = check_run(self.repo, self.base, self.run, verify_base=True)
        receipts = {phase: phase_receipt(self.run, phase)
                    for phase in ("prepare", "fit", "tune", "confirm", "evaluate")}
        self.decision = validate_release(self.run, receipts)
        self.version = sha256(self.run / "manifest.json")
        self.model_ids = {f"{name}-seed{seed}": receipts["fit"]["files"][f"models/{name}-seed{seed}.npz"]["sha256"]
                          for name in MODEL_NAMES for seed in SEEDS}
        self._arrays = []
        self._limiter = None
        self.closed = False
        try:
            def load(path):
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                self._arrays.append(array)
                return array

            self.first_seen = load(self.base / "prepare/first_seen.npy")
            self.local = load(self.run / "prepare/representation/features.npy")
            self.snapshots = load(self.run / "prepare/snapshots.npy")
            n_items = len(self.first_seen)
            if (self.first_seen.shape != (n_items,) or self.first_seen.dtype != np.dtype("int64")
                    or self.local.shape != (n_items, 115) or self.local.dtype != np.dtype("float32")
                    or self.snapshots.shape != ((END - START) // HOUR + 1, n_items)
                    or not all(np.isfinite(array).all() for array in self._arrays)
                    or np.any(self.local[0]) or not np.all(self.local[1:, 96] == 1)
                    or np.any(self.local[:, 97:])):
                raise ValueError("invalid bound representation or replay arrays")
            connection = connect_readonly(self.base / "prepare/events.sqlite")
            try:
                count, low, high = connection.execute("SELECT COUNT(*),MIN(id),MAX(id) FROM users").fetchone()
                if not count or low != 1 or high != count:
                    raise ValueError("users must have contiguous positive opaque integer IDs")
                self.user_max = int(high)
            finally:
                connection.close()
            self.models = load_adapted_models(self.run)
            for name in MODEL_NAMES:
                if len(self.models[name]) != len(SEEDS):
                    raise ValueError("adapted model ensemble has unexpected seed count")
                for seed, model in zip(SEEDS, self.models[name]):
                    expected = {"manifest_sha256": self.version,
                        "prepare_receipt_sha256": sha256(self.run / "prepare/receipt.json"),
                        "pair_indices_sha256": receipts["fit"]["files"][f"pairs-seed{seed}.npy"]["sha256"],
                        "seed": seed,
                        "label_semantics": "next observed read versus unobserved candidates, not exposure negatives"}
                    if model.bindings != expected or model.variant != ("no_recent" if name == "local_no_recent" else "full13"):
                        raise ValueError("adapted model training provenance mismatch")
            self._limiter = threadpool_limits(limits=1)
        except BaseException:
            self.close()
            raise

    def validate_request(self, request):
        if self.closed:
            raise RuntimeError("engine is closed")
        if not isinstance(request, RecommendRequest):
            request = RecommendRequest.model_validate(request)
        if request.user > self.user_max:
            raise ValueError("user is outside the bound opaque integer ID range")
        return request

    def recommend(self, request):
        request = self.validate_request(request)
        connection = connect_readonly(self.base / "prepare/events.sqlite")
        try:
            history, pool = replay_query(connection, self.first_seen, self.snapshots, request.user, request.timestamp)
        finally:
            connection.close()
        history_length = min(len(history), 20)
        original = np.arange(len(pool), dtype=np.int64)
        scores = -original.astype(np.float64)
        if self.decision["gate_passed"] and history_length and len(pool):
            local = frozen_features(self.local, history, pool)
            scores += self.decision["delta"] * (history_length / (history_length + 5.)) * ensemble(local, self.models["local"])
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite deployed ranking scores")
        chosen = np.lexsort((original, -scores))[:request.top_k]
        return {"items": [{"item_id": int(pool[position]), "score": float(scores[position])}
                          for position in chosen],
                "user": request.user, "timestamp": request.timestamp, "top_k": request.top_k,
                "recipe": self.decision["deployed_recipe"], "delta": self.decision["delta"],
                "gate_passed": self.decision["gate_passed"], "version": self.version,
                "snapshot_cutoff": request.timestamp // HOUR * HOUR - HOUR,
                "candidate_count": len(pool), "history_length_used": history_length,
                "shortfall": max(0, request.top_k - len(chosen)),
                "api_version": "adressa-adaptation-v1"}

    def recommend_batch(self, requests):
        # Validate the complete batch before opening any per-request database.
        values = [self.validate_request(request) for request in requests]
        if not 1 <= len(values) <= 32:
            raise ValueError("batch must contain 1 to 32 requests")
        return {"results": [self.recommend(value) for value in values], "batch_size": len(values)}

    def health(self):
        return {"status": "ok" if not self.closed else "closed", "api_version": "adressa-adaptation-v1",
                "scope": "local retrospective Jan 5-7 replay; not a live production service",
                "server_pid": os.getpid(), "version": self.version, "model_ids": self.model_ids,
                "active_model_ids": {name: digest for name, digest in self.model_ids.items()
                                     if name.startswith("local-seed") and self.decision["gate_passed"]},
                "minimum_timestamp": EARLIEST_TIMESTAMP, "maximum_timestamp_exclusive": END,
                "minimum_user": 1, "maximum_user": self.user_max, "maximum_top_k": 100,
                "maximum_batch_size": 32, "recipe": self.decision["deployed_recipe"],
                "delta": self.decision["delta"], "gate_passed": self.decision["gate_passed"],
                "numeric_threads": 1, "score_semantics": "negative candidate rank plus bounded preference adjustment; not CTR"}

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        for array in getattr(self, "_arrays", []):
            array._mmap.close()
        if self._limiter is not None:
            self._limiter.restore_original_limits()


def create_app(engine):
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(title="NewsForge Adressa Retrospective Replay", version="1.0", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    def health():
        return engine.health()

    @app.post("/recommend")
    def recommend(request: RecommendRequest):
        try:
            return engine.recommend(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/recommend_batch")
    def recommend_batch(request: BatchRequest):
        try:
            return engine.recommend_batch(request.requests)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    import uvicorn
    engine = Engine(args.repo, args.base, args.run)
    try:
        uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, workers=1, access_log=False)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
