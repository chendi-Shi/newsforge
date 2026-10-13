"""Full-candidate impression evaluation, separated from ANN neighbor accuracy."""
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import impression_metrics, summarize


def select_validation_rows(store, limit=12000):
    rows = np.flatnonzero(store.split == 1)
    if len(rows) > limit:
        rows = np.sort(np.random.default_rng(2026).choice(rows, limit, replace=False))
    return rows


def encode_catalog(store, model, batch_size=4096):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model.encode_news(torch.from_numpy(np.array(store.features[start:start + batch_size]))).numpy()
                               for start in range(0, len(store.features), batch_size)])


def evaluate(store, model, kind, rows, *, output=None, batch_size=128):
    started = time.perf_counter()
    values = np.empty((len(rows), 4), dtype=np.float64)
    if model is not None:
        model.eval()
    if kind == "tower":
        catalog = encode_catalog(store, model)
    elif kind == "content":
        catalog = np.array(store.features)
        catalog /= np.maximum(np.linalg.norm(catalog, axis=1, keepdims=True), 1e-12)
    elif kind == "popularity":
        prior = store.clicks.sum() / max(store.exposures.sum(), 1)
        catalog = (store.clicks + 20 * prior) / (store.exposures + 20)
    elif kind != "ranker":
        raise ValueError(kind)
    if kind == "ranker":
        batch_size = min(batch_size, 32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            ids, candidate_mask = store.padded_candidates(batch)
            if kind == "popularity":
                scores = catalog[ids]
            elif kind == "content":
                pooled = np.array(store.mean_history[batch])
                pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
                scores = np.einsum("bd,bcd->bc", pooled, catalog[ids])
            elif kind == "tower":
                history, mask = store.history_batch(batch, mean=model.user_pooling == "mean")
                users = model.encode_user(torch.from_numpy(history), torch.from_numpy(mask)).numpy()
                scores = np.einsum("bd,bcd->bc", users, catalog[ids])
            else:
                history, mask = store.history_batch(batch)
                scores = model.score_candidates(torch.from_numpy(history), torch.from_numpy(mask),
                                                torch.from_numpy(store.features[ids])).numpy()
            for offset, row in enumerate(batch):
                _, labels = store.candidates_for(row)
                values[start + offset] = impression_metrics(labels, scores[offset, :len(labels)])
    histories = (store.history[rows] > 0).sum(1)
    groups = {}
    for name, mask in [("empty_history", histories == 0), ("history_1_to_5", (histories > 0) & (histories <= 5)),
                       ("history_6_to_20", histories > 5)]:
        groups[name] = {"impressions": int(mask.sum()), "metrics": summarize(values[mask]) if mask.any() else None}
    cold_rows = np.zeros(len(rows), dtype=bool)
    zero_click_rows = np.zeros(len(rows), dtype=bool)
    cutoff = int(store.manifest["train_cutoff_timestamp"])
    for offset, row in enumerate(rows):
        ids, labels = store.candidates_for(row)
        positive = ids[labels == 1]
        cold_rows[offset] = np.any(store.first_seen[positive] > cutoff)
        zero_click_rows[offset] = np.any(store.clicks[positive] == 0)
    for name, mask in [("has_train_unseen_clicked_news", cold_rows), ("has_zero_train_click_news", zero_click_rows)]:
        groups[name] = {"impressions": int(mask.sum()), "metrics": summarize(values[mask]) if mask.any() else None}
    report = {"impressions": len(rows), "candidate_exposures": int((store.offsets[rows + 1] - store.offsets[rows]).sum()),
              "metrics": summarize(values), "groups": groups, "seconds": time.perf_counter() - started,
              "protocol": "all logged candidates; impression macro average; no evaluation negative sampling"}
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, values)
    return report, values
