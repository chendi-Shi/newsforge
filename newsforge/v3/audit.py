"""Independent v3 evidence audit; no training or benchmark re-selection.

Run only after fit/tune/confirm/dev reports are complete. This module does not
import the experiment's sampling, ranking, metrics, or model implementation.
It reconstructs those decisions from saved arrays and source exposure records.
Only a successful complete audit writes release_integrity.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from threadpoolctl import threadpool_limits


SEEDS = (17, 29, 43)
DELTAS = (0, 2, 5, 10)
VARIANTS = {"full13": tuple(range(13)),
            "no_category": tuple(i for i in range(13) if i not in (10, 11)),
            "no_recent": tuple(range(12))}
WINDOWS = {"fit": (1573722000, 1573736400, 20000),
           "tune": (1573740000, 1573754400, 2000),
           "confirm": (1573758000, 1573772400, 2000)}
DATA_FILES = ("manifest.json", "news_features.npy", "news_first_seen.npy",
              "news_train_clicks.npy", "news_train_exposures.npy", "news_ids.json",
              "all_history.npy", "all_candidates.npy", "all_labels.npy", "all_offsets.npy",
              "all_timestamps.npy", "all_users.npy", "all_split.npy")
CACHE_FILES = {"rows", "timestamps", "users", "denominators", "history_length",
               "lengths", "ids", "hit", "features", "clipped_scores"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, tolerance=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=0., atol=tolerance), message)


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def independent_order(clipped_scores, history_length, delta, gate=True):
    """Sort a seed-clipped ensemble without the production rank function."""
    values = np.asarray(clipped_scores, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] in (1, 3)
            and np.isfinite(values).all() and np.all(np.abs(values) <= 1.),
            "invalid independently ranked residuals")
    require(0 <= history_length <= 20 and delta in DELTAS, "invalid rank configuration")
    factor = 0. if history_length == 0 else (history_length / (history_length + 5.) if gate else 1.)
    residual = values.mean(axis=0)
    scores = [-i + delta * factor * float(value) for i, value in enumerate(residual)]
    order = np.asarray(sorted(range(len(scores)), key=lambda i: (-scores[i], i)), dtype=np.int64)
    require(set(order.tolist()) == set(range(len(scores))), "ranking is not a permutation")
    if not history_length or not delta:
        require(np.array_equal(order, np.arange(len(scores))), "baseline fallback changed order")
    # A score tie at the exact 2*delta boundary retains the earlier item.
    positions = np.empty(len(order), dtype=np.int64)
    positions[order] = np.arange(len(order))
    distance = max(1, int(2 * delta))
    if len(order) > distance:
        suffix_minimum = np.minimum.accumulate(positions[::-1])[::-1]
        require(np.all(positions[:-distance] < suffix_minimum[distance:]), "bounded inversion property failed")
    return order


def independent_interval(before, after, denominators, users):
    """Percentile bootstrap of paired micro differences, clustered by user."""
    before, after, denominators = [np.asarray(x, dtype=np.int64)
                                    for x in (before, after, denominators)]
    users = np.asarray(users)
    require(before.shape == after.shape == denominators.shape == users.shape
            and before.ndim == 1 and len(before), "bootstrap input shapes")
    require(np.all(before >= 0) and np.all(after >= 0)
            and np.all(before <= denominators) and np.all(after <= denominators)
            and denominators.sum() > 0, "bootstrap counts")
    unique_users, inverse = np.unique(users, return_inverse=True)
    grouped_delta = np.zeros(len(unique_users), dtype=np.int64)
    grouped_total = np.zeros(len(unique_users), dtype=np.int64)
    np.add.at(grouped_delta, inverse, after - before)
    np.add.at(grouped_total, inverse, denominators)
    rng = np.random.default_rng(2026)
    samples = []
    for _ in range(1000):
        selected = rng.integers(0, len(unique_users), size=len(unique_users))
        denominator = grouped_total[selected].sum()
        if denominator:
            samples.append(float(grouped_delta[selected].sum() / denominator))
    require(bool(samples), "bootstrap has no positive denominators")
    return {"delta": float((after - before).sum() / denominators.sum()),
            "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "resampling_unit": "user", "replicates": len(samples), "seed": 2026}


class Audit:
    def __init__(self, root, data="data/processed", temporal="data/temporal_v2",
                 artifacts="artifacts/v3", reports="reports/v3"):
        self.root = Path(root).resolve()
        self.data, self.temporal = self.path(data), self.path(temporal)
        self.artifacts, self.reports = self.path(artifacts), self.path(reports)
        self.digests, self.file_stats, self.mapped, self.checks = {}, {}, [], {}
        self.model_arrays, self.catalogues = {}, {}
        self.engine_cases = []

    def path(self, value):
        path = Path(value)
        path = (path if path.is_absolute() else self.root / path).resolve()
        require(path.is_relative_to(self.root), "audit input path escapes project root")
        return path

    def key(self, value):
        return self.path(value).relative_to(self.root).as_posix()

    def digest(self, value, expected=None):
        path, key = self.path(value), self.key(value)
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        if key in self.digests:
            require(self.file_stats[key] == fingerprint, f"file changed during audit: {key}")
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    hasher.update(block)
            require((path.stat().st_size, path.stat().st_mtime_ns) == fingerprint,
                    f"file changed while hashing: {key}")
            self.digests[key], self.file_stats[key] = hasher.hexdigest(), fingerprint
        if expected is not None:
            require(self.digests[key] == expected, f"SHA256 mismatch: {key}")
        return self.digests[key]

    def read(self, value, expected=None):
        self.digest(value, expected)
        return json.loads(self.path(value).read_text(encoding="utf-8"))

    def array(self, value, expected=None):
        self.digest(value, expected)
        result = np.load(self.path(value), mmap_mode="r", allow_pickle=False)
        self.mapped.append(result)
        return result

    def close(self):
        for value in self.mapped:
            mapping = getattr(value, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    def source_group(self, row):
        lo, hi = map(int, self.offsets[row:row + 2])
        return np.asarray(self.candidates[lo:hi]), np.asarray(self.labels[lo:hi])

    def expected_rows(self, phase):
        if phase == "dev":
            population = np.flatnonzero(self.split == 2)
            count = 5000
        else:
            low, high, count = WINDOWS[phase]
            population = np.flatnonzero((self.split == 1) & (self.timestamps >= low)
                                       & (self.timestamps < high))
        require(len(population) >= count, f"population too small: {phase}")
        return np.sort(np.random.default_rng(2026).choice(population, count, replace=False))

    def row_summary(self, rows):
        return {"count": len(rows), "row_sha256": row_hash(rows),
                "timestamp_min": int(self.timestamps[rows].min()),
                "timestamp_max": int(self.timestamps[rows].max()),
                "split_values": np.unique(self.split[rows]).astype(int).tolist(),
                "sampling_seed": 2026}

    def initialize(self):
        self.fit = self.read(self.reports / "fit.json")
        self.fit_digest = self.digest(self.reports / "fit.json")
        self.selection = self.read(self.artifacts / "selection.json")
        self.selection_digest = self.digest(self.artifacts / "selection.json")
        self.v2 = self.read("artifacts/v2/selection.json", self.selection["v2_selection_sha256"])
        required_sources = {"docs/V3_PROTOCOL.md", "newsforge/v3/experiment.py",
                            "newsforge/v3/residual.py", "newsforge/v3/fast_retrieval.py",
                            "newsforge/v2/ranker.py", "newsforge/v2/temporal.py",
                            "newsforge/v2/retrieval.py", "newsforge/v2/experiment.py",
                            "newsforge/v2/serve.py", "newsforge/store.py", "newsforge/metrics.py"}
        require(required_sources <= set(self.fit["sources"]), "incomplete source binding")
        for path, expected in self.fit["sources"].items():
            self.digest(path, expected)
        for path, expected in self.fit["input_files"].items():
            self.digest(path, expected)
        for name in DATA_FILES:
            path = self.data / name
            expected = (self.v2["bindings"]["data_manifest"] if name == "manifest.json"
                        else self.v2["bindings"]["processed_files"][name])
            require(self.fit["input_files"].get(self.key(path)) == expected,
                    f"actual input absent from fit binding: {name}")
            self.digest(path, expected)
        for name in ("history", "candidates", "labels", "offsets", "timestamps", "users", "split"):
            setattr(self, name, self.array(self.data / f"all_{name}.npy"))
        self.first_seen = self.array(self.data / "news_first_seen.npy")
        self.features = self.array(self.data / "news_features.npy")
        require(self.history.shape == (len(self.split), 20), "history representation changed")
        manifest = self.read(self.temporal / "manifest.json", self.v2["bindings"]["temporal_manifest"])
        self.temporal_manifest = manifest
        for name, record in manifest["sources"].items():
            self.digest(self.data / name, record["sha256"])
        for name, record in manifest["arrays"].items():
            array = self.array(self.temporal / name, record["sha256"])
            require(list(array.shape) == record["shape"] and str(array.dtype) == record["dtype"],
                    f"temporal shape/dtype: {name}")
            if name in ("snapshot_times.npy", "snapshot_clicks.npy", "snapshot_exposures.npy"):
                setattr(self, name[:-4], array)
        require(manifest["policy"]["feedback_delay_seconds"] == 3600, "feedback lag changed")
        require(self.selection["sources"] == self.fit["sources"]
                and self.selection["fit_report_sha256"] == self.fit_digest
                and self.selection["models"] == self.fit["models"], "selection/fit mismatch")
        config = self.v2["serving"]["candidate_config"]
        require(not self.v2["serving"]["use_reranker"] and config["channels"] == ["popular"]
                and config["pool_size"] == 100 and config["route_size"] == 200
                and config["rank_constant"] == 60 and config["recent_hours"] == 24
                and not config["freeze_feedback"], "inherited baseline configuration changed")
        self.checks["sources_and_inputs"] = {"all_bound_files_match": True,
                                            "source_files": len(self.fit["sources"]),
                                            "input_files": len(self.fit["input_files"])}

    def fit_pairs_and_models(self):
        rows = self.array(self.artifacts / "fit/rows.npy")
        require(np.array_equal(rows, self.expected_rows("fit")), "fit sampling mismatch")
        require(self.fit["fit_rows"] == self.row_summary(rows), "fit row summary mismatch")
        start = self.read(self.artifacts / "fit/start.json")
        require(start == {"sources": self.fit["sources"], "inputs": self.fit["input_files"],
                          "environment": self.fit["environment"]}, "fit start binding mismatch")
        fit_groups = {}
        for row in rows:
            ids, labels = self.source_group(int(row))
            require(np.isin(labels, [0, 1]).all() and np.all(ids > 0), "invalid source labels/IDs")
            positive = ids[labels == 1].copy()
            positive_set = set(positive.tolist())
            negative = np.asarray([item for item, label in zip(ids, labels)
                                   if label == 0 and item not in positive_set], dtype=ids.dtype)
            fit_groups[int(row)] = positive, negative
        pair_counts = {}
        for seed in SEEDS:
            run = self.fit["runs"][str(seed)]
            pairs = self.array(self.artifacts / f"fit/pairs-{seed}.npy", run["pairs_sha256"])
            feature_cache = self.array(self.artifacts / f"fit/features-{seed}.npy", run["pair_features_sha256"])
            require(pairs.dtype == np.dtype("int64") and pairs.ndim == 2 and pairs.shape[1] == 3,
                    "invalid pair array")
            require(feature_cache.shape == (len(pairs), 2, 13)
                    and feature_cache.dtype == np.dtype("float32") and np.isfinite(feature_cache).all(),
                    "invalid fit feature cache")
            rng, cursor, no_positive, no_negative = np.random.default_rng(seed), 0, 0, 0
            for row in rows:
                positive, negative = fit_groups[int(row)]
                if not len(positive):
                    no_positive += 1
                    continue
                if not len(negative):
                    no_negative += 1
                    continue
                for item in positive:
                    actual = np.asarray(pairs[cursor:cursor + 4])
                    expected_negative = negative[rng.integers(0, len(negative), size=4)]
                    require(actual.shape == (4, 3) and np.all(actual[:, 0] == row)
                            and np.all(actual[:, 1] == item)
                            and np.array_equal(actual[:, 2], expected_negative),
                            f"same-exposure pair sampling mismatch at seed={seed}, row={row}")
                    cursor += 4
            require(cursor == len(pairs), "extra or missing fit pairs")
            sampling = run["sampling"]
            require(sampling["requests"] == len(rows) and sampling["pairs"] == len(pairs)
                    and sampling["seed"] == seed and sampling["replacement"] is True
                    and sampling["negative_draws_per_positive"] == 4
                    and sampling["skipped_no_positive"] == no_positive
                    and sampling["skipped_no_negative"] == no_negative,
                    "sampling report mismatch")
            feature_sum = np.zeros(13, dtype=np.float64)
            for begin in range(0, len(pairs), 4096):
                block = np.asarray(feature_cache[begin:begin + 4096], dtype=np.float64).reshape(-1, 13)
                feature_sum += block.sum(axis=0)
            expected_mean = feature_sum / (2 * len(pairs))
            squared_sum = np.zeros(13, dtype=np.float64)
            for begin in range(0, len(pairs), 4096):
                block = np.asarray(feature_cache[begin:begin + 4096], dtype=np.float64).reshape(-1, 13)
                squared_sum += np.square(block - expected_mean).sum(axis=0)
            expected_scale = np.sqrt(squared_sum / (2 * len(pairs)))
            expected_scale[expected_scale == 0] = 1.
            full_stats = None
            for variant, columns in VARIANTS.items():
                record = run["models"][variant]
                path = self.path(record["path"])
                require(self.fit["models"][variant][str(seed)] == record["path"], "model path mismatch")
                self.digest(path, record["npz_sha256"])
                metadata = self.read(str(path) + ".json", record["sidecar_sha256"])
                require(metadata["npz_sha256"] == record["npz_sha256"]
                        and metadata["source_sha256"] == self.digest("newsforge/v3/residual.py")
                        and metadata["variant"] == variant and metadata["feature_indices"] == list(columns)
                        and metadata["fit_metadata"] == record["fit_metadata"], "model sidecar mismatch")
                bound = metadata["bindings"]
                require(bound == {"sources": self.fit["sources"], "input_files": self.fit["input_files"],
                                  "fit_rows": self.fit["fit_rows"], "seed": seed,
                                  "pairs_sha256": run["pairs_sha256"],
                                  "pair_features_sha256": run["pair_features_sha256"],
                                  "environment": self.fit["environment"]}, "model training inputs mismatch")
                fit_meta = metadata["fit_metadata"]
                require(fit_meta["optimizer_success"] is True and fit_meta["optimizer"] == "L-BFGS-B"
                        and fit_meta["max_iterations"] == 200 and fit_meta["l2"] == .01
                        and fit_meta["pairs"] == len(pairs)
                        and fit_meta["pair_weighting"] == "equal weight per supplied pair", "model fit policy")
                with np.load(path, allow_pickle=False) as saved:
                    require(set(saved.files) == {"mean", "scale", "weights"}, "unexpected model arrays")
                    model = {name: saved[name].copy() for name in saved.files}
                require(all(value.dtype == np.dtype("float64") and value.shape == (len(columns),)
                            and np.isfinite(value).all() for value in model.values())
                        and np.all(model["scale"] > 0), "invalid model parameter arrays")
                close(model["mean"], expected_mean[list(columns)], "fit-only population mean", 1e-10)
                close(model["scale"], expected_scale[list(columns)], "fit-only population std", 1e-10)
                if full_stats is None:
                    full_stats = model
                else:
                    close(model["mean"], full_stats["mean"][list(columns)], "shared column means", 1e-12)
                    close(model["scale"], full_stats["scale"][list(columns)], "shared column scales", 1e-12)
                logistic_sum = 0.
                for begin in range(0, len(pairs), 4096):
                    block = np.asarray(feature_cache[begin:begin + 4096], dtype=np.float64)
                    difference = (block[:, 0, columns] - block[:, 1, columns]) / model["scale"]
                    logistic_sum += float(np.logaddexp(0., -(difference @ model["weights"])).sum())
                loss = logistic_sum / len(pairs) + .005 * np.square(model["weights"]).sum()
                close(loss, fit_meta["loss"], "independent fitted objective", 1e-10)
                self.model_arrays[(variant, seed)] = model
                self.digest(path, self.selection["model_files"][record["path"]])
                self.digest(str(path) + ".json", self.selection["model_files"][record["path"] + ".json"])
            pair_counts[str(seed)] = len(pairs)
            print(f"audit fit pairs and models seed={seed}: {len(pairs)} pairs", flush=True)
        self.checks["fit_pairs_and_normalization"] = {
            "fit_requests": len(rows), "fit_row_sha256": row_hash(rows), "pairs_per_seed": pair_counts,
            "same_exposure_labelled_pairs_and_rng_reproduced": True,
            "four_draws_per_positive_with_replacement": True,
            "all_nine_models_share_their_seed_column_statistics": True,
            "fit_only_population_normalization_and_objective_recomputed": True}

    def canonical_candidates(self, at, history):
        start, cutoff = (int(at) // 3600) * 3600, (int(at) // 3600) * 3600 - 3600
        if start not in self.catalogues:
            matches = np.flatnonzero(self.snapshot_times == start)
            require(len(matches) == 1, "snapshot hour missing")
            slot = int(matches[0])
            visible = np.flatnonzero((self.first_seen >= 0) & (self.first_seen < cutoff))
            visible = visible[visible > 0]
            recent = visible[self.first_seen[visible] >= start - 24 * 3600]
            clicks, exposures = self.snapshot_clicks[slot], self.snapshot_exposures[slot]
            def popularity(ids):
                return ids[np.lexsort((ids, -self.first_seen[ids], -exposures[ids], -clicks[ids]))]
            self.catalogues[start] = (popularity(visible), popularity(recent))
        global_ids, recent_ids = self.catalogues[start]
        blocked = set(int(item) for item in history if item > 0)
        chosen = []
        for item in recent_ids:
            if int(item) not in blocked:
                chosen.append(int(item))
                if len(chosen) == 100:
                    break
        if len(chosen) < 100:
            blocked.update(chosen)
            for item in global_ids:
                if int(item) not in blocked:
                    chosen.append(int(item))
                    if len(chosen) == 100:
                        break
        return np.asarray(chosen, dtype=np.int64)

    def stage_cache(self, phase):
        report = self.read(self.reports / f"{phase}.json")
        require(report["phase"] == phase and report["environment"] == self.fit["environment"], "stage identity")
        rows = self.expected_rows(phase)
        expected_bindings = {"phase": phase, "rows": self.row_summary(rows),
                             "fit_report_sha256": self.fit_digest, "sources": self.fit["sources"]}
        if phase != "tune":
            expected_bindings["selection_sha256"] = self.selection_digest
        require(report["bindings"] == expected_bindings, "stage row/source bindings")
        directory = self.artifacts / phase
        manifest = self.read(directory / "cache.json", report["cache_manifest_sha256"])
        require(manifest["bindings"] == expected_bindings
                and set(manifest["arrays"]) == {name + ".npy" for name in CACHE_FILES}, "cache schema/bindings")
        cache = {name: self.array(directory / (name + ".npy"), manifest["arrays"][name + ".npy"])
                 for name in CACHE_FILES}
        n = len(rows)
        shapes = {"rows": (n,), "timestamps": (n,), "users": (n,), "denominators": (n, 3),
                  "history_length": (n,), "lengths": (n,), "ids": (n, 100), "hit": (n, 100),
                  "features": (n, 100, 13), "clipped_scores": (n, 3, 3, 100)}
        require(all(cache[name].shape == shape for name, shape in shapes.items()), "cache shapes")
        require(np.array_equal(cache["rows"], rows)
                and np.array_equal(cache["timestamps"], self.timestamps[rows])
                and np.array_equal(cache["users"], self.users[rows]), "cache identities")
        runtime = report["runtime"]
        require(runtime["temporal_manifest_sha256"] == self.digest(self.temporal / "manifest.json")
                and runtime["verified_temporal_array_hashes"] == self.temporal_manifest["arrays"],
                "runtime temporal binding")
        for p in np.argsort(self.timestamps[rows], kind="stable"):
            row = rows[p]
            ids, labels = self.source_group(int(row))
            positive = set(ids[labels == 1].astype(int).tolist())
            history, at = self.history[row], int(self.timestamps[row])
            seen, cutoff = set(history[history > 0].astype(int).tolist()), (at // 3600) * 3600 - 3600
            expected_ids = self.canonical_candidates(at, history)
            m = int(cache["lengths"][p])
            require(m == len(expected_ids) and np.array_equal(cache["ids"][p, :m], expected_ids),
                    f"canonical popularity candidates differ: {phase} row={row}")
            require(np.all(cache["ids"][p, m:] == 0) and not np.any(cache["hit"][p, m:])
                    and np.all(cache["features"][p, m:] == 0)
                    and np.all(cache["clipped_scores"][p, :, :, m:] == 0), "nonzero cache padding")
            expected_hits = np.asarray([int(item) in positive for item in expected_ids], dtype=bool)
            require(np.array_equal(cache["hit"][p, :m], expected_hits), "cached click hits differ from source logs")
            eligible = {item for item in positive if self.first_seen[item] < cutoff and item not in seen}
            require(np.array_equal(cache["denominators"][p], [len(positive), len(eligible), len(positive & seen)]),
                    "cached click denominators differ from source logs")
            history_length = int(np.any(self.features[history[-20:]] != 0, axis=1).sum())
            require(cache["history_length"][p] == history_length, "cached effective history length")
            feature_values = np.asarray(cache["features"][p, :m], dtype=np.float64)
            require(np.isfinite(feature_values).all(), "nonfinite cached content features")
            for vi, (variant, columns) in enumerate(VARIANTS.items()):
                for si, seed in enumerate(SEEDS):
                    model = self.model_arrays[(variant, seed)]
                    prediction = ((feature_values[:, columns] - model["mean"]) / model["scale"]) @ model["weights"]
                    close(cache["clipped_scores"][p, vi, si, :m], np.clip(prediction, -1., 1.),
                          "independent model-to-cache prediction", 1e-12)
        require(report["all_clicks"] == int(cache["denominators"][:, 0].sum())
                and report["eligible_clicks"] == int(cache["denominators"][:, 1].sum())
                and report["seen_clicks"] == int(cache["denominators"][:, 2].sum())
                and report["shortfall_requests"] == int((cache["lengths"] < 100).sum()), "stage aggregate counts")
        return report, cache

    def ranked_counts(self, cache, delta, all_ablations):
        methods = {"full13": (0, None, True)}
        if all_ablations:
            methods.update({"no_category": (1, None, True), "no_recent": (2, None, True),
                            "no_gate": (0, None, False)})
            methods.update({f"full13_seed{seed}": (0, slot, True) for slot, seed in enumerate(SEEDS)})
        n = len(cache["rows"])
        hits = {name: np.zeros(n, dtype=np.int64) for name in ("baseline", *methods)}
        hit100 = np.zeros(n, dtype=np.int64)
        changed = {name: 0 for name in hits}
        for p in range(n):
            m, length = int(cache["lengths"][p]), int(cache["history_length"][p])
            clicked = cache["hit"][p, :m]
            hits["baseline"][p], hit100[p] = clicked[:10].sum(), clicked.sum()
            for name, (variant, seed_slot, gate) in methods.items():
                score = np.asarray(cache["clipped_scores"][p, variant, :, :m])
                if seed_slot is not None:
                    score = score[seed_slot:seed_slot + 1]
                order = independent_order(score, length, delta, gate)
                hits[name][p] = clicked[order[:10]].sum()
                require(clicked[order].sum() == hit100[p], "R100 changed after independent sorting")
                changed[name] += int(not np.array_equal(order, np.arange(m)))
        invariants = {"per_request_candidate_and_hit100_invariant": True,
                      "empty_history_exact_fallback_requests": int((cache["history_length"] == 0).sum()),
                      "changed_order_requests": changed}
        return hits, hit100, invariants

    def compare_interval(self, actual, expected):
        close(actual["delta"], expected["delta"], "bootstrap point delta")
        close(actual["ci95"], expected["ci95"], "independent user-cluster interval")
        require(all(actual[name] == expected[name] for name in ("resampling_unit", "replicates", "seed")),
                "bootstrap metadata mismatch")

    def check_metrics(self, actual, cache, hits, hit100, intervals):
        require(set(actual) == set(hits), "unexpected metric methods")
        denominator = cache["denominators"][:, 0]
        total = int(denominator.sum())
        require(total > 0, "no click denominator")
        for name, values in hits.items():
            value = actual[name]
            require(value["hits10"] == int(values.sum()) and value["hits100"] == int(hit100.sum()),
                    f"hit aggregates differ: {name}")
            close(value["recall10"], values.sum() / total, f"micro R10: {name}")
            close(value["recall100"], hit100.sum() / total, f"micro R100: {name}")
            if intervals and name != "baseline":
                expected = independent_interval(hits["baseline"], values, denominator, cache["users"])
                self.compare_interval(value["versus_baseline"], expected)
        if intervals:
            for name in ("no_category", "no_recent", "no_gate"):
                if name in hits:
                    expected = independent_interval(hits[name], hits["full13"], denominator, cache["users"])
                    self.compare_interval(actual["full13"]["versus_" + name], expected)

    def check_slices(self, reported, cache, hits):
        length, times = cache["history_length"], cache["timestamps"]
        masks = {"empty": length == 0, "history_1_5": (length >= 1) & (length <= 5),
                 "history_6_20": (length >= 6) & (length <= 20)}
        masks.update({f"source_6h_block_{int(block)}": times // 21600 == block
                      for block in np.unique(times // 21600)})
        require(set(reported) == set(masks), "unexpected diagnostic slices")
        for name, mask in masks.items():
            value, denominator = reported[name], int(cache["denominators"][mask, 0].sum())
            require(value["requests"] == int(mask.sum()) and value["all_clicks"] == denominator,
                    "slice denominators mismatch")
            require(set(value["recall10"]) == set(hits), "slice methods mismatch")
            for method, counts in hits.items():
                if denominator:
                    close(value["recall10"][method], counts[mask].sum() / denominator, "slice micro recall")
                else:
                    require(value["recall10"][method] is None, "empty slice must have null metric")

    def audit_stages(self):
        tune, cache = self.stage_cache("tune")
        require(set(tune["delta_grid"]) == {str(delta) for delta in DELTAS}, "delta grid changed")
        total_hits = {}
        for delta in DELTAS:
            hits, hit100, invariants = self.ranked_counts(cache, delta, False)
            record = tune["delta_grid"][str(delta)]
            self.check_metrics(record["metrics"], cache, hits, hit100, False)
            require(record["invariants"] == invariants, "tune ranking invariants")
            total_hits[delta] = int(hits["full13"].sum())
        selected = min(DELTAS, key=lambda delta: (-total_hits[delta], delta))
        require(tune["selected_delta"] == self.selection["delta"] == selected
                and self.selection["tune_cache_manifest_sha256"] == tune["cache_manifest_sha256"]
                and self.selection["primary_recipe"] == "full13+gate"
                and self.selection["confirmation_gate"] == "delta_R10>0 and paired_user_bootstrap_ci95_lower>0",
                "tune winner or predeclared confirmation policy changed")
        self.checks["tune"] = {"requests": len(cache["rows"]), "selected_delta": selected,
                               "all_four_deltas_independently_recomputed": True,
                               "all_primary_ties_prefer_smaller_delta": True}
        print(f"audit tune: independently selected delta={selected}", flush=True)
        confirmation_gate = None
        for phase in ("confirm", "dev"):
            report, cache = self.stage_cache(phase)
            require(report["delta"] == selected and report["selection_sha256"] == self.selection_digest
                    and report["tune_report_sha256"] == self.digest(self.reports / "tune.json"),
                    "successful tune-to-stage linkage")
            hits, hit100, invariants = self.ranked_counts(cache, selected, phase == "dev")
            self.check_metrics(report["metrics"], cache, hits, hit100, True)
            self.check_slices(report["slices"], cache, hits)
            require(report["invariants"] == invariants, "stage ranking invariants differ")
            counts_path = self.artifacts / phase / "counts.npz"
            self.digest(counts_path, report["counts_sha256"])
            with np.load(counts_path, allow_pickle=False) as saved:
                require(set(saved.files) == set(hits) | {"hit100"}, "saved count methods mismatch")
                for name, values in {**hits, "hit100": hit100}.items():
                    require(np.array_equal(saved[name], values), f"saved {phase} {name} counts mismatch")
            if phase == "confirm":
                expected = independent_interval(hits["baseline"], hits["full13"],
                                                cache["denominators"][:, 0], cache["users"])
                confirmation_gate = bool(expected["delta"] > 0 and expected["ci95"][0] > 0)
                require(type(report["deployment_gate_passed"]) is bool
                        and report["deployment_gate_passed"] == confirmation_gate
                        and report["deployment_available_as_of"] == 1573776000,
                        "confirmation gate or maturity differs")
            else:
                require(type(report["deployment_gate_passed"]) is bool
                        and report["deployment_gate_passed"] == confirmation_gate
                        and report["deployed_recipe"] == ("full13+gate" if confirmation_gate else "baseline")
                        and report["confirm_sha256"] == self.digest(self.reports / "confirm.json"),
                        "dev silently changed deployment choice")
                old_path = self.path("artifacts/v2/dev_catalog_counts.npz")
                old_audit = self.read("reports/v2/release_integrity.json")
                require(old_audit["passed"] is True, "v2 release audit did not pass")
                self.digest(old_path, old_audit["verified_sha256_files"][self.key(old_path)])
                with np.load(old_path, allow_pickle=False) as old:
                    require(np.array_equal(old["rows"], cache["rows"])
                            and np.array_equal(old["selected"][:, 0], hits["baseline"])
                            and np.array_equal(old["selected"][:, 1], hit100)
                            and np.array_equal(old["denominators"][:, [0, 1, 4]], cache["denominators"]),
                            "v3 baseline differs from v2 saved selected counts/denominators")
            self.checks[phase] = {"requests": len(cache["rows"]), "methods": list(hits),
                                  "canonical_popularity_candidates_reconstructed": True,
                                  "source_click_denominators_and_hits_reconstructed": True,
                                  "clipped_scores_recomputed_from_model_parameters": True,
                                  "independent_sort_counts_and_bootstrap_intervals_match": True,
                                  "per_request_R100_invariant": True,
                                  "deployment_gate_passed": confirmation_gate}
            if phase == "dev":
                self.checks[phase]["baseline_matches_frozen_v2_per_request_counts"] = True
                self.prepare_engine_cases(cache, selected, confirmation_gate)
            print(f"audit {phase}: {len(hits)} methods, counts and intervals verified", flush=True)

    def prepare_engine_cases(self, cache, delta, deployment_gate):
        """Keep just 64 requests and independently reconstructed expected output."""
        news_ids = self.read(self.data / "news_ids.json")
        positions = np.sort(np.random.default_rng(2026).choice(len(cache["rows"]), 64, replace=False))
        for p in positions:
            row, m = int(cache["rows"][p]), int(cache["lengths"][p])
            history_length = int(cache["history_length"][p])
            enabled = bool(deployment_gate and delta > 0 and history_length > 0 and m > 0)
            clipped = np.asarray(cache["clipped_scores"][p, 0, :, :m])
            order = independent_order(clipped, history_length, delta, True) if enabled else np.arange(m)
            scores = (-np.arange(m, dtype=np.float64)
                      + delta * history_length / (history_length + 5.) * clipped.mean(axis=0)) if enabled else None
            self.engine_cases.append({
                "row": row, "as_of": int(cache["timestamps"][p]),
                "history": [news_ids[int(item)] for item in self.history[row] if item > 0],
                "expected_ids": [news_ids[int(cache["ids"][p, k])] for k in order[:10]],
                "expected_scores": None if scores is None else scores[order[:10]].tolist(),
                "history_length": history_length, "candidate_count": m,
                "residual_applied": enabled, "deployment_gate": bool(deployment_gate)})

    def quality_cache_to_engine(self):
        # Release audited mmap pages before the real service independently loads
        # its own inputs. The 64 cases above contain only ordinary Python values.
        require(len(self.engine_cases) == 64, "engine equivalence cases missing")
        self.close()
        self.catalogues.clear()
        from newsforge.v2.serve import V2RecommendRequest
        from newsforge.v3.serve import load_engine, runtime_sources
        previous_cwd, engine = Path.cwd(), None
        tolerance, max_difference, scores_checked, empty_fallbacks = 1e-5, 0., 0, 0
        try:
            os.chdir(self.root)
            engine = load_engine(processed_dir=self.data, artifact_dir=self.path("artifacts"),
                                 temporal_dir=self.temporal,
                                 selection_path=self.artifacts / "selection.json",
                                 confirm_path=self.reports / "confirm.json",
                                 v2_selection_path=self.path("artifacts/v2/selection.json"),
                                 fit_report_path=self.reports / "fit.json")
            for name, expected in runtime_sources().items():
                self.digest(self.root / "newsforge" / name, expected)
            for case in self.engine_cases:
                response = engine.recommend(V2RecommendRequest(history=case["history"],
                                                               as_of=case["as_of"], top_k=10))
                require([item["item_id"] for item in response["items"]] == case["expected_ids"],
                        f"offline quality cache/service ID or order mismatch at row={case['row']}")
                require(response["candidate_count"] == case["candidate_count"]
                        and response["history_length_used"] == case["history_length"]
                        and response["residual_applied"] == case["residual_applied"]
                        and response["deployment_gate_passed"] == case["deployment_gate"],
                        "service quality configuration mismatch")
                actual = [item["rerank_score"] for item in response["items"]]
                if case["expected_scores"] is None:
                    require(all(value is None for value in actual), "baseline service unexpectedly applies residual")
                    empty_fallbacks += int(case["history_length"] == 0)
                else:
                    close(actual, case["expected_scores"], "offline quality/service rank scores", tolerance)
                    difference = np.abs(np.asarray(actual) - np.asarray(case["expected_scores"]))
                    max_difference = max(max_difference, float(difference.max(initial=0.)))
                    scores_checked += len(actual)
            version = engine.model_version
        finally:
            if engine is not None:
                engine.close()
            os.chdir(previous_cwd)
        self.checks["offline_quality_cache_to_service"] = {
            "queries": 64, "sampling_seed": 2026,
            "sampled_row_sha256": row_hash([case["row"] for case in self.engine_cases]),
            "exact_top10_id_and_order_matches": 64, "rank_scores_compared": scores_checked,
            "max_absolute_rank_score_difference": max_difference,
            "absolute_score_tolerance": tolerance,
            "score_tolerance_reason": "Float32 content reductions may differ between padded 20-slot offline histories and unpadded service histories; IDs and order must still match exactly.",
            "empty_history_null_score_fallbacks": empty_fallbacks,
            "deployed_model_version": version,
            "scope": "Independent cached-score ranking versus actual loaded V3Engine; HTTP transport is tested separately."}
        print("audit quality cache -> V3Engine: 64 exact ID/order matches", flush=True)

    def publication_evidence(self):
        service = self.read(self.reports / "service_benchmark.json")
        require(service["passed"] is True, "HTTP service verification did not pass")
        for name, expected in service["source_sha256"].items():
            self.digest(self.root / "newsforge" / name, expected)
        service_inputs = {"selection": self.artifacts / "selection.json",
                          "confirm": self.reports / "confirm.json", "fit_report": self.reports / "fit.json",
                          "v2_selection": self.path("artifacts/v2/selection.json")}
        require(set(service["release_evidence_sha256"]) == set(service_inputs), "HTTP release bindings")
        for name, path in service_inputs.items():
            self.digest(path, service["release_evidence_sha256"][name])
        metadata = self.read(self.reports / "v3_results.metadata.json")
        self.digest(metadata["plot_source"]["path"], metadata["plot_source"]["sha256"])
        for record in [*metadata["inputs"].values(), *metadata["outputs"].values()]:
            require(Path(record["file_name"]).name == record["file_name"], "invalid plot evidence filename")
            self.digest(self.reports / record["file_name"], record["sha256"])
        require(metadata["plotted_data"]["selection_sha256"] == self.selection_digest, "plot selection binding")
        test_path = self.path("reports/v3_release_unit_tests.txt")
        self.digest(test_path)
        test_log = test_path.read_text(encoding="utf-8")
        result = re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK\s*$", test_log)
        require(result is not None and int(result.group(1)) > 0, "unit-test release log lacks successful completion")
        test_sources = sorted(self.path("tests").glob("test_*.py"))
        for path in test_sources:
            self.digest(path)
        self.checks["publication_evidence"] = {
            "http_benchmark_passed_and_release_bound": True,
            "plot_source_inputs_and_outputs_sha256_match": True,
            "unit_test_log_passed_count": int(result.group(1)),
            "unit_test_log_observed_seconds": float(result.group(2)),
            "test_source_files_bound": len(test_sources),
            "note": "These existing HTTP, plot and unit-test artifacts are hash-bound; this audit does not rerun the HTTP benchmark or unit suite. Mutable README/resume/result prose is intentionally excluded."}

    def run(self):
        started = time.perf_counter()
        self.digest(Path(__file__))
        self.initialize()
        self.fit_pairs_and_models()
        self.audit_stages()
        self.quality_cache_to_engine()
        self.publication_evidence()
        for key in list(self.digests):
            self.digest(key)
        return {"passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Independent source-pair, parameter, candidate, saved-score, ranking, bootstrap and stage-decision audit; no fitting or new tuning",
                "checks": self.checks, "verified_sha256_files": dict(sorted(self.digests.items())),
                "elapsed_seconds": time.perf_counter() - started,
                "limits": ["Content feature extraction is hash-bound to the frozen implementation and caches, not independently reimplemented here.",
                           "This audit verifies the released evidence chain; it does not turn an observed benchmark into an untouched test set.",
                           "Bootstrap intervals remain conditional on the observed day and feedback stream; multiple dev ablation intervals are exploratory."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--temporal", default="data/temporal_v2")
    parser.add_argument("--artifacts", default="artifacts/v3")
    parser.add_argument("--reports", default="reports/v3")
    args = parser.parse_args()
    audit = Audit(args.root, args.data, args.temporal, args.artifacts, args.reports)
    output = audit.reports / "release_integrity.json"
    if output.exists():
        raise FileExistsError("release audit already exists; preserve prior evidence")
    try:
        with threadpool_limits(limits=1):
            result = audit.run()
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    finally:
        audit.close()
    print(json.dumps({"passed": result["passed"], "output": audit.key(output),
                      "verified_files": len(result["verified_sha256_files"]),
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
