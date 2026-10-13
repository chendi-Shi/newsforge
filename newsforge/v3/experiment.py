"""Time-separated fit, tune, confirm and frozen catalog ablations.

Run stages in that order. All evidence is append-only: a failed stage must be
retained and rerun under a new output root after documenting the correction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from numpy.lib.format import open_memmap
from threadpoolctl import threadpool_limits

from newsforge.store import Store
from newsforge.v2.experiment import recall_interval, row_hash
from newsforge.v2.ranker import content_features
from newsforge.v2.retrieval import sha256_file
from newsforge.v2.serve import V2ServiceConfig, verify_generator_selection
from newsforge.v2.temporal import TemporalStore
from .fast_retrieval import FastCandidateGenerator
from .residual import PairwiseResidual, VARIANTS

SEEDS = (17, 29, 43)
DELTA_GRID = (0, 2, 5, 10)
WINDOWS = {"fit": (1573722000, 1573736400, 20000),
           "tune": (1573740000, 1573754400, 2000),
           "confirm": (1573758000, 1573772400, 2000)}
SOURCE_PATHS = ("docs/V3_PROTOCOL.md", "newsforge/store.py", "newsforge/metrics.py",
                "newsforge/v2/experiment.py", "newsforge/v2/ranker.py",
                "newsforge/v2/retrieval.py", "newsforge/v2/serve.py",
                "newsforge/v2/temporal.py", "newsforge/v3/experiment.py",
                "newsforge/v3/residual.py", "newsforge/v3/fast_retrieval.py")
DATA_FILES = ("manifest.json", "news_features.npy", "news_first_seen.npy",
              "news_train_clicks.npy", "news_train_exposures.npy", "news_ids.json",
              "all_history.npy", "all_candidates.npy", "all_labels.npy", "all_offsets.npy",
              "all_timestamps.npy", "all_users.npy", "all_split.npy")


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sources():
    return {path: sha256_file(path) for path in SOURCE_PATHS}


def verify_hashes(records):
    for path, expected in records.items():
        if sha256_file(path) != expected:
            raise ValueError(f"SHA256 mismatch: {path}")


def fixed_sample_rows(store, phase):
    if phase not in WINDOWS:
        raise ValueError("unknown internal phase")
    low, high, count = WINDOWS[phase]
    population = np.flatnonzero((store.split == 1) & (store.timestamps >= low)
                               & (store.timestamps < high))
    if len(population) < count:
        raise ValueError("insufficient fixed-window population")
    return np.sort(np.random.default_rng(2026).choice(population, count, replace=False))


def sample_pairs(store, rows, seed):
    rng = np.random.default_rng(seed)
    pairs, skipped_no_positive, skipped_no_negative = [], 0, 0
    for row in rows:
        ids, labels = store.candidates_for(int(row))
        ids, labels = np.asarray(ids), np.asarray(labels)
        if np.any((labels != 0) & (labels != 1)) or np.any(ids <= 0):
            raise ValueError("invalid logged exposure group")
        positive = ids[labels == 1]
        negative = ids[(labels == 0) & ~np.isin(ids, positive)]
        if not len(positive):
            skipped_no_positive += 1
            continue
        if not len(negative):
            skipped_no_negative += 1
            continue
        for item in positive:
            for other in rng.choice(negative, 4, replace=True):
                pairs.append((int(row), int(item), int(other)))
    if not pairs:
        raise ValueError("no valid same-exposure pairs")
    return np.asarray(pairs, dtype=np.int64), {
        "requests": len(rows), "pairs": len(pairs), "negative_draws_per_positive": 4,
        "replacement": True, "seed": int(seed),
        "skipped_no_positive": skipped_no_positive, "skipped_no_negative": skipped_no_negative,
        "weighting": "equal pairs; repeated draws and multi-click requests have greater weight"}


def features_for(store, rows, candidate_ids):
    history = np.asarray(store.history[rows, -20:])
    return content_features(store.features[history], store.features[candidate_ids],
                            history_mask=history > 0,
                            text_dimensions=int(store.manifest["features"]["text_dimensions"]))[..., :13]


def order_from_residual(residual, history_length, delta, gate=True):
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 1 or not np.isfinite(residual).all() or np.any(np.abs(residual) > 1):
        raise ValueError("expected clipped ensemble residual")
    if (isinstance(history_length, (bool, np.bool_)) or not isinstance(history_length, (int, np.integer))
            or not 0 <= history_length <= 20):
        raise ValueError("history length must be an integer in [0,20]")
    if not isinstance(gate, (bool, np.bool_)) or isinstance(delta, (bool, np.bool_)):
        raise ValueError("invalid ranking options")
    if not np.isscalar(delta) or not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    factor = 0. if history_length == 0 else history_length / (history_length + 5.) if gate else 1.
    rank = np.arange(len(residual))
    scores = -rank.astype(np.float64) + delta * factor * residual
    return np.lexsort((rank, -scores)), scores


def save_cache_manifest(directory, bindings):
    directory = Path(directory)
    records = {path.name: sha256_file(path) for path in sorted(directory.glob("*.npy"))}
    value = {"arrays": records, "bindings": bindings}
    dump_new(directory / "cache.json", value)
    return value


def load_cache(directory, expected_bindings):
    directory = Path(directory)
    manifest = read_json(directory / "cache.json")
    if manifest["bindings"] != expected_bindings:
        raise ValueError("cache bindings mismatch")
    arrays = {}
    for name, expected in manifest["arrays"].items():
        if Path(name).name != name or not name.endswith(".npy"):
            raise ValueError("invalid cache array name")
        if sha256_file(directory / name) != expected:
            raise ValueError("cache array SHA256 mismatch")
        arrays[Path(name).stem] = np.load(directory / name, mmap_mode="r", allow_pickle=False)
    return arrays


def environment():
    return {"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        "native_threads": 1, "platform": platform.platform()}


def input_hashes(args):
    result = {(Path(args.data) / name).as_posix(): sha256_file(Path(args.data) / name) for name in DATA_FILES}
    selection = read_json(args.v2_selection)
    for name in DATA_FILES:
        expected = selection["bindings"]["data_manifest"] if name == "manifest.json" else selection["bindings"]["processed_files"][name]
        if result[(Path(args.data) / name).as_posix()] != expected:
            raise ValueError(f"input differs from frozen v2: {name}")
    result[Path(args.v2_selection).as_posix()] = sha256_file(args.v2_selection)
    return result


def row_summary(store, rows):
    return {"count": len(rows), "row_sha256": row_hash(rows),
            "timestamp_min": int(store.timestamps[rows].min()),
            "timestamp_max": int(store.timestamps[rows].max()),
            "split_values": np.unique(store.split[rows]).astype(int).tolist(),
            "sampling_seed": 2026}


def close_arrays(arrays):
    for array in arrays:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def fit(args):
    started = time.perf_counter()
    if (Path(args.reports) / "fit.json").exists():
        raise FileExistsError("fit report already exists")
    output = Path(args.artifacts) / "fit"
    output.mkdir(parents=True, exist_ok=False)
    frozen_sources, inputs, env = sources(), input_hashes(args), environment()
    dump_new(output / "start.json", {"sources": frozen_sources, "inputs": inputs, "environment": env})
    store = Store(args.data)
    if store.history.shape[1] != 20:
        raise ValueError("v3 requires the inherited 20-slot padded history representation")
    rows = fixed_sample_rows(store, "fit")
    np.save(output / "rows.npy", rows)
    row_info = row_summary(store, rows)
    records, model_paths = {}, {variant: {} for variant in VARIANTS}
    for seed in SEEDS:
        tick = time.perf_counter()
        pairs, pair_info = sample_pairs(store, rows, seed)
        pair_path = output / f"pairs-{seed}.npy"
        np.save(pair_path, pairs)
        feature_path = output / f"features-{seed}.npy"
        features = open_memmap(feature_path, mode="w+", dtype="float32", shape=(len(pairs), 2, 13))
        for start in range(0, len(pairs), 256):
            batch = pairs[start:start + 256]
            features[start:start + len(batch)] = features_for(store, batch[:, 0], batch[:, 1:])
        features.flush()
        bindings = {"sources": frozen_sources, "input_files": inputs, "fit_rows": row_info,
                    "seed": seed, "pairs_sha256": sha256_file(pair_path),
                    "pair_features_sha256": sha256_file(feature_path), "environment": env}
        per_seed = {}
        for variant in VARIANTS:
            model = PairwiseResidual(variant).fit(features[:, 0], features[:, 1], bindings=bindings)
            path = Path(args.artifacts) / "models" / f"{variant}-seed{seed}.npz"
            model.save(path)
            model_paths[variant][str(seed)] = path.as_posix()
            per_seed[variant] = {"path": path.as_posix(), "npz_sha256": sha256_file(path),
                                 "sidecar_sha256": sha256_file(str(path) + ".json"),
                                 "fit_metadata": model.fit_metadata}
            print(f"fit seed={seed} variant={variant} pairs={len(pairs)} loss={model.fit_metadata['loss']:.6f}", flush=True)
        records[str(seed)] = {"sampling": pair_info, "models": per_seed,
                              "pairs_sha256": bindings["pairs_sha256"],
                              "pair_features_sha256": bindings["pair_features_sha256"],
                              "elapsed_seconds": time.perf_counter() - tick}
        close_arrays([features])
    verify_hashes(frozen_sources)
    verify_hashes(inputs)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "sources": frozen_sources,
              "input_files": inputs, "environment": env, "fit_rows": row_info,
              "runs": records, "models": model_paths, "elapsed_seconds": time.perf_counter() - started}
    dump_new(Path(args.reports) / "fit.json", report)


def load_models(fit_report):
    models = {}
    for variant in VARIANTS:
        models[variant] = []
        for seed in SEEDS:
            record = fit_report["runs"][str(seed)]["models"][variant]
            path = record["path"]
            verify_hashes({path: record["npz_sha256"], path + ".json": record["sidecar_sha256"]})
            model = PairwiseResidual.load(path)
            if (model.variant != variant or model.bindings["sources"] != fit_report["sources"]
                    or model.bindings["seed"] != seed or model.bindings["input_files"] != fit_report["input_files"]
                    or model.bindings["fit_rows"] != fit_report["fit_rows"]
                    or model.bindings["pairs_sha256"] != fit_report["runs"][str(seed)]["pairs_sha256"]):
                raise ValueError("model/fit report binding mismatch")
            models[variant].append(model)
    return models


def build_cache(args, phase, store, rows, models, bindings):
    directory = Path(args.artifacts) / phase
    directory.mkdir(parents=True, exist_ok=False)
    n = len(rows)
    arrays = {"rows": rows, "timestamps": np.asarray(store.timestamps[rows]),
              "users": np.asarray(store.users[rows]),
              "denominators": np.zeros((n, 3), dtype=np.int32),
              "history_length": np.zeros(n, dtype=np.int32), "lengths": np.zeros(n, dtype=np.int32)}
    maps = {"ids": open_memmap(directory / "ids.npy", mode="w+", dtype="int64", shape=(n, 100)),
            "hit": open_memmap(directory / "hit.npy", mode="w+", dtype="bool", shape=(n, 100)),
            "features": open_memmap(directory / "features.npy", mode="w+", dtype="float32", shape=(n, 100, 13)),
            "clipped_scores": open_memmap(directory / "clipped_scores.npy", mode="w+", dtype="float64", shape=(n, 3, 3, 100))}
    for array in maps.values():
        array[:] = 0
    config = V2ServiceConfig.from_selection(args.v2_selection)
    temporal, generator = None, None
    try:
        temporal = TemporalStore(args.temporal, processed_dir=args.data, verify=True)
        generator = FastCandidateGenerator.from_paths(args.data, "artifacts", temporal,
                         encoder_path=config.encoder_path, config=config.candidate_config)
        verify_generator_selection(config, generator)
        if (config.use_reranker or config.candidate_config.pool_size != 100
                or config.candidate_config.channels != ("popular",) or config.candidate_config.recent_hours != 24):
            raise ValueError("unexpected inherited catalog baseline")
        for progress, position in enumerate(np.argsort(store.timestamps[rows], kind="stable")):
            row = int(rows[position])
            history, at = store.history[row], int(store.timestamps[row])
            batch = generator.retrieve(history, at)
            ids, labels = store.candidates_for(row)
            positive = np.unique(ids[labels == 1])
            seen = history[history > 0]
            eligible = positive[(store.first_seen[positive] < batch.cutoff) & ~np.isin(positive, seen)]
            m = len(batch.ids)
            if m > 100 or len(np.unique(batch.ids)) != m:
                raise ValueError("candidate budget or uniqueness violation")
            features = features_for(store, np.asarray([row]), batch.ids[None])[0]
            arrays["denominators"][position] = [len(positive), len(eligible), int(np.isin(positive, seen).sum())]
            arrays["history_length"][position] = batch.metadata["history_length_used"]
            arrays["lengths"][position] = m
            maps["ids"][position, :m] = batch.ids
            maps["hit"][position, :m] = np.isin(batch.ids, positive)
            maps["features"][position, :m] = features
            for vi, variant in enumerate(VARIANTS):
                for si, model in enumerate(models[variant]):
                    maps["clipped_scores"][position, vi, si, :m] = np.clip(model.predict_raw(features), -1., 1.)
            if (progress + 1) % 500 == 0:
                print(f"{phase} catalog cache {progress + 1}/{n}", flush=True)
        runtime = {"generator_version": generator.version, "generator_bindings": generator.bindings,
                   "temporal_manifest_sha256": temporal.version,
                   "verified_temporal_array_hashes": temporal.manifest["arrays"]}
    finally:
        if generator is not None:
            generator.close()
        if temporal is not None:
            temporal.close()
        for array in maps.values():
            array.flush()
        close_arrays(maps.values())
    for name, array in arrays.items():
        np.save(directory / f"{name}.npy", array)
    save_cache_manifest(directory, bindings)
    return directory, runtime


def score_cache(cache, delta, all_ablations=False):
    n = len(cache["rows"])
    hits = {"baseline": np.zeros(n, dtype=np.int32), "full13": np.zeros(n, dtype=np.int32)}
    if all_ablations:
        hits.update({name: np.zeros(n, dtype=np.int32) for name in ("no_category", "no_recent", "no_gate")})
        hits.update({f"full13_seed{seed}": np.zeros(n, dtype=np.int32) for seed in SEEDS})
    hit100 = np.zeros(n, dtype=np.int32)
    changed, empty_same = {name: 0 for name in hits}, 0
    for position in range(n):
        m, length = int(cache["lengths"][position]), int(cache["history_length"][position])
        original = np.arange(m)
        relevant = np.asarray(cache["hit"][position, :m])
        hits["baseline"][position] = relevant[:10].sum()
        hit100[position] = relevant.sum()
        for name in hits:
            if name == "baseline":
                continue
            vi = list(VARIANTS).index(name) if name in VARIANTS else 0
            scores = np.asarray(cache["clipped_scores"][position, vi, :, :m])
            residual = scores[SEEDS.index(int(name.split("seed")[1]))] if "_seed" in name else scores.mean(axis=0)
            order, _ = order_from_residual(residual, length, delta, gate=name != "no_gate")
            if not np.array_equal(np.sort(order), original) or relevant[order].sum() != hit100[position]:
                raise ValueError("same-candidate Recall@100 invariant failed")
            if (length == 0 or delta == 0) and not np.array_equal(order, original):
                raise ValueError("exact baseline fallback failed")
            hits[name][position] = relevant[order[:10]].sum()
            changed[name] += int(not np.array_equal(order, original))
        empty_same += int(length == 0)
    return hits, hit100, {"per_request_candidate_and_hit100_invariant": True,
                           "empty_history_exact_fallback_requests": empty_same,
                           "changed_order_requests": changed}


def metrics(cache, hits, hit100, intervals):
    denominator = np.asarray(cache["denominators"][:, 0])
    total = int(denominator.sum())
    if total <= 0:
        raise ValueError("no logged clicks for micro recall")
    result = {name: {"hits10": int(values.sum()), "recall10": float(values.sum() / total),
                     "hits100": int(hit100.sum()), "recall100": float(hit100.sum() / total)} for name, values in hits.items()}
    if intervals:
        for name in result:
            if name != "baseline":
                result[name]["versus_baseline"] = recall_interval(hits["baseline"], hits[name], denominator, cache["users"])
        for name in ("no_category", "no_recent", "no_gate"):
            if name in hits:
                result["full13"]["versus_" + name] = recall_interval(hits[name], hits["full13"], denominator, cache["users"])
    return result


def slices(cache, hits):
    length, timestamp = np.asarray(cache["history_length"]), np.asarray(cache["timestamps"])
    masks = {"empty": length == 0, "history_1_5": (length > 0) & (length <= 5),
             "history_6_20": length > 5}
    masks.update({f"source_6h_block_{int(block)}": timestamp // 21600 == block for block in np.unique(timestamp // 21600)})
    result = {}
    for name, mask in masks.items():
        total = int(cache["denominators"][mask, 0].sum())
        result[name] = {"requests": int(mask.sum()), "all_clicks": total,
                        "recall10": {method: float(value[mask].sum() / total) if total else None for method, value in hits.items()}}
    return result


def evaluate(args, phase):
    started = time.perf_counter()
    report_path = Path(args.reports) / f"{phase}.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    fit_path = Path(args.reports) / "fit.json"
    fit_report = read_json(fit_path)
    fit_digest = sha256_file(fit_path)
    verify_hashes(fit_report["sources"])
    if input_hashes(args) != fit_report["input_files"]:
        raise ValueError("actual evaluation inputs differ from fit bindings")
    if fit_report["environment"] != environment():
        raise ValueError("fit/evaluation environments differ")
    models = load_models(fit_report)
    selection_path = Path(args.artifacts) / "selection.json"
    selection = None
    if phase != "tune":
        selection = read_json(selection_path)
        if selection["fit_report_sha256"] != sha256_file(fit_path) or selection["sources"] != fit_report["sources"]:
            raise ValueError("selection/fit mismatch")
        verify_hashes(selection["model_files"])
        tune_path = Path(args.reports) / "tune.json"
        tune = read_json(tune_path)
        if (tune["phase"] != "tune" or tune["selected_delta"] != selection["delta"]
                or selection["delta"] not in DELTA_GRID
                or tune["bindings"]["fit_report_sha256"] != fit_digest
                or tune["cache_manifest_sha256"] != selection["tune_cache_manifest_sha256"]
                or sha256_file(Path(args.artifacts) / "tune" / "cache.json") != tune["cache_manifest_sha256"]):
            raise ValueError("incomplete or inconsistent tune/selection evidence")
    store = Store(args.data)
    if store.history.shape[1] != 20:
        raise ValueError("v3 requires the inherited 20-slot padded history representation")
    if phase == "dev":
        confirm_path = Path(args.reports) / "confirm.json"
        confirm = read_json(confirm_path)
        expected_confirm_rows = row_summary(store, fixed_sample_rows(store, "confirm"))
        if (confirm["phase"] != "confirm" or confirm["selection_sha256"] != sha256_file(selection_path)
                or confirm["bindings"]["fit_report_sha256"] != fit_digest
                or confirm["bindings"]["rows"] != expected_confirm_rows
                or confirm["tune_report_sha256"] != sha256_file(tune_path)
                or confirm["cache_manifest_sha256"] != sha256_file(Path(args.artifacts) / "confirm" / "cache.json")
                or not isinstance(confirm["deployment_gate_passed"], bool)):
            raise ValueError("confirmation/selection mismatch")
        interval = confirm["metrics"]["full13"]["versus_baseline"]
        if confirm["deployment_gate_passed"] != bool(interval["delta"] > 0 and interval["ci95"][0] > 0):
            raise ValueError("confirmation decision violates the frozen gate")
        with np.load("artifacts/v2/dev_catalog_counts.npz", allow_pickle=False) as old:
            rows = old["rows"].copy()
        expected = np.sort(np.random.default_rng(2026).choice(np.flatnonzero(store.split == 2), 5000, replace=False))
        if not np.array_equal(rows, expected):
            raise ValueError("dev rows differ from fixed v2 sample")
    else:
        rows = fixed_sample_rows(store, phase)
    bindings = {"phase": phase, "rows": row_summary(store, rows), "fit_report_sha256": sha256_file(fit_path),
                "sources": fit_report["sources"]}
    if selection is not None:
        bindings["selection_sha256"] = sha256_file(selection_path)
    directory, runtime = build_cache(args, phase, store, rows, models, bindings)
    cache = load_cache(directory, bindings)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
              "bindings": bindings, "runtime": runtime, "cache_manifest_sha256": sha256_file(directory / "cache.json"),
              "environment": environment(), "all_clicks": int(cache["denominators"][:, 0].sum()),
              "eligible_clicks": int(cache["denominators"][:, 1].sum()),
              "seen_clicks": int(cache["denominators"][:, 2].sum()),
              "shortfall_requests": int((cache["lengths"] < 100).sum())}
    if phase == "tune":
        grid = {}
        for delta in DELTA_GRID:
            hits, hit100, invariants = score_cache(cache, delta)
            grid[str(delta)] = {"metrics": metrics(cache, hits, hit100, False), "invariants": invariants}
        delta = max(DELTA_GRID, key=lambda value: (grid[str(value)]["metrics"]["full13"]["hits10"], -value))
        report.update(delta_grid=grid, selected_delta=delta, tie_rule="maximum total hits10, then smaller delta")
        paths = [record["path"] for run in fit_report["runs"].values() for record in run["models"].values()]
        selection = {"delta": delta, "sources": fit_report["sources"], "models": fit_report["models"],
                     "model_files": {path: sha256_file(path) for base in paths for path in (base, base + ".json")},
                     "fit_report_sha256": sha256_file(fit_path), "primary_recipe": "full13+gate",
                     "confirmation_gate": "delta_R10>0 and paired_user_bootstrap_ci95_lower>0",
                     "v2_selection_sha256": sha256_file(args.v2_selection),
                     "tune_cache_manifest_sha256": report["cache_manifest_sha256"]}
        dump_new(selection_path, selection)
    else:
        hits, hit100, invariants = score_cache(cache, selection["delta"], all_ablations=phase == "dev")
        values = metrics(cache, hits, hit100, True)
        report.update(delta=selection["delta"], metrics=values, invariants=invariants,
                      selection_sha256=sha256_file(selection_path), slices=slices(cache, hits),
                      tune_report_sha256=sha256_file(tune_path))
        with (directory / "counts.npz").open("xb") as stream:
            np.savez_compressed(stream, hit100=hit100, **hits)
        report["counts_sha256"] = sha256_file(directory / "counts.npz")
        if phase == "confirm":
            interval = values["full13"]["versus_baseline"]
            report["deployment_gate_passed"] = bool(interval["delta"] > 0 and interval["ci95"][0] > 0)
            report["deployment_available_as_of"] = 1573776000
        else:
            report["deployment_gate_passed"] = confirm["deployment_gate_passed"]
            report["deployed_recipe"] = "full13+gate" if confirm["deployment_gate_passed"] else "baseline"
            report["confirm_sha256"] = sha256_file(confirm_path)
            report["limitations"] = ["Previously observed dev benchmark, not an untouched test set.",
                                      "Column-group ablations do not erase category/recent-history information from every other feature.",
                                      "Ablation confidence intervals are exploratory and unadjusted for multiple comparisons.",
                                      "User-cluster intervals are conditional on the realized day and feedback stream; no online CTR or causal claim."]
    close_arrays(cache.values())
    verify_hashes(fit_report["sources"])
    verify_hashes(fit_report["input_files"])
    verify_hashes({fit_path: fit_digest})
    if selection is not None:
        verify_hashes(selection["model_files"])
    report["elapsed_seconds"] = time.perf_counter() - started
    dump_new(report_path, report)
    print(json.dumps({"phase": phase, "delta": report.get("delta", report.get("selected_delta")),
                      "gate": report.get("deployment_gate_passed"), "metrics": report.get("metrics")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "tune", "confirm", "dev"))
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--temporal", default="data/temporal_v2")
    parser.add_argument("--v2-selection", default="artifacts/v2/selection.json")
    parser.add_argument("--artifacts", default="artifacts/v3")
    parser.add_argument("--reports", default="reports/v3")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase == "fit":
            fit(args)
        else:
            evaluate(args, args.phase)


if __name__ == "__main__":
    main()
