"""Replay v3 training from verified processed data and compare frozen evidence.

This does not retrain the inherited text encoder or reconstruct raw MIND data.
Its comparison protocol is written before any model training starts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PARAMETER_ATOL = 1e-8
PHASES = ("fit", "tune", "confirm", "dev")
SEEDS = {"17", "29", "43"}
VARIANTS = {"full13", "no_category", "no_recent"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_protocol(protocol, reference):
    if digest(__file__) != protocol["runner_sha256"]:
        raise ValueError("retraining runner changed during execution")
    for phase in PHASES:
        if digest(reference / f"{phase}.json") != protocol["reference_reports"][phase]:
            raise ValueError("frozen reference report changed: " + phase)


def compare_arrays(left, right, *, atol, expected_names=None):
    import numpy as np
    result = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if not a.files or set(a.files) != set(b.files) or (expected_names is not None and set(a.files) != expected_names):
            raise ValueError("array names differ")
        for key in sorted(a.files):
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("array shape or dtype differs: " + key)
            exact = bool(np.array_equal(x, y))
            numeric = np.issubdtype(x.dtype, np.number)
            finite = not numeric or bool(np.isfinite(x).all() and np.isfinite(y).all())
            error = (float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                     if numeric and finite and x.size else 0. if finite else None)
            passed = finite and (exact or (numeric and np.issubdtype(x.dtype, np.floating) and error <= atol))
            result[key] = {"exact": exact, "max_absolute_error": error, "passed": passed}
    return result


def compare(reference, reports, artifacts):
    from newsforge.v3.residual import PairwiseResidual
    old = {phase: read(reference / f"{phase}.json") for phase in PHASES}
    new = {phase: read(reports / f"{phase}.json") for phase in PHASES}
    for value in (old["fit"], new["fit"]):
        if set(value["runs"]) != SEEDS or set(value["models"]) != VARIANTS:
            raise ValueError("expected exactly three seeds and three feature variants")
        if any(set(run["models"]) != VARIANTS for run in value["runs"].values()):
            raise ValueError("incomplete seed model inventory")
        if any(set(models) != SEEDS for models in value["models"].values()):
            raise ValueError("incomplete variant model inventory")
    checks = {}
    for key in ("sources", "input_files", "fit_rows"):
        checks["fit_" + key] = old["fit"][key] == new["fit"][key]
    models = {}
    for seed, previous in old["fit"]["runs"].items():
        current = new["fit"]["runs"][seed]
        for key in ("sampling", "pairs_sha256", "pair_features_sha256"):
            checks[f"seed{seed}_{key}"] = previous[key] == current[key]
        for variant, model in previous["models"].items():
            other = current["models"][variant]
            if digest(model["path"]) != model["npz_sha256"] or digest(other["path"]) != other["npz_sha256"]:
                raise ValueError("model bytes do not match their fit report")
            for item in (model, other):
                if digest(item["path"] + ".json") != item["sidecar_sha256"]:
                    raise ValueError("model sidecar changed")
                PairwiseResidual.load(item["path"])
            arrays = compare_arrays(model["path"], other["path"], atol=PARAMETER_ATOL,
                                    expected_names={"mean", "scale", "weights"})
            label = f"{variant}_seed{seed}"
            models[label] = {"arrays": arrays, "serialized_bytes_equal": model["npz_sha256"] == other["npz_sha256"]}
            checks[label] = all(row["passed"] for row in arrays.values())
            checks[label + "_converged"] = other["fit_metadata"]["optimizer_success"] is True
    counts = {}
    for phase in PHASES[1:]:
        for key in ("all_clicks", "eligible_clicks", "seen_clicks", "shortfall_requests"):
            checks[f"{phase}_{key}"] = old[phase][key] == new[phase][key]
        checks[phase + "_rows"] = old[phase]["bindings"]["rows"] == new[phase]["bindings"]["rows"]
        keys = ("delta_grid", "selected_delta", "tie_rule") if phase == "tune" else (
            "delta", "metrics", "invariants", "slices", "deployment_gate_passed")
        for key in keys:
            checks[f"{phase}_{key}"] = old[phase][key] == new[phase][key]
        if phase == "confirm":
            checks["confirm_availability"] = old[phase]["deployment_available_as_of"] == new[phase]["deployment_available_as_of"]
        if phase == "dev":
            checks["dev_deployed_recipe"] = old[phase]["deployed_recipe"] == new[phase]["deployed_recipe"]
        if phase != "tune":
            original = Path("artifacts/v3") / phase / "counts.npz"
            fresh = artifacts / phase / "counts.npz"
            if digest(original) != old[phase]["counts_sha256"] or digest(fresh) != new[phase]["counts_sha256"]:
                raise ValueError("per-query counts do not match their reports")
            counts[phase] = compare_arrays(original, fresh, atol=0., expected_names=set(old[phase]["metrics"]) | {"hit100"})
            checks[phase + "_per_query_counts"] = all(row["exact"] and row["passed"] for row in counts[phase].values())
    return {"passed": all(checks.values()), "checks": checks, "models": models,
            "per_query_counts": counts, "reference_environment": old["fit"]["environment"],
            "retrained_environment": new["fit"]["environment"],
            "retrained_dev_metrics": new["dev"]["metrics"],
            "reports": {phase: {"reference_sha256": digest(reference / f"{phase}.json"),
                                 "retrained_sha256": digest(reports / f"{phase}.json")} for phase in PHASES}}


def run(root):
    from newsforge.delivery.acceptance import audit_environment
    root = Path(root)
    if root.is_absolute() or ".." in root.parts or not root.parts or root.parts[0] != "artifacts":
        raise ValueError("use a new relative output under artifacts/")
    root.mkdir(parents=True, exist_ok=False)
    environment = audit_environment(Path.cwd(), Path("requirements-delivery-windows.lock"))
    save(root / "environment.json", environment)
    artifacts, reports = root / "models_and_caches", root / "reports"
    reference = Path("reports/v3")
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "same-host isolated-environment v3 retraining from frozen processed data and v2 baseline",
                "raw_data_preprocessing_repeated": False, "text_encoder_retrained": False,
                "parameter_atol": PARAMETER_ATOL, "parameter_rtol": 0.,
                "fit_pairs_and_features": "SHA256 exact", "per_query_counts": "exact",
                "metrics_intervals_selection_and_gate": "exact", "phases": list(PHASES),
                "runner_sha256": digest(__file__),
                "environment_preflight_passed": environment["passed"],
                "environment_receipt_sha256": digest(root / "environment.json"),
                "wheel_lock_sha256": environment["wheel_lock_sha256"],
                "reference_reports": {phase: digest(reference / f"{phase}.json") for phase in PHASES}}
    save(root / "protocol.json", protocol)
    protocol_digest = digest(root / "protocol.json")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONNOUSERSITE="1")
    stages = []
    for phase in PHASES:
        command = [sys.executable, "-m", "newsforge.v3.experiment", phase,
                   "--artifacts", artifacts.as_posix(), "--reports", reports.as_posix()]
        started = time.perf_counter()
        with (root / f"{phase}.log").open("xb") as stream:
            completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = {"phase": phase, "returncode": completed.returncode,
                  "elapsed_seconds": time.perf_counter() - started, "log_sha256": digest(root / f"{phase}.log")}
        save(root / f"{phase}_receipt.json", record)
        stages.append(record)
        print(json.dumps(record), flush=True)
        if completed.returncode:
            save(root / "comparison.json", {"passed": False, "failed_phase": phase, "stages": stages})
            return 1
    verify_protocol(protocol, reference)
    result = compare(reference, reports, artifacts)
    verify_protocol(protocol, reference)
    if digest(root / "protocol.json") != protocol_digest:
        raise ValueError("saved pre-training comparison protocol changed")
    result.update(protocol=protocol, protocol_sha256=protocol_digest, stages=stages)
    save(root / "comparison.json", result)
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"])}), flush=True)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/delivery/retrain_v3")
    args = parser.parse_args()
    raise SystemExit(run(args.output))


if __name__ == "__main__":
    main()
