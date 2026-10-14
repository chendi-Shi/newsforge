"""Append-only, isolated raw-to-v3 reproduction orchestration.

This orchestrates frozen research modules; it never changes their algorithms or
copies released metrics into a new run. ``plan`` is stdlib-only and performs no
training. A successful stage receipt means that the command exited successfully
and its declared outputs were materialized, not that historical numbers match.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shutil
import subprocess
import sys
import time


SCHEMA = 1
STATE = Path("artifacts/delivery/pipeline")
ROOT = Path(__file__).resolve().parents[2]
SEEDS = (17, 29, 43)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Stage:
    name: str
    dependencies: tuple[str, ...]
    argv: tuple[str, ...]  # Python argv without the interpreter.
    outputs: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    notes: str = ""


def stages():
    """The minimal dependency closure for the *existing full v3 auditor*.

    v2.audit binds the real v1 full-ranking report. v3.audit binds v2.audit and
    the v2 catalog counts. Omitting those stages would manufacture a shortcut.
    Legacy GBDT, ANN and v1 HTTP runs are not dependencies of this release.
    """
    items = []

    def add(name, deps, module, args=(), outputs=(), forbidden=(), notes=""):
        items.append(Stage(name, tuple(deps), ("-m", module, *args), tuple(outputs),
                           tuple(forbidden), notes))

    own = "newsforge.delivery.pipeline"
    add("environment", (), own, ("_environment",),
        ("reports/delivery/environment.json",), notes="Check Python 3.10 and both dependency files; does not install packages.")
    add("unit-tests", ("environment",), "unittest", ("discover", "-s", "tests", "-v"),
        notes="The runner saves the successful complete output as reports/v3_release_unit_tests.txt.")
    add("data", ("unit-tests",), "newsforge.data", ("--threads", "2"),
        ("data/processed/manifest.json", "data/processed/provenance.json"), ("data/processed",),
        "Use the explicitly imported train/dev raw TSV files, or download when init omitted --raw.")
    add("data-audit", ("data",), "newsforge.audit_data",
        ("--data", "data/processed", "--output", "reports/data_validation.json"),
        ("reports/data_validation.json",))
    neural = tuple(f"artifacts/{v}_seed{s}.pt" for v in ("mean_random", "mean_mixed", "ranker") for s in SEEDS)
    add("neural-train", ("data-audit",), "newsforge.train",
        ("--variants", "mean_random", "mean_mixed", "ranker", "--seeds", "17", "29", "43",
         "--epochs", "2", "--batch-size", "1024", "--learning-rate", "0.001", "--threads", "1",
         "--validation-impressions", "12000", "--skip-test"),
        neural + ("reports/matched_warmup_checks.json",),
        neural + ("artifacts/retriever.pt", "artifacts/reranker.pt", "artifacts/article_embeddings.npy"))
    add("v1-full-ranking", ("neural-train",), "newsforge.final_evaluation", ("--threads", "1"),
        ("reports/ranking_experiment.json",), ("reports/ranking_experiment.json",),
        "Runs its own internal scheduling gates before official dev. This dependency precedes replacing v1 serving aliases.")
    add("temporal", ("data-audit",), "newsforge.v2.temporal", (),
        ("data/temporal_v2/manifest.json",), ("data/temporal_v2",))
    add("backbone-initial", ("v1-full-ranking",), own, ("_materialize", "--phase", "initial"),
        ("reports/delivery/backbone_initial.json",), ("artifacts/delivery/backbone_initial",),
        "Require all three mean_random checkpoints to contain recorded full epoch 2; archive v1 exports, materialize seed17 only to let the frozen selector run.")
    add("backbone-provisional", ("backbone-initial",), "newsforge.v2.select_backbone",
        ("--output", "reports/delivery/backbone_provisional.json", "--threads", "1"),
        ("reports/delivery/backbone_provisional.json",))
    add("backbone-materialize", ("backbone-provisional",), own,
        ("_materialize", "--phase", "selected"), ("reports/delivery/backbone_selected.json",),
        ("artifacts/delivery/backbone_selected",), "Materialize the internal mature-window winner and matching article vectors; never select from dev.")
    add("backbone-final", ("backbone-materialize",), "newsforge.v2.select_backbone", ("--threads", "1"),
        ("reports/v2/backbone_matured_selection.json",),
        notes="Recompute canonical selection so matches_current_retriever_state is true; no report-field patching.")
    add("encoder", ("backbone-final",), "newsforge.v2.retrieval",
        ("export-encoder", "--checkpoint", "artifacts/retriever.pt", "--output", "artifacts/v2/user_encoder.npz",
         "--processed-dir", "data/processed", "--article-embeddings", "artifacts/article_embeddings.npy",
         "--validation-count", "512"), ("artifacts/v2/user_encoder.npz", "artifacts/v2/user_encoder.npz.json"))
    add("ranker-train", ("temporal", "encoder"), "newsforge.v2.ranker",
        ("--train-only", "--variants", "content_only", "context_only", "full", "--seeds", "17", "29", "43",
         "--max-positives", "100000", "--validation-impressions", "6000", "--threads", "1"),
        ("reports/v2/ranker/selection.json",), ("artifacts/v2/ranker", "reports/v2/ranker"))
    add("ranker-mature", ("ranker-train",), "newsforge.v2.mature_selection", (),
        ("reports/v2/ranker_matured_selection.json", "reports/v2/ranker_matured_internal_evaluation.json"),
        ("artifacts/v2/matured_internal_ranking",))
    add("cache-equivalence", ("encoder", "temporal"), "newsforge.v2.verify_cache", (),
        ("reports/v2/query_cache_equivalence.json",))
    add("v2-select", ("ranker-mature", "cache-equivalence"), "newsforge.v2.experiment", ("--phase", "select"),
        ("reports/v2/select.json", "artifacts/v2/selection.json"))
    add("v3-baseline-gate", ("v2-select",), own, ("_baseline_gate",),
        ("reports/delivery/v3_baseline_gate.json",),
        notes="Fail if fresh internal selection is not popularity/24h/pool100/route200/no reranker. Do not force the historical winner or tune on dev.")
    for phase in ("catalog", "ranking", "ranking-full"):
        add(f"v2-{phase}", ("v3-baseline-gate",), "newsforge.v2.experiment", ("--phase", phase),
            (f"reports/v2/{phase}.json",))
    add("v2-audit", ("v2-catalog", "v2-ranking", "v2-ranking-full", "v1-full-ranking"),
        "newsforge.v2.audit", (), ("reports/v2/release_integrity.json",))
    add("fast-equivalence", ("v3-baseline-gate",), "newsforge.v3.benchmark_fast", (),
        ("reports/v3/fast_retrieval_benchmark.json",))
    previous = "fast-equivalence"
    for phase in ("fit", "tune", "confirm", "dev"):
        deps = (previous, "v2-audit") if phase == "dev" else (previous,)
        outputs = (f"reports/v3/{phase}.json",)
        if phase == "tune":
            outputs += ("artifacts/v3/selection.json",)
        add(f"v3-{phase}", deps, "newsforge.v3.experiment", (phase,), outputs,
            (f"artifacts/v3/{phase}",), "Fixed time windows, seeds and delta grid; a failed confirm keeps the baseline service.")
        previous = f"v3-{phase}"
    add("v3-http", ("v3-confirm",), "newsforge.v3.benchmark_service", (),
        ("reports/v3/service_benchmark.json",), ("artifacts/v3/service_benchmark",))
    items.append(Stage("v3-plot", ("v3-dev", "fast-equivalence"), ("scripts/plot_v3_results.py",),
                       ("reports/v3/v3_results.metadata.json", "reports/v3/v3_results.png", "reports/v3/v3_results.svg")))
    add("v3-audit", ("v3-dev", "v3-http", "v3-plot", "unit-tests"), "newsforge.v3.audit", (),
        ("reports/v3/release_integrity.json",),
        notes="Independent array and source audit, including cache-to-V3Engine parity. Only this stage completes the raw research/service verification chain.")
    return tuple(items)


def ordered_stages(target="v3-audit", definitions=None):
    definitions = stages() if definitions is None else tuple(definitions)
    lookup = {stage.name: stage for stage in definitions}
    if len(lookup) != len(definitions):
        raise ValueError("duplicate stage name")
    if target not in lookup:
        raise ValueError(f"unknown stage: {target}")
    result, pending, complete = [], set(), set()

    def visit(name):
        if name in pending:
            raise ValueError("stage dependency cycle")
        if name in complete:
            return
        if name not in lookup:
            raise ValueError(f"missing dependency: {name}")
        pending.add(name)
        for dep in lookup[name].dependencies:
            visit(dep)
        pending.remove(name)
        complete.add(name)
        result.append(lookup[name])

    visit(target)
    return tuple(result)


def make_plan(target="v3-audit"):
    return {"schema_version": SCHEMA, "kind": "raw_retraining_plan", "executed": False,
            "target": target, "stages": [asdict(stage) for stage in ordered_stages(target)],
            "python": "Windows CPython 3.10 with requirements-windows.lock and requirements-plot.txt installed",
            "workspace_policy": "Fresh independent source copy, standard relative paths, shipped reports kept under reports-delivered; no data or weights imported except explicitly supplied raw TSV files.",
            "resume_policy": "Verify completed receipts and source hashes; never rerun or overwrite a started/failed stage. Preserve the entire failed workspace and initialize a fresh one.",
            "selection_policy": "No official-dev-driven changes; all frozen selectors retain their internal-only decision rules. The inherited v3 baseline gate may stop reproduction if a fresh internal winner differs.",
            "limitations": ["A plan is not a completed raw-data reproduction.",
                "Byte-identical manifests/metrics are not promised: elapsed time, runtime, and numerical libraries can differ.",
                "The frozen v1 trainer retains its validation-best epoch; the adapter fails if any required mean_random seed did not retain epoch2, rather than relabeling epoch1 as epoch2.",
                "Prior logged preprocessing, training, and evaluation subsets alone total about 4.55 hours; downloads, repeated validation and other stages add time. This is historical workload context, not an ETA.",
                "No production deployment, online A/B test or independent unseen dataset is created by this pipeline."]}


def inside(root, relative):
    """Reject traversal/symlink escapes before writes or evidence reads."""
    root = Path(root).resolve()
    # Reject foreign-platform drive/UNC paths too: manifests are portable data.
    value = str(relative).replace("\\", "/")
    if (not value or PureWindowsPath(value).drive or PurePosixPath(value).is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"invalid workspace-relative path: {relative}")
    path = root / value
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"workspace path escapes root: {relative}")
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError(f"workspace symlink is not allowed: {relative}")
    return path


def source_files(source):
    source = Path(source).resolve()
    found = set()
    for directory in ("newsforge", "tests", "scripts", "docs"):
        for path in (source / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".md"):
                found.add(path.relative_to(source).as_posix())
    for pattern in ("README*.md", "requirements*.txt", "requirements*.lock", "LICENSE*", ".gitignore", ".gitattributes"):
        found.update(path.relative_to(source).as_posix() for path in source.glob(pattern) if path.is_file())
    required = {"newsforge/delivery/pipeline.py", "newsforge/train.py", "newsforge/v3/experiment.py",
                "docs/V3_PROTOCOL.md", "requirements-windows.lock", "requirements-plot.txt"}
    if not required <= found:
        raise ValueError("source tree lacks required reproduction files")
    return tuple(sorted(found))


def initialize(source, workspace, raw=None):
    source, workspace = Path(source).resolve(), Path(workspace).resolve()
    if workspace == source or workspace.is_relative_to(source) or source.is_relative_to(workspace):
        raise ValueError("workspace must be outside and independent of the source checkout")
    if workspace.exists():
        raise FileExistsError("new workspace must not exist, even if empty")
    files = source_files(source)
    raw_root = Path(raw).resolve() if raw is not None else None
    raw_names = tuple(f"{split}/{name}" for split in ("train", "dev") for name in ("news.tsv", "behaviors.tsv"))
    if raw_root is not None:
        for name in raw_names:
            path = inside(raw_root, name)
            if not path.is_file():
                raise FileNotFoundError(f"required raw file missing: {name}")
    workspace.mkdir(parents=True, exist_ok=False)
    # No deletion/rollback: partially copied initialization evidence is retained.
    write_new(workspace / STATE / "initializing.json", {"started_utc": utc(), "schema_version": SCHEMA})
    hashes = {}
    for name in files:
        original, target = inside(source, name), inside(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        hashes[name] = digest(target)
        if hashes[name] != digest(original):
            raise ValueError(f"source changed while copying: {name}")
    delivered = source / "reports"
    if delivered.is_dir():
        for path in sorted(delivered.rglob("*")):
            if path.is_file():
                name = path.relative_to(source).as_posix()
                inside(source, name)
                target = inside(workspace, "reports-delivered/" + path.relative_to(delivered).as_posix())
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    raw_hashes = {}
    if raw_root is not None:
        for name in raw_names:
            origin, target = inside(raw_root, name), inside(workspace, f"data/raw/{name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            raw_hashes[name] = digest(target)
            if raw_hashes[name] != digest(origin):
                raise ValueError(f"raw source changed while copying: {name}")
    manifest = {"schema_version": SCHEMA, "created_utc": utc(), "source_files": hashes,
                "raw_mode": "provided" if raw_root is not None else "download",
                "raw_files": raw_hashes, "plan_sha256": canonical_hash(make_plan()),
                "scope": "New outputs only; reports-delivered is reference material and is never an executable stage input."}
    write_new(workspace / STATE / "workspace.json", manifest)
    (workspace / "reports").mkdir(exist_ok=False)
    return manifest


def verify_workspace(workspace, definitions=None):
    workspace = Path(workspace).resolve()
    manifest = read(inside(workspace, STATE / "workspace.json"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported workspace schema")
    for name, expected in manifest["source_files"].items():
        if digest(inside(workspace, name)) != expected:
            raise ValueError(f"frozen reproduction source changed: {name}")
    if definitions is None and manifest["plan_sha256"] != canonical_hash(make_plan()):
        raise ValueError("pipeline plan changed after initialization")
    for name, expected in manifest["raw_files"].items():
        if digest(inside(workspace, f"data/raw/{name}")) != expected:
            raise ValueError(f"imported raw data changed: {name}")
    return manifest


def completed(workspace, stage, python, dependency_hashes):
    directory = inside(workspace, STATE / stage.name)
    receipt = directory / "success.json"
    if not directory.exists():
        return None
    if not receipt.is_file():
        raise FileExistsError(f"stage {stage.name} was started/failed; preserve this workspace and use a new one")
    value = read(receipt)
    if (value["stage_definition_sha256"] != canonical_hash(asdict(stage))
            or value["python"] != str(python) or value["dependencies"] != dependency_hashes
            or value.get("workspace_manifest_sha256") != digest(inside(workspace, STATE / "workspace.json"))):
        raise ValueError(f"stage receipt identity changed: {stage.name}")
    if value.get("status") != "success" or value.get("returncode") != 0:
        raise ValueError(f"invalid success receipt: {stage.name}")
    if digest(directory / "output.log") != value["log_sha256"]:
        raise ValueError(f"stage output log changed: {stage.name}")
    for name, expected in value["outputs"].items():
        if digest(inside(workspace, name)) != expected:
            raise ValueError(f"completed stage output changed: {name}")
    return digest(receipt)


def dependency_outputs(workspace, stage):
    """Bind concrete declared dependency outputs as well as their receipt files."""
    inputs = {}
    for name in stage.dependencies:
        receipt = read(inside(workspace, STATE / name / "success.json"))
        for path, expected in receipt["outputs"].items():
            if path in inputs and inputs[path] != expected:
                raise ValueError(f"dependency outputs disagree: {path}")
            if digest(inside(workspace, path)) != expected:
                raise ValueError(f"dependency input changed: {path}")
            inputs[path] = expected
    return inputs


def run(workspace, python, target="v3-audit", *, definitions=None, executor=None):
    workspace, python = Path(workspace).resolve(), Path(python).resolve()
    if not python.is_file():
        raise FileNotFoundError("explicit Python interpreter is required")
    manifest = verify_workspace(workspace, definitions)
    receipt_hashes = {}
    for stage in ordered_stages(target, definitions):
        dependency_hashes = {name: receipt_hashes[name] for name in stage.dependencies}
        saved = completed(workspace, stage, python, dependency_hashes)
        if saved is not None:
            receipt_hashes[stage.name] = saved
            print(json.dumps({"stage": stage.name, "status": "verified_previous_success"}), flush=True)
            continue
        for name in set(stage.outputs) | set(stage.forbidden):
            if inside(workspace, name).exists():
                raise FileExistsError(f"stage {stage.name} would reuse/overwrite an existing output: {name}")
        if stage.name == "unit-tests" and inside(workspace, "reports/v3_release_unit_tests.txt").exists():
            raise FileExistsError("unit test release log already exists")
        command = [str(python), *stage.argv]
        if stage.name == "data" and manifest["raw_mode"] == "download":
            command.append("--download")
        inputs = dependency_outputs(workspace, stage)
        directory = inside(workspace, STATE / stage.name)
        directory.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        running = {"stage": stage.name, "started_utc": utc(), "argv": command,
                   "stage_definition_sha256": canonical_hash(asdict(stage)), "python": str(python),
                   "workspace_manifest_sha256": digest(inside(workspace, STATE / "workspace.json")),
                   "dependencies": dependency_hashes, "inputs": inputs, "status": "running",
                   "evidence_scope": "Frozen source and supplied raw files, dependency receipts and their declared outputs; frozen downstream auditors validate their own full artifact bindings."}
        write_new(directory / "started.json", running)
        print(json.dumps({"stage": stage.name, "status": "running", "log": str(directory / "output.log")}), flush=True)
        result = None
        try:
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
            with (directory / "output.log").open("xb") as log:
                invoke = subprocess.run if executor is None else executor
                result = invoke(command, cwd=str(workspace), stdout=log, stderr=subprocess.STDOUT,
                                env=env, check=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode != 0:
                raise RuntimeError(f"stage {stage.name} exited with code {result.returncode}")
            outputs = list(stage.outputs)
            if stage.name == "unit-tests":
                output = inside(workspace, "reports/v3_release_unit_tests.txt")
                shutil.copyfile(directory / "output.log", output)
                outputs.append("reports/v3_release_unit_tests.txt")
            hashes = {name: digest(inside(workspace, name)) for name in outputs}
            # Detect source/raw mutation during a command, before declaring success.
            verify_workspace(workspace, definitions)
            for name, expected in inputs.items():
                if digest(inside(workspace, name)) != expected:
                    raise ValueError(f"stage mutated dependency input: {name}")
            if digest(inside(workspace, STATE / "workspace.json")) != running["workspace_manifest_sha256"]:
                raise ValueError("workspace manifest changed during stage")
            success = {**running, "status": "success", "finished_utc": utc(), "returncode": result.returncode,
                       "elapsed_seconds": time.perf_counter() - started, "outputs": hashes,
                       "log_sha256": digest(directory / "output.log")}
            write_new(directory / "success.json", success)
            receipt_hashes[stage.name] = digest(directory / "success.json")
        except BaseException as exc:
            failure = {**running, "status": "failed", "finished_utc": utc(),
                       "returncode": None if result is None else result.returncode,
                       "elapsed_seconds": time.perf_counter() - started,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "partial_outputs_retained": True}
            write_new(directory / "failed.json", failure)
            raise
    return {"target": target, "status": "success", "completed_stages": list(receipt_hashes),
            "receipt_sha256": receipt_hashes,
            "scope": "Only the requested dependency closure ran; success is not a claim about skipped later stages."}


def check_environment():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("frozen raw pipeline requires Python 3.10")
    expected = {}
    for filename in ("requirements-windows.lock", "requirements-plot.txt"):
        for line in Path(filename).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "--")):
                name, version = line.split("==", 1)
                expected[name] = version
    installed, mismatch = {}, {}
    for name, version in expected.items():
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
        if installed[name] != version:
            mismatch[name] = {"expected": version, "actual": installed[name]}
    if mismatch:
        raise RuntimeError("install both locked requirements before execution: " + json.dumps(mismatch))
    write_new("reports/delivery/environment.json", {"passed": True, "python": platform.python_version(),
              "platform": platform.platform(), "packages": installed,
              "requirements_sha256": {name: digest(name) for name in ("requirements-windows.lock", "requirements-plot.txt")}})


def baseline_gate():
    selection = read("artifacts/v2/selection.json")
    serving = selection["serving"]
    config = serving["candidate_config"]
    expected = {"channels": ["popular"], "pool_size": 100, "route_size": 200,
                "rank_constant": 60, "recent_hours": 24, "freeze_feedback": False}
    if serving["use_reranker"] or any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("fresh internal selection differs from the inherited v3 protocol; preserve this run and define a new study, never force the old winner")
    write_new("reports/delivery/v3_baseline_gate.json", {"passed": True, "candidate_config": config,
              "v2_selection_sha256": digest("artifacts/v2/selection.json"),
              "policy": "Checked fresh internal selection only; no official-dev metrics consulted."})


def selected_backbone_seed(candidates, selection):
    """Require the internally selected winner to bind the current checkpoints."""
    from newsforge.v2.select_backbone import select_winner

    if set(candidates) != set(SEEDS):
        raise ValueError("materialization requires every fixed backbone seed")
    winner = select_winner(selection["candidates"])
    for item in selection["candidates"]:
        entry = candidates[item["seed"]]
        if (item["checkpoint_sha256"] != entry["checkpoint_sha256"]
                or item["state_sha256"] != entry["state_sha256"]):
            raise ValueError("checkpoint changed after provisional selection")
    if selection["selected"]["seed"] != winner["seed"]:
        raise ValueError("provisional winner differs from its internal-only decision rule")
    return winner["seed"]


def materialize(phase):
    """Bridge a known gap in frozen select_backbone without changing its code.

    This validates every eligible checkpoint, then archives the current serving
    aliases and writes a matched checkpoint/catalog pair. A failed export is
    retained by the stage runner. Only the provisional internal report chooses
    the second export; the canonical selection is then rerun independently.
    """
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from newsforge.evaluate import encode_catalog
    from newsforge.store import Store
    from newsforge.train import build_model, state_digest, training_cache_signature, training_source_digests
    from newsforge.v2.select_backbone import checkpoint_evidence

    if phase not in ("initial", "selected"):
        raise ValueError("unknown materialization phase")
    torch.set_num_threads(1)
    threadpool_limits(1)
    store = Store("data/processed")
    store.ensure_training_cache()
    candidates = {}
    for seed in SEEDS:
        path = Path(f"artifacts/mean_random_seed{seed}.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model, _, _ = build_model("mean_random", store.features.shape[1])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        state = state_digest(model)
        evidence = checkpoint_evidence(checkpoint, seed=seed, state_sha256=state,
            feature_dim=store.features.shape[1], manifest_sha256=digest("data/processed/manifest.json"),
            source_sha256=training_source_digests(), cache_signature=training_cache_signature(store),
            train_cutoff_timestamp=int(store.manifest["train_cutoff_timestamp"]), eligible_positives=len(store.positive_rows))
        candidates[seed] = {"path": path, "model": model, "state_sha256": state,
                            "checkpoint_sha256": digest(path), "evidence": evidence}
    seed = 17
    if phase == "selected":
        selection = read("reports/delivery/backbone_provisional.json")
        seed = selected_backbone_seed(candidates, selection)
    selected = candidates[seed]
    output = Path(f"artifacts/delivery/backbone_{phase}")
    output.mkdir(parents=True, exist_ok=False)
    old_hashes = {}
    for name in ("retriever.pt", "article_embeddings.npy"):
        origin = Path("artifacts") / name
        if not origin.is_file():
            raise FileNotFoundError(f"missing prior serving alias: {origin}")
        old_hashes[name] = digest(origin)
        shutil.copyfile(origin, output / ("previous_" + name))
    shutil.copyfile(selected["path"], output / "retriever.pt")
    values = encode_catalog(store, selected["model"])
    if values.shape != (len(store.news_ids), 64) or not np.isfinite(values).all():
        raise ValueError("exported vectors failed shape/finite validation")
    with (output / "article_embeddings.npy").open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    new_hashes = {name: digest(output / name) for name in ("retriever.pt", "article_embeddings.npy")}
    for name, expected in old_hashes.items():
        if digest(Path("artifacts") / name) != expected:
            raise RuntimeError("prior serving alias changed during export")
    # This is the one explicitly declared mutable alias pair. Prior bytes are
    # already archived; no training checkpoint or previous report is overwritten.
    for name in new_hashes:
        shutil.copyfile(output / name, Path("artifacts") / name)
    write_new(f"reports/delivery/backbone_{phase}.json", {"passed": True, "phase": phase,
        "seed": seed, "state_sha256": selected["state_sha256"], "archived_previous_sha256": old_hashes,
        "materialized_sha256": new_hashes, "checkpoint_sha256": selected["checkpoint_sha256"],
        "source_sha256": digest(__file__), "official_dev_used_for_selection": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan", help="List stages; no training, installation or download")
    p.add_argument("--through", default="v3-audit")
    p.add_argument("--output", type=Path)
    p = commands.add_parser("init", help="Create a new isolated reproduction source copy")
    p.add_argument("--source", type=Path, default=ROOT)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--raw", type=Path, help="Existing raw root with train/dev/{news,behaviors}.tsv; omit for explicit download stage")
    p = commands.add_parser("run", help="Execute stages serially and retain success/failure receipts")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--through", default="v3-audit")
    commands.add_parser("_environment", help=argparse.SUPPRESS)
    commands.add_parser("_baseline_gate", help=argparse.SUPPRESS)
    p = commands.add_parser("_materialize", help=argparse.SUPPRESS)
    p.add_argument("--phase", required=True, choices=("initial", "selected"))
    args = parser.parse_args()
    if args.command == "plan":
        value = make_plan(args.through)
        if args.output is not None:
            write_new(args.output, value)
        else:
            print(json.dumps(value, indent=2, ensure_ascii=False))
    elif args.command == "init":
        value = initialize(args.source, args.workspace, args.raw)
        print(json.dumps({"workspace": str(args.workspace), "source_files": len(value["source_files"]), "raw_mode": value["raw_mode"]}))
    elif args.command == "run":
        print(json.dumps(run(args.workspace, args.python, args.through), indent=2))
    elif args.command == "_environment":
        check_environment()
    elif args.command == "_baseline_gate":
        baseline_gate()
    elif args.command == "_materialize":
        materialize(args.phase)


if __name__ == "__main__":
    main()
