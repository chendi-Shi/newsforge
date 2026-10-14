"""Copy and verify a private directory bundle for frozen v3 artifact replay.

This is not raw-data training reproduction. It transfers the exact inputs needed
by ``v3.serve.load_engine`` without bypassing its source/data/model checks. No
Python environment is copied and no model is deserialized by this module.
SHA256 detects byte changes; authenticate a bundle by obtaining its manifest
digest through a trusted channel, not from the bundle alone.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import uuid


FORMAT = "newsforge-v3-private-artifact-replay-directory-v1"
CHUNK_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_FILES = 10000
MAX_TOTAL_BYTES = 32 * 1024**3
SCOPE = "artifact replay of frozen v3; no raw-data rebuild, retraining or new evaluation"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])(?:\..*)?\Z", re.I)
REPORTS = ("fit", "tune", "confirm", "release_integrity")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    """Canonical cross-platform relative file names; reject Windows aliases."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("path must be a canonical relative POSIX path")
    parts = value.split("/")
    if any(not part or part in (".", "..") or part[-1:] in (".", " ")
           or any(ord(char) < 32 or char in '<>\"|?*' for char in part)
           or _RESERVED.fullmatch(part) for part in parts):
        raise ValueError("unsafe relative path")
    if PurePosixPath(value).is_absolute():
        raise ValueError("absolute paths are forbidden")
    return value


def _linked(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0)
                                            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def safe_path(root, relative):
    relative = relative_path(relative)
    root = Path(root).absolute()
    # Reject links/junctions even when their resolved target happens to stay
    # inside root. Cloud placeholders must be hydrated to ordinary files first.
    for item in (root, *root.parents):
        if _linked(item):
            raise ValueError("linked/reparse bundle or checkout root is forbidden")
        if item.exists() and not item.is_dir():
            raise ValueError("bundle/checkout root ancestor is not a directory")
    result = root
    components = relative.split("/")
    for index, component in enumerate(components):
        result = result / component
        if _linked(result):
            raise ValueError(f"linked/reparse path is forbidden: {relative}")
        if result.exists() and index < len(components) - 1 and not result.is_dir():
            raise ValueError(f"path ancestor is not a directory: {relative}")
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("path escapes root")
    return result


def _object_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("missing or oversized JSON metadata")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("JSON metadata must be an object")
    return value


def _payload_allowed(path):
    """Import can never install source code, executable scripts or environments."""
    p = PurePosixPath(path)
    if p.parent.as_posix() == "data/processed":
        return p.suffix in (".npy", ".json")
    if p.parent.as_posix() == "data/temporal_v2":
        return p.suffix in (".npy", ".json")
    if path in ("artifacts/article_embeddings.npy", "artifacts/retriever.pt",
                "artifacts/v2/selection.json", "artifacts/v2/dev_catalog_counts.npz", "artifacts/v2/user_encoder.npz",
                "artifacts/v2/user_encoder.npz.json", "artifacts/v3/selection.json",
                "artifacts/v3/tune/cache.json", "artifacts/v3/confirm/cache.json",
                "artifacts/v3/confirm/counts.npz", "artifacts/v3/dev/counts.npz"):
        return True
    if p.parent.as_posix() == "artifacts/v3/models":
        return bool(re.fullmatch(r"(?:full13|no_category|no_recent)-seed(?:17|29|43)\.npz(?:\.json)?", p.name))
    return path in {f"reports/v3/{name}.json" for name in REPORTS}


def _checkout_allowed(path):
    return (path.startswith("newsforge/") and path.endswith(".py")) or path == "docs/V3_PROTOCOL.md"


def _record(root, path, expected=None):
    target = safe_path(root, path)
    if not target.is_file():
        raise ValueError(f"required file is missing: {path}")
    if expected is not None and not _SHA.fullmatch(expected):
        raise ValueError("invalid expected SHA256")
    return {"path": path, "size": target.stat().st_size,
            "sha256": expected if expected is not None else sha256_file(target)}


def build_plan(repo, *, extra_files=(), git_executable=None):
    """Inventory required files without reading bound multi-GB arrays yet.

    Hashes from frozen manifests are expectations, not proof that payload bytes
    match. Export/verify/import stream every payload before declaring success.
    Extra replay inputs must already be hash-bound in the v3 fit/audit report.
    """
    repo = Path(repo).absolute()
    extra_files = tuple(extra_files)
    read = lambda name: _json(safe_path(repo, name))
    v2 = read("artifacts/v2/selection.json")
    selected = read("artifacts/v3/selection.json")
    fit, tune, confirm, audit = (read(f"reports/v3/{name}.json") for name in REPORTS)
    temporal = read("data/temporal_v2/manifest.json")
    if audit.get("passed") is not True:
        raise ValueError("bundle requires the completed v3 independent audit")
    if (selected["sources"] != fit["sources"] or selected["models"] != fit["models"]
            or selected["fit_report_sha256"] != sha256_file(safe_path(repo, "reports/v3/fit.json"))
            or selected["v2_selection_sha256"] != sha256_file(safe_path(repo, "artifacts/v2/selection.json"))
            or confirm["selection_sha256"] != sha256_file(safe_path(repo, "artifacts/v3/selection.json"))
            or confirm["tune_report_sha256"] != sha256_file(safe_path(repo, "reports/v3/tune.json"))
            or tune["cache_manifest_sha256"] != selected["tune_cache_manifest_sha256"]):
        raise ValueError("v3 release evidence chain differs")
    if v2["serving"]["encoder_path"] != "artifacts/v2/user_encoder.npz":
        raise ValueError("bundle supports the frozen v3 encoder location only")
    expected = {}

    def add(path, digest=None):
        path = relative_path(path)
        if not _payload_allowed(path):
            raise ValueError(f"not a permitted replay payload: {path}")
        if path in expected and digest is not None and expected[path] not in (None, digest):
            raise ValueError(f"conflicting source bindings: {path}")
        expected[path] = digest if digest is not None else expected.get(path)

    for name, record in temporal["sources"].items():
        add("data/processed/" + relative_path(name), record["sha256"])
    for name, record in temporal["arrays"].items():
        add("data/temporal_v2/" + relative_path(name), record["sha256"])
    frozen = v2["bindings"]
    for name in ("news_features.npy", "news_first_seen.npy", "news_train_clicks.npy",
                 "news_train_exposures.npy", "news_ids.json"):
        add("data/processed/" + name, frozen["processed_files"][name])
    add("data/processed/manifest.json", frozen["data_manifest"])
    add("data/temporal_v2/manifest.json", frozen["temporal_manifest"])
    retrieval = frozen["retrieval_artifacts"]
    for key, name in (("article_embeddings", "artifacts/article_embeddings.npy"),
                      ("retriever", "artifacts/retriever.pt"),
                      ("encoder", "artifacts/v2/user_encoder.npz"),
                      ("encoder_manifest", "artifacts/v2/user_encoder.npz.json")):
        add(name, retrieval[key])
    for path, digest in selected["model_files"].items():
        add(path, digest)
    for name in REPORTS:
        add(f"reports/v3/{name}.json")
    for path in ("artifacts/v2/selection.json", "artifacts/v3/selection.json"):
        add(path)
    add("artifacts/v3/tune/cache.json", tune["cache_manifest_sha256"])
    add("artifacts/v3/confirm/cache.json", confirm["cache_manifest_sha256"])
    service_files = sorted(expected)
    bound_extras = {**audit["verified_sha256_files"], **fit["input_files"]}
    for path in extra_files:
        path = relative_path(path)
        if path not in bound_extras:
            raise ValueError(f"extra input is not bound to the frozen fit/audit report: {path}")
        add(path, bound_extras[path])
    for path, digest in audit["verified_sha256_files"].items():
        if path in expected:
            add(path, digest)
    records = [_record(repo, path, expected[path]) for path in sorted(expected)]
    required = dict(selected["sources"])
    for path in (repo / "newsforge").rglob("*.py"):
        relative = path.relative_to(repo).as_posix()
        if not relative.startswith("newsforge/delivery/"):
            required.setdefault(relative, audit["verified_sha256_files"].get(
                relative, sha256_file(safe_path(repo, relative))))
    checkout = [_record(repo, path, digest) for path, digest in sorted(required.items())]
    commit = None
    executable = git_executable or shutil.which("git")
    if executable:
        result = subprocess.run([str(executable), "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", result.stdout.strip()):
            commit = result.stdout.strip()
    manifest = {"format": FORMAT, "scope": SCOPE,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_commit": commit, "source_commit_is_informational": True,
                "private_data": True, "redistribution": "local authorized use only; not for public upload",
                "files": records, "required_checkout_files": checkout,
                "total_bytes": sum(record["size"] for record in records),
                "extra_files": sorted(set(extra_files)),
                "payload_profiles": {"service_and_release_evidence": service_files,
                                     "experiment_input_extras": sorted(set(expected) - set(service_files))},
                "verification": "streamed SHA256 and byte counts; source hashes enforced; no deserialization",
                "omitted": ["raw downloaded MIND archives", "Python environment and site-packages",
                            "training caches/pairs and evaluation arrays not needed by v3.load_engine"]}
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest):
    if (manifest.get("format") != FORMAT or manifest.get("scope") != SCOPE
            or manifest.get("private_data") is not True):
        raise ValueError("unsupported bundle format/scope")
    all_names = set()
    for key, allowed in (("files", _payload_allowed), ("required_checkout_files", _checkout_allowed)):
        records = manifest.get(key)
        if not isinstance(records, list) or not records or len(records) > MAX_FILES:
            raise ValueError("invalid manifest inventory size")
        for record in records:
            if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
                raise ValueError("invalid manifest file record")
            name = relative_path(record["path"])
            if not allowed(name) or name.casefold() in all_names:
                raise ValueError("forbidden or duplicate/case-colliding manifest path")
            all_names.add(name.casefold())
            if (not isinstance(record["size"], int) or isinstance(record["size"], bool)
                    or not 0 <= record["size"] <= MAX_TOTAL_BYTES
                    or not isinstance(record["sha256"], str) or not _SHA.fullmatch(record["sha256"])):
                raise ValueError("invalid manifest size or SHA256")
    total = sum(record["size"] for record in manifest["files"])
    if (not isinstance(manifest.get("total_bytes"), int) or isinstance(manifest["total_bytes"], bool)
            or manifest["total_bytes"] != total or total > MAX_TOTAL_BYTES):
        raise ValueError("manifest total size exceeds limit or differs from inventory")


def _verify_record(root, record):
    path = safe_path(root, record["path"])
    if not path.is_file() or path.stat().st_size != record["size"]:
        raise ValueError(f"file size/missing mismatch: {record['path']}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"file SHA256 mismatch: {record['path']}")


def _copy_record(source_root, target_root, record):
    """Independent bytes, bounded streaming, exclusive no-overwrite install."""
    source = safe_path(source_root, record["path"])
    target = safe_path(target_root, record["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".delivery-part-" + uuid.uuid4().hex)
    digest, copied = hashlib.sha256(), 0
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while True:
                block = reader.read(min(CHUNK_BYTES, record["size"] - copied + 1))
                if not block:
                    break
                copied += len(block)
                if copied > record["size"]:
                    raise ValueError("source grew beyond manifest byte boundary")
                digest.update(block)
                writer.write(block)
        if copied != record["size"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"copy source SHA256/size mismatch: {record['path']}")
        # Atomic hard-link installation references our NEW temporary inode,
        # never source bytes. Removing the temporary name leaves one link.
        # Unlike replace/rename on POSIX this refuses even a racing target.
        safe_path(target_root, record["path"])
        os.link(temporary, target)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def export_bundle(repo, bundle, *, extra_files=(), git_executable=None):
    repo, bundle = Path(repo).absolute(), Path(bundle).absolute()
    if bundle.exists():
        raise FileExistsError("bundle destination already exists; choose a new directory")
    safe_path(bundle.parent, bundle.name)
    manifest = build_plan(repo, extra_files=extra_files, git_executable=git_executable)
    for record in manifest["required_checkout_files"]:
        _verify_record(repo, record)
    bundle.mkdir(parents=True)
    payload = bundle / "payload"
    payload.mkdir()
    # A failed export intentionally has no manifest and cannot be imported.
    for record in manifest["files"]:
        _copy_record(repo, payload, record)
    for record in manifest["required_checkout_files"]:
        _verify_record(repo, record)
    manifest_path = bundle / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    digest = sha256_file(manifest_path)
    (bundle / "manifest.sha256").write_text(digest + "\n", encoding="ascii")
    return {"scope": SCOPE, "manifest_sha256": digest, "files": len(manifest["files"]),
            "total_bytes": manifest["total_bytes"], "source_commit": manifest["source_commit"]}


def _read_bundle(bundle, expected_manifest_sha256=None):
    bundle = Path(bundle).absolute()
    manifest_path = safe_path(bundle, "manifest.json")
    if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("missing or oversized bundle manifest")
    digest = sha256_file(manifest_path)
    checksum = safe_path(bundle, "manifest.sha256")
    if not checksum.is_file() or checksum.stat().st_size > 128 or checksum.read_text(encoding="ascii").strip() != digest:
        raise ValueError("bundle manifest checksum mismatch")
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("bundle manifest differs from the trusted expected digest")
    manifest = _json(manifest_path)
    _validate_manifest(manifest)
    return manifest, digest


def verify_bundle(bundle, *, expected_manifest_sha256=None):
    manifest, digest = _read_bundle(bundle, expected_manifest_sha256)
    for record in manifest["files"]:
        _verify_record(Path(bundle) / "payload", record)
    return {"scope": SCOPE, "verified": True, "manifest_sha256": digest,
            "files": len(manifest["files"]), "total_bytes": manifest["total_bytes"]}


def import_bundle(bundle, repo, *, expected_manifest_sha256=None):
    """Preflight ALL conflicts and payload corruption before creating files.

    An interrupted copy may leave verified new files but never overwrites any
    file. Retry validates and skips identical files, so import is resumable.
    """
    bundle, repo = Path(bundle).absolute(), Path(repo).absolute()
    manifest, digest = _read_bundle(bundle, expected_manifest_sha256)
    for record in manifest["required_checkout_files"]:
        _verify_record(repo, record)
    existing = set()
    for record in manifest["files"]:
        destination = safe_path(repo, record["path"])
        if destination.exists():
            _verify_record(repo, record)
            existing.add(record["path"])
    for record in manifest["files"]:
        _verify_record(bundle / "payload", record)
    copied = 0
    for record in manifest["files"]:
        if record["path"] not in existing:
            _copy_record(bundle / "payload", repo, record)
            copied += 1
    for record in manifest["required_checkout_files"] + manifest["files"]:
        _verify_record(repo, record)
    return {"scope": SCOPE, "verified": True, "manifest_sha256": digest,
            "files": len(manifest["files"]), "copied_files": copied,
            "identical_existing_files": len(existing), "total_bytes": manifest["total_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "export", "verify", "import"))
    parser.add_argument("--repo", default=".")
    parser.add_argument("--bundle")
    parser.add_argument("--extra-file", action="append", default=[])
    parser.add_argument("--git-executable")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.command != "plan" and not args.bundle:
        parser.error("--bundle is required")
    if args.command == "plan":
        result = build_plan(args.repo, extra_files=args.extra_file, git_executable=args.git_executable)
    elif args.command == "export":
        result = export_bundle(args.repo, args.bundle, extra_files=args.extra_file,
                               git_executable=args.git_executable)
    elif args.command == "verify":
        result = verify_bundle(args.bundle, expected_manifest_sha256=args.expected_manifest_sha256)
    else:
        result = import_bundle(args.bundle, args.repo, expected_manifest_sha256=args.expected_manifest_sha256)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
