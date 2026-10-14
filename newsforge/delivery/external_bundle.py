"""Transfer exact frozen MIND prerequisites for the Adressa experiment.

This separate private bundle contains no Adressa events. Code and protocols must
already exist in the receiving checkout and are never installed by this tool.
The two explicitly allowed local joblib files are copied as opaque bytes, never
loaded. Verify/import require a manifest SHA256 obtained from a trusted source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from newsforge.delivery import bundle as b


FORMAT = "newsforge-adressa-frozen-mind-private-bundle-v1"
SCOPE = "Exact preregistered MIND inputs for external Adressa evaluation; no external events, fitting or evaluation"
SOURCE_REPORT = "reports/external/adressa_source.json"
MAX_TOTAL_BYTES = 256 * 1024**2
CHECKOUT_FILES = frozenset(("docs/EXTERNAL_ADRESSA_PROTOCOL.md", "docs/V3_PROTOCOL.md",
                           "newsforge/data.py", "newsforge/v2/ranker.py", "newsforge/v2/temporal.py",
                           "newsforge/v2/retrieval.py", "newsforge/v3/residual.py",
                           "newsforge/v3/fast_retrieval.py"))
PAYLOAD_FILES = frozenset(("data/processed/tfidf.joblib", "data/processed/svd.joblib",
                          "data/processed/category_vocabulary.json", "data/processed/manifest.json",
                          "artifacts/v2/selection.json", "artifacts/v3/selection.json",
                          "reports/v3/fit.json", "reports/v3/tune.json", "reports/v3/confirm.json")) | frozenset(
    f"artifacts/v3/models/{variant}-seed{seed}.npz{suffix}"
    for variant in ("full13", "no_category", "no_recent")
    for seed in (17, 29, 43) for suffix in ("", ".json"))


def validate_record(record):
    if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
        raise ValueError("invalid frozen-input file record")
    b.relative_path(record["path"])
    if (not isinstance(record["size"], int) or isinstance(record["size"], bool)
            or not 0 < record["size"] <= MAX_TOTAL_BYTES
            or not isinstance(record["sha256"], str) or not b._SHA.fullmatch(record["sha256"])):
        raise ValueError("invalid frozen-input size or SHA256")


def validate_manifest(manifest):
    if (set(manifest) != {"format", "scope", "private_data", "files", "required_checkout_files", "total_bytes"}
            or manifest.get("format") != FORMAT or manifest.get("scope") != SCOPE
            or manifest.get("private_data") is not True):
        raise ValueError("unsupported external bundle format or scope")
    for key, expected in (("files", PAYLOAD_FILES),
                          ("required_checkout_files", CHECKOUT_FILES | {SOURCE_REPORT})):
        records = manifest.get(key)
        if not isinstance(records, list) or len(records) != len(expected):
            raise ValueError("incomplete external bundle inventory")
        for record in records:
            validate_record(record)
        names = [record["path"] for record in records]
        if len(set(name.casefold() for name in names)) != len(names) or set(names) != expected:
            raise ValueError("external bundle contains forbidden, duplicate or missing paths")
        if names != sorted(names):
            raise ValueError("external bundle inventory must use canonical order")
    total = sum(record["size"] for record in manifest["files"])
    if (not isinstance(manifest["total_bytes"], int) or isinstance(manifest["total_bytes"], bool)
            or manifest["total_bytes"] != total or total > MAX_TOTAL_BYTES):
        raise ValueError("external bundle total size differs or exceeds the limit")


def expected_manifest(repo):
    """Build expectations only from the complete preregistered source inventory."""
    source_path = b.safe_path(repo, SOURCE_REPORT)
    source = b._json(source_path)
    records = source.get("frozen_mind_and_protocol_files")
    if (source.get("schema_version") != 1 or not isinstance(records, dict)
            or set(records) != PAYLOAD_FILES | CHECKOUT_FILES):
        raise ValueError("preregistered source report has missing or unsupported frozen inputs")
    files, checkout = [], []
    for name, value in sorted(records.items()):
        if not isinstance(value, dict) or set(value) != {"bytes", "sha256"}:
            raise ValueError("invalid preregistered frozen-input record")
        record = {"path": name, "size": value["bytes"], "sha256": value["sha256"]}
        validate_record(record)
        (files if name in PAYLOAD_FILES else checkout).append(record)
    checkout.append({"path": SOURCE_REPORT, "size": source_path.stat().st_size,
                     "sha256": b.sha256_file(source_path)})
    manifest = {"format": FORMAT, "scope": SCOPE, "private_data": True,
                "files": files, "required_checkout_files": sorted(checkout, key=lambda row: row["path"]),
                "total_bytes": sum(row["size"] for row in files)}
    validate_manifest(manifest)
    return manifest


def verify_checkout(repo, manifest):
    # These are existing checkout files, never payload. Check the source report
    # bytes before trusting its declared model and representation digests.
    for record in manifest["required_checkout_files"]:
        b._verify_record(repo, record)
    if expected_manifest(repo) != manifest:
        raise ValueError("bundle inventory differs from the receiving preregistration")


def export_bundle(repo, bundle):
    repo, bundle = Path(repo).absolute(), Path(bundle).absolute()
    if bundle.exists():
        raise FileExistsError("retain existing bundles; choose a new destination")
    b.safe_path(bundle.parent, bundle.name)
    manifest = expected_manifest(repo)
    verify_checkout(repo, manifest)
    # All inputs are checked before beginning an export; a later interrupted
    # copy has no finalized manifest and cannot be imported as a valid bundle.
    for record in manifest["files"]:
        b._verify_record(repo, record)
    bundle.mkdir(parents=True)
    payload = bundle / "payload"
    payload.mkdir()
    for record in manifest["files"]:
        b._copy_record(repo, payload, record)
    verify_checkout(repo, manifest)
    for record in manifest["files"]:
        b._verify_record(repo, record)
        b._verify_record(payload, record)
    manifest_path = bundle / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    digest = b.sha256_file(manifest_path)
    with (bundle / "manifest.sha256").open("x", encoding="ascii") as stream:
        stream.write(digest + "\n")
    return summary(manifest, digest, "export")


def read_bundle(bundle, expected_manifest_sha256):
    if not isinstance(expected_manifest_sha256, str) or not b._SHA.fullmatch(expected_manifest_sha256):
        raise ValueError("a trusted complete manifest SHA256 is required")
    manifest_path = b.safe_path(bundle, "manifest.json")
    manifest = b._json(manifest_path)
    digest = b.sha256_file(manifest_path)
    if digest != expected_manifest_sha256:
        raise ValueError("external bundle differs from the trusted manifest SHA256")
    checksum = b.safe_path(bundle, "manifest.sha256")
    if (not checksum.is_file() or checksum.stat().st_size > 128
            or checksum.read_text(encoding="ascii").strip() != digest):
        raise ValueError("external bundle manifest checksum mismatch")
    validate_manifest(manifest)
    return manifest, digest


def summary(manifest, digest, action):
    return {"scope": SCOPE, "action": action, "verified": True,
            "manifest_sha256": digest, "payload_files": len(manifest["files"]),
            "required_checkout_files": len(manifest["required_checkout_files"]),
            "total_bytes": manifest["total_bytes"], "external_events_included": False,
            "model_deserialization_performed": False}


def verify_bundle(repo, bundle, *, expected_manifest_sha256):
    manifest, digest = read_bundle(bundle, expected_manifest_sha256)
    verify_checkout(repo, manifest)
    for record in manifest["files"]:
        b._verify_record(Path(bundle) / "payload", record)
    verify_checkout(repo, manifest)
    return summary(manifest, digest, "verify")


def import_bundle(repo, bundle, *, expected_manifest_sha256):
    repo, bundle = Path(repo).absolute(), Path(bundle).absolute()
    if repo.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("receiving checkout must be outside the bundle")
    manifest, digest = read_bundle(bundle, expected_manifest_sha256)
    verify_checkout(repo, manifest)
    existing = set()
    # Preflight every destination and every payload before creating any file.
    for record in manifest["files"]:
        target = b.safe_path(repo, record["path"])
        if target.exists():
            b._verify_record(repo, record)
            existing.add(record["path"])
    for record in manifest["files"]:
        b._verify_record(bundle / "payload", record)
    for record in manifest["files"]:
        if record["path"] not in existing:
            b._copy_record(bundle / "payload", repo, record)
    verify_checkout(repo, manifest)
    for record in manifest["files"]:
        b._verify_record(repo, record)
    return {**summary(manifest, digest, "import"),
            "copied_files": len(manifest["files"]) - len(existing),
            "identical_existing_files": len(existing)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "verify", "import"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.command == "export":
        if args.expected_manifest_sha256 is not None:
            parser.error("export produces a new manifest SHA256; no expected value is accepted")
        result = export_bundle(args.repo, args.bundle)
    else:
        if args.expected_manifest_sha256 is None:
            parser.error("verify/import require --expected-manifest-sha256 from a trusted source")
        action = verify_bundle if args.command == "verify" else import_bundle
        result = action(args.repo, args.bundle, expected_manifest_sha256=args.expected_manifest_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
