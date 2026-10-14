"""Create a platform-specific hash lock from downloaded wheel bytes."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import zipfile


def wheel_records(directory):
    records = {}
    for path in sorted(Path(directory).glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist()
                     if name.endswith(".dist-info/METADATA") and name.count("/") == 1]
            if len(names) != 1:
                raise ValueError("wheel must contain exactly one distribution metadata file")
            metadata = BytesParser().parsebytes(archive.read(names[0]))
        if not metadata["Name"] or not metadata["Version"]:
            raise ValueError("wheel distribution metadata lacks Name or Version")
        name = re.sub(r"[-_.]+", "-", str(metadata["Name"])).lower()
        version = str(metadata["Version"])
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or not re.fullmatch(r"[A-Za-z0-9.+!_-]+", version):
            raise ValueError("invalid wheel name or version")
        if name in records:
            raise ValueError("duplicate wheel distribution: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        records[name] = {"name": name, "version": version, "filename": path.name,
                         "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if not records:
        raise ValueError("no downloaded wheels")
    return [records[name] for name in sorted(records)]


def render_lock(records):
    return ("# CPython 3.10, Windows x86-64. Generated from actual downloaded wheel bytes.\n"
            "# Install with --require-hashes; other platforms require a separately verified lock.\n"
            "--index-url https://pypi.org/simple\n"
            "--extra-index-url https://download.pytorch.org/whl/cpu\n"
            "--only-binary=:all:\n\n" + "".join(
                f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}\n" for row in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    lock, receipt = Path(args.lock), Path(args.receipt)
    if lock.exists() or receipt.exists():
        raise FileExistsError("retain previous lock and receipts; use a new destination")
    records = wheel_records(args.wheelhouse)
    content = render_lock(records).encode("utf-8")
    for path in (lock, receipt):
        path.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("xb") as stream:
        stream.write(content)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"schema": 1, "platform": "CPython 3.10 Windows x86-64",
                   "lock_sha256": hashlib.sha256(content).hexdigest(), "wheels": records}, stream, indent=2)
    print(json.dumps({"wheels": len(records), "lock_sha256": hashlib.sha256(content).hexdigest()}))


if __name__ == "__main__":
    main()
