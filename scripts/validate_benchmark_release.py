#!/usr/bin/env python3
"""Validate Phase 13 archive safety, checksums, and declared record counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def records(path: Path) -> int:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return sum(bool(line.strip()) for line in stream)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="amongus-validate-") as temp:
        root = Path(temp)
        with tarfile.open(args.archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe archive member: {member.name}")
            bundle.extractall(root, filter="data")
        packages = [path for path in root.iterdir() if path.is_dir()]
        if len(packages) != 1:
            raise ValueError("archive must contain exactly one package root")
        package = packages[0]
        manifest = json.loads((package / "MANIFEST.json").read_text())
        failures = []
        for item in manifest["files"]:
            path = package / item["path"]
            if not path.is_file():
                failures.append(f"missing {item['path']}")
            elif digest(path) != item["sha256"]:
                failures.append(f"checksum {item['path']}")
            elif path.stat().st_size != item["bytes"]:
                failures.append(f"size {item['path']}")
        for relative, details in manifest["datasets"].items():
            actual = records(package / relative)
            if actual != details["record_count"]:
                failures.append(
                    f"records {relative}: expected {details['record_count']}, got {actual}"
                )
        if failures:
            raise ValueError("; ".join(failures))
        print(
            json.dumps(
                {
                    "status": "valid",
                    "release_id": manifest["release_id"],
                    "file_count": len(manifest["files"]),
                    "dataset_count": len(manifest["datasets"]),
                    "archive_sha256": digest(args.archive),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
