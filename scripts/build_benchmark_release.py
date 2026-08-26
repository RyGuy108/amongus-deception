#!/usr/bin/env python3
"""Build a deterministic, checksum-manifested Phase 13 benchmark archive."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "amongus-belief-grounded-deception-v0.1.0"


def release_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError(
            "release ID must contain only letters, digits, dots, underscores, and hyphens"
        )
    return value


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sanitize(value: Any, workspace: Path) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item, workspace) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, workspace) for item in value]
    if isinstance(value, str):
        text = value.replace(str(workspace), "<WORKSPACE>")
        if text.startswith("local:") and "Qwen2.5-1.5B-Instruct" in text:
            return "local-model:Qwen2.5-1.5B-Instruct"
        return text
    return value


def copy_sanitized(source: Path, destination: Path, workspace: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        value = sanitize(json.loads(source.read_text()), workspace)
        destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    elif source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as incoming, destination.open(
            "w", encoding="utf-8"
        ) as outgoing:
            for line in incoming:
                if line.strip():
                    value = sanitize(json.loads(line), workspace)
                    outgoing.write(json.dumps(value, sort_keys=True) + "\n")
    else:
        shutil.copy2(source, destination)


def record_count(path: Path) -> int:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return sum(bool(line.strip()) for line in stream)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    return 0


def validation_gate_status(
    validation_dir: Path,
    packet_type: str,
    target: int,
    *,
    minimum: int | None = None,
) -> str:
    manifest_path = validation_dir / "validation-manifest.json"
    if not manifest_path.exists():
        return "unavailable"
    manifest = json.loads(manifest_path.read_text())
    count = int(manifest.get("packets", {}).get(packet_type, {}).get("item_count", 0))
    report_path = validation_dir / f"{packet_type}-validation-report.json"
    passed = False
    if report_path.exists():
        report = json.loads(report_path.read_text())
        passed = bool(report.get("gate", {}).get("passed"))
    if passed:
        return f"validated; {count} items resolved"
    if minimum is not None:
        minimum_status = "met" if count >= minimum else "not met"
        return (
            f"pending human validation; {count} items prepared; "
            f"minimum {minimum} {minimum_status}; stretch target {target}"
        )
    return f"pending human validation; {count} of target {target} items prepared"


def belief_reliability_gate_status(phase4_dir: Path | None) -> str:
    if phase4_dir is None:
        return "pending"
    report_path = phase4_dir / "belief-reliability-report.json"
    if not report_path.exists():
        return "collection included; aggregate decision unavailable"
    report = json.loads(report_path.read_text())
    decision = report.get("decision_gate", {}).get("recommendation", "unknown")
    contexts = report.get("frozen_context_count", 0)
    return f"completed on {contexts} contexts; {decision}"


def git_state() -> tuple[str | None, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, True


def normalized_archive(source_dir: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as tar:
                for path in sorted(source_dir.rglob("*")):
                    relative = Path(source_dir.name) / path.relative_to(source_dir)
                    info = tar.gettarinfo(str(path), arcname=str(relative))
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
                    else:
                        tar.addfile(info)
    os.replace(temporary, archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase6-9-dir", type=Path, required=True)
    parser.add_argument("--phase10-dir", type=Path, required=True)
    parser.add_argument("--phase11-report", type=Path, required=True)
    parser.add_argument("--phase12-dir", type=Path, required=True)
    parser.add_argument(
        "--secondary-judge-results",
        type=Path,
        help="Optional judge JSONL when it is stored outside --phase12-dir.",
    )
    parser.add_argument(
        "--activation-cost",
        type=Path,
        help="Optional activation-monitor cost report stored outside --phase12-dir.",
    )
    parser.add_argument("--human-validation-dir", type=Path, required=True)
    parser.add_argument(
        "--automatic-key",
        type=Path,
        help="Private automatic-label key path for explicit opt-in inclusion.",
    )
    parser.add_argument(
        "--include-automatic-key",
        action="store_true",
        help="Include the private key in the archive; do not use during blinded review.",
    )
    parser.add_argument(
        "--claim-validation-records",
        type=Path,
        help="Optional supplemental matched-state JSONL used by the larger claim packet.",
    )
    parser.add_argument(
        "--phase4-dir",
        type=Path,
        help="Optional completed Phase 4 controlled reliability experiment directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--release-id", type=release_id, default=RELEASE_ID)
    parser.add_argument("--release-date", default="2026-08-23")
    parser.add_argument("--include-activations", action="store_true")
    args = parser.parse_args()
    bundled_key = args.human_validation_dir / "automatic-label-key.csv"
    if args.include_automatic_key and not (
        (args.automatic_key and args.automatic_key.exists()) or bundled_key.exists()
    ):
        parser.error(
            "--include-automatic-key requires --automatic-key or a key in "
            "--human-validation-dir"
        )
    archive = args.output_dir / f"{args.release_id}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="amongus-release-") as temp:
        package = Path(temp) / args.release_id
        package.mkdir()
        direct_files = {
            ROOT / "benchmark" / "README.md": package / "README.md",
            ROOT / "benchmark" / "DATA_CARD.md": package / "DATA_CARD.md",
            ROOT / "benchmark" / "LEADERBOARD.md": package / "LEADERBOARD.md",
            ROOT / "benchmark" / "leaderboard-template.csv": package / "leaderboard-template.csv",
            ROOT / "HUMAN_VALIDATION_GUIDE.md": package / "validation" / "HUMAN_VALIDATION_GUIDE.md",
            ROOT / "LICENSE": package / "LICENSE",
            ROOT / "requirements-core.txt": package / "requirements-core.txt",
            ROOT / "requirements-research.txt": package / "requirements-research.txt",
            ROOT / "pyproject.toml": package / "pyproject.toml",
        }
        for source, destination in direct_files.items():
            copy_sanitized(source, destination, ROOT)
        for source in sorted(ROOT.glob("PHASES*.md")):
            copy_sanitized(source, package / "reports" / source.name, ROOT)
        for source in sorted((ROOT / "benchmark" / "configs").glob("*.json")):
            copy_sanitized(source, package / "configs" / source.name, ROOT)
        for source in sorted((ROOT / "docs" / "schemas").glob("*.json")):
            copy_sanitized(source, package / "schemas" / source.name, ROOT)

        secondary_judge = (
            args.secondary_judge_results
            or args.phase12_dir / "secondary-judge-results.jsonl"
        )
        data_files = {
            args.phase6_9_dir / "matched-incentive-trials.jsonl": package / "data" / "natural" / "matched-incentive-trials.jsonl",
            args.phase6_9_dir / "counterfactual-influence.jsonl": package / "data" / "natural" / "counterfactual-influence.jsonl",
            args.phase6_9_dir / "phase8" / "taxonomy-annotations.jsonl": package / "data" / "natural" / "taxonomy-candidates.jsonl",
            args.phase10_dir / "controlled-generalization-dataset.jsonl": package / "data" / "controlled" / "generalization.jsonl",
            args.phase12_dir / "control-decisions.jsonl": package / "data" / "controlled" / "control-decisions.jsonl",
            secondary_judge: package / "data" / "controlled" / "secondary-judge-results.jsonl",
        }
        if args.claim_validation_records:
            data_files[args.claim_validation_records] = (
                package / "data" / "natural" / "claim-validation-supplement.jsonl"
            )
        if args.phase4_dir:
            phase4_contexts = args.phase4_dir / "belief-reliability-contexts.jsonl"
            if phase4_contexts.exists():
                data_files[phase4_contexts] = (
                    package / "data" / "controlled" / "belief-reliability-contexts.jsonl"
                )
        for source, destination in data_files.items():
            copy_sanitized(source, destination, ROOT)
        if args.include_activations:
            shutil.copy2(args.phase10_dir / "activations.npz", package / "data" / "controlled" / "activations.npz")

        report_files = {
            args.phase6_9_dir / "matched-incentive-report.json": "phase6-matched-incentive-report.json",
            args.phase6_9_dir / "counterfactual-influence-report.json": "phase7-counterfactual-influence-report.json",
            args.phase6_9_dir / "phase8" / "taxonomy-report.json": "phase8-taxonomy-report.json",
            args.phase6_9_dir / "phase9" / "mechanistic-probe-report.json": "phase9-mechanistic-probe-report.json",
            args.phase10_dir / "frozen-generalization-report.json": "phase10-frozen-generalization-report.json",
            args.phase11_report: "phase11-research-branch-report.json",
            args.phase12_dir / "control-intervention-report.json": "phase12-control-intervention-report.json",
        }
        activation_cost = args.activation_cost or (
            args.phase12_dir / "activation-monitor-cost.json"
        )
        if activation_cost.exists():
            report_files[activation_cost] = "phase12-activation-monitor-cost.json"
        cross_model = args.phase10_dir / "cross-model-replication-report.json"
        if cross_model.exists():
            report_files[cross_model] = "phase10-cross-model-replication-report.json"
        if args.phase4_dir:
            phase4_reports = {
                args.phase4_dir / "collection-manifest.json": "phase4-collection-manifest.json",
                args.phase4_dir / "belief-reliability-report.json": "phase4-belief-reliability-report.json",
            }
            report_files.update(
                {source: name for source, name in phase4_reports.items() if source.exists()}
            )
        for source, name in report_files.items():
            copy_sanitized(source, package / "reports" / name, ROOT)

        for source in sorted(args.human_validation_dir.glob("*.csv")):
            if source.name == "automatic-label-key.csv" and not args.include_automatic_key:
                continue
            folder = "private" if source.name == "automatic-label-key.csv" else "packets"
            copy_sanitized(source, package / "validation" / folder / source.name, ROOT)
        if args.automatic_key and args.include_automatic_key:
            copy_sanitized(
                args.automatic_key,
                package / "validation" / "private" / "automatic-label-key.csv",
                ROOT,
            )
        copy_sanitized(
            args.human_validation_dir / "validation-manifest.json",
            package / "validation" / "validation-manifest.json",
            ROOT,
        )

        source_roots = [
            ROOT / "among-agents" / "amongagents" / "analysis",
            ROOT / "among-agents" / "amongagents" / "control",
            ROOT / "among-agents" / "amongagents" / "interpretability",
            ROOT / "among-agents" / "amongagents" / "validation",
            ROOT / "among-agents" / "amongagents" / "models",
            ROOT / "scripts",
            ROOT / "tests",
        ]
        for source_root in source_roots:
            for source in sorted(source_root.rglob("*.py")):
                relative = source.relative_to(ROOT)
                copy_sanitized(source, package / "source" / relative, ROOT)

        revision, dirty = git_state()
        datasets = {}
        for path in sorted((package / "data").rglob("*")):
            if path.is_file() and path.suffix in {".jsonl", ".csv"}:
                relative = str(path.relative_to(package))
                datasets[relative] = {"record_count": record_count(path)}
        files = []
        for path in sorted(package.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.json":
                files.append(
                    {
                        "path": str(path.relative_to(package)),
                        "sha256": digest(path),
                        "bytes": path.stat().st_size,
                    }
                )
        manifest = {
            "schema_version": "amongus.benchmark-release-manifest.v1",
            "release_id": args.release_id,
            "release_date": args.release_date,
            "source_revision": revision,
            "working_tree_dirty": dirty,
            "files": files,
            "datasets": datasets,
            "known_gates": {
                "natural_claim_validation": validation_gate_status(
                    args.human_validation_dir, "claims", 400, minimum=300
                ),
                "natural_taxonomy_validation": validation_gate_status(
                    args.human_validation_dir, "taxonomy", 120, minimum=120
                ),
                "control_utility_validation": validation_gate_status(
                    args.human_validation_dir, "control", 120, minimum=120
                ),
                "belief_reliability": belief_reliability_gate_status(
                    args.phase4_dir
                ),
                "activation_monitor_cost": (
                    "measured_local" if activation_cost.exists() else "pending"
                ),
                "model_ood": (
                    "coordinate-compatible replication included"
                    if cross_model.exists()
                    else "unavailable"
                ),
                "deployment_ready": False,
                "automatic_label_key_included": args.include_automatic_key,
            },
        }
        (package / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        normalized_archive(package, archive)
    sidecar = args.output_dir / f"{args.release_id}.sha256"
    sidecar.write_text(f"{digest(archive)}  {archive.name}\n")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": digest(archive),
                "bytes": archive.stat().st_size,
                "datasets": datasets,
                "working_tree_dirty": dirty,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
