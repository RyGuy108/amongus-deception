#!/usr/bin/env python3
"""Analyze independent human packets and create an adjudication sheet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.validation import (  # noqa: E402
    analyze_packets,
    audit_validation_packet,
)


def require_ready(path: Path, annotator: str) -> None:
    audit = audit_validation_packet(
        path,
        expected_annotator=annotator,
        require_complete=True,
    )
    if audit["ready_for_independent_agreement"]:
        return
    first = audit["issues"][0] if audit["issues"] else None
    detail = first["message"] if first else f"{audit['remaining_count']} rows remain"
    raise ValueError(f"{path.name} is not ready for agreement analysis: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-disagreements", type=Path, required=True)
    args = parser.parse_args()
    require_ready(args.annotator_a, "A")
    require_ready(args.annotator_b, "B")
    report, disagreements = analyze_packets(
        args.annotator_a,
        args.annotator_b,
        args.key,
        adjudication_path=args.adjudication,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = [
        "packet_type", "item_id", "field", "annotator_a", "annotator_b",
        "adjudicated_value", "adjudicator", "rationale",
    ]
    with args.output_disagreements.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disagreements)
    print(json.dumps({"gate": report["gate"], "disagreements": len(disagreements)}, sort_keys=True))


if __name__ == "__main__":
    main()
