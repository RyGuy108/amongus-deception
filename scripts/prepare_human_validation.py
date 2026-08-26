#!/usr/bin/env python3
"""Create blinded two-annotator packets for all outstanding human gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.validation import prepare_validation_packets  # noqa: E402


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claim-records",
        type=Path,
        required=True,
        action="append",
        help="JSONL claim source; repeat to combine non-overlapping collections.",
    )
    parser.add_argument("--taxonomy-annotations", type=Path, required=True)
    parser.add_argument("--taxonomy-records", type=Path, required=True)
    parser.add_argument("--control-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--automatic-key-output",
        type=Path,
        help="Optional path outside the annotator packet directory for the private key.",
    )
    parser.add_argument("--control-target", type=int, default=120)
    parser.add_argument(
        "--minimum-claims",
        type=int,
        default=0,
        help="Refuse packet creation when extracted claim count is below this gate.",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    claim_records = [
        record
        for path in args.claim_records
        for record in read_jsonl(path)
    ]
    claim_count = sum(
        len(record.get("atomic_claims", {}).get("items", []))
        for record in claim_records
    )
    if claim_count < args.minimum_claims:
        parser.error(
            f"combined claim sources contain {claim_count} claims; "
            f"--minimum-claims requires {args.minimum_claims}"
        )
    manifest = prepare_validation_packets(
        claim_records,
        list(read_jsonl(args.taxonomy_annotations)),
        list(read_jsonl(args.taxonomy_records)),
        list(read_jsonl(args.control_decisions)),
        args.output_dir,
        control_target=args.control_target,
        seed=args.seed,
        automatic_key_path=args.automatic_key_output,
    )
    print(json.dumps(manifest["packets"], sort_keys=True))


if __name__ == "__main__":
    main()
