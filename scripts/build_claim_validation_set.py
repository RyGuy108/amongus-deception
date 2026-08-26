#!/usr/bin/env python3
"""Build a deterministic, stratified human-review sheet from grounded claims."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_claims(path: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            for claim in record.get("atomic_claims", {}).get("items", []):
                rows.append((record, claim))
    return rows


def stratified_sample(
    rows: List[Tuple[Dict[str, Any], Dict[str, Any]]], target: int, seed: int
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        claim = row[1]
        groups[(claim["claim_type"], claim["preliminary_label"])].append(row)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    keys = sorted(groups)
    while len(selected) < min(target, len(rows)):
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < target:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.target <= 0:
        parser.error("--target must be positive")

    selected = stratified_sample(load_claims(args.input), args.target, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "claim_id",
        "game_index",
        "turn_index",
        "speaker",
        "utterance",
        "source_text",
        "claim_type_auto",
        "subject",
        "predicate",
        "object",
        "location",
        "objective_truth_auto",
        "speaker_support_auto",
        "preliminary_label_auto",
        "human_claim_correct",
        "human_claim_type",
        "human_objective_truth",
        "human_speaker_support",
        "human_notes",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for record, claim in selected:
            writer.writerow(
                {
                    "claim_id": claim["claim_id"],
                    "game_index": record["game_index"],
                    "turn_index": record["turn_index"],
                    "speaker": record["player"]["name"],
                    "utterance": record.get("utterance"),
                    "source_text": claim["source_text"],
                    "claim_type_auto": claim["claim_type"],
                    "subject": claim.get("subject"),
                    "predicate": claim["predicate"],
                    "object": claim.get("object"),
                    "location": claim.get("location"),
                    "objective_truth_auto": claim["objective_truth"]["status"],
                    "speaker_support_auto": claim["speaker_support"]["status"],
                    "preliminary_label_auto": claim["preliminary_label"],
                }
            )
    print(f"Wrote {len(selected)} claims to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
