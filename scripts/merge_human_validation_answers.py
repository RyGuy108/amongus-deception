#!/usr/bin/env python3
"""Copy completed human answers into a larger blinded packet by stable item_id."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def merge_answers(source: Path, target: Path, output: Path) -> dict[str, object]:
    source_fields, source_rows = _read(source)
    target_fields, target_rows = _read(target)
    if "item_id" not in source_fields or "item_id" not in target_fields:
        raise ValueError("source and target packets must contain item_id")
    source_types = {row.get("packet_type", "") for row in source_rows}
    target_types = {row.get("packet_type", "") for row in target_rows}
    if len(source_types) != 1 or source_types != target_types:
        raise ValueError("source and target must contain the same single packet_type")
    review_fields = [
        field
        for field in source_fields
        if field.startswith("human_") or field == "annotator_id"
    ]
    missing_fields = sorted(set(review_fields) - set(target_fields))
    if missing_fields:
        raise ValueError(f"target is missing answer fields: {missing_fields}")
    source_by_id = {row["item_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source packet contains duplicate item_id values")

    copied_items = 0
    copied_fields = 0
    matched_items = 0
    for row in target_rows:
        old = source_by_id.get(row["item_id"])
        if old is None:
            continue
        matched_items += 1
        item_copied = False
        for field in review_fields:
            source_value = str(old.get(field, ""))
            target_value = str(row.get(field, ""))
            if not source_value.strip():
                continue
            if target_value.strip() and target_value != source_value:
                raise ValueError(
                    f"conflicting existing answer for {row['item_id']} field {field}"
                )
            if not target_value.strip():
                row[field] = source_value
                copied_fields += 1
                item_copied = True
        copied_items += int(item_copied)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(target_rows)
    return {
        "source_items": len(source_rows),
        "target_items": len(target_rows),
        "matched_items": matched_items,
        "copied_items": copied_items,
        "copied_fields": copied_fields,
        "unmatched_source_items": len(source_rows) - matched_items,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.source.resolve(), args.target.resolve()}:
        parser.error("--output must be a new path; source packets are never overwritten")
    print(json.dumps(merge_answers(args.source, args.target, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
