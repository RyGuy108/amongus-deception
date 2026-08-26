#!/usr/bin/env python3
"""Build Phase 8 taxonomy annotations, report, and human-review sheet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.analysis import annotate_taxonomy, taxonomy_report  # noqa: E402


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [
        record for record in read_jsonl(args.input) if record.get("status") == "collected"
    ]
    annotations = [annotate_taxonomy(record) for record in records]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "taxonomy-annotations.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for annotation in annotations:
            stream.write(json.dumps(annotation, sort_keys=True) + "\n")
    report = taxonomy_report(annotations)
    (args.output_dir / "taxonomy-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "taxonomy-human-validation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "experiment_id",
            "match_id",
            "condition",
            "role",
            "utterance",
            "automatic_primary",
            "automatic_categories",
            "human_primary",
            "human_secondary",
            "annotator",
            "notes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in annotations:
            writer.writerow(
                {
                    "experiment_id": item["source"]["experiment_id"],
                    "match_id": item["source"]["match_id"],
                    "condition": item["source"]["condition"],
                    "role": item["speaker"].get("role"),
                    "utterance": item["utterance"],
                    "automatic_primary": item["primary_category"],
                    "automatic_categories": "|".join(item["categories"]),
                    "human_primary": "",
                    "human_secondary": "",
                    "annotator": "",
                    "notes": "",
                }
            )
    print(json.dumps(report["primary_distribution"], sort_keys=True))


if __name__ == "__main__":
    main()
