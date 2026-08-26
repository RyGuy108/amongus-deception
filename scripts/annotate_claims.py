#!/usr/bin/env python3
"""Extract and ground atomic claims in existing turn-record JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.analysis import GroundingPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pipeline = GroundingPipeline()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    claims = 0
    labels: Counter[str] = Counter()
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "agent_turn":
                raise ValueError(f"Line {line_number} is not an agent_turn record")
            record["atomic_claims"] = pipeline.annotate_record(record)
            destination.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            destination.write("\n")
            records += 1
            claims += record["atomic_claims"]["summary"]["total"]
            labels.update(record["atomic_claims"]["summary"]["labels"])

    print(
        json.dumps(
            {
                "records": records,
                "claims": claims,
                "labels": dict(sorted(labels.items())),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
