#!/usr/bin/env python3
"""Apply the Phase 11 decision gate to Phase 8–10 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.analysis import select_research_branch  # noqa: E402


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8-report", type=Path, required=True)
    parser.add_argument("--phase9-report", type=Path, required=True)
    parser.add_argument("--phase10-report", type=Path, required=True)
    parser.add_argument(
        "--phase4-report",
        type=Path,
        help="Optional completed Phase 4 report for a later adaptive update.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select_research_branch(
        load(args.phase8_report),
        load(args.phase9_report),
        load(args.phase10_report),
        load(args.phase4_report) if args.phase4_report else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "primary_branch": report["primary_branch"],
                "name": report["primary_branch_name"],
                "ranking": report["ranking"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
