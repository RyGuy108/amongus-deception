#!/usr/bin/env python3
"""Check one blinded human-validation CSV without opening the private key."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.validation import audit_validation_packet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--expected-annotator")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_validation_packet(
        args.packet,
        expected_annotator=args.expected_annotator,
        require_complete=args.require_complete,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_complete and not report["ready_for_independent_agreement"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
