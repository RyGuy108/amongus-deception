#!/usr/bin/env python3
"""Audit the role-by-incentive design matrix in structured turn records."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.envs.incentives import CONDITIONS


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def audit(records: Iterable[Mapping[str, Any]], min_cell: int = 5) -> Dict[str, Any]:
    assignments = {}
    plans = set()
    for record in records:
        key = (record.get("game_index"), record.get("player", {}).get("name"))
        incentive = record.get("incentive", {})
        assignments[key] = {
            "role": record.get("player", {}).get("role"),
            "condition": incentive.get("condition", "missing"),
            "target_player": incentive.get("target_player"),
            "target_role": incentive.get("target_role_ground_truth"),
        }
        plans.add(record.get("incentive", {}).get("plan", "missing"))
    counts = Counter(
        (assignment["role"], assignment["condition"])
        for assignment in assignments.values()
    )
    roles = sorted({role for role, _ in counts if role})
    observed_conditions = {condition for _, condition in counts if condition}
    conditions = sorted(
        set(CONDITIONS) if "balanced_factorial" in plans else observed_conditions
    )
    matrix = {
        role: {condition: counts[(role, condition)] for condition in conditions}
        for role in roles
    }
    total = sum(counts.values())
    row_totals = {role: sum(matrix[role].values()) for role in roles}
    column_totals = {
        condition: sum(matrix[role][condition] for role in roles)
        for condition in conditions
    }
    chi_square = 0.0
    if total:
        for role in roles:
            for condition in conditions:
                expected = row_totals[role] * column_totals[condition] / total
                if expected:
                    chi_square += (matrix[role][condition] - expected) ** 2 / expected
    denominator = total * min(max(len(roles) - 1, 0), max(len(conditions) - 1, 0))
    cramers_v = math.sqrt(chi_square / denominator) if denominator else None
    missing_cells = [
        {"role": role, "condition": condition, "count": matrix[role][condition]}
        for role in roles
        for condition in conditions
        if matrix[role][condition] < min_cell
    ]
    target_role_counts = Counter(
        (
            assignment["role"],
            assignment["condition"],
            assignment["target_role"] or "none",
        )
        for assignment in assignments.values()
    )
    target_role_distribution = {
        role: {
            condition: {
                target_role: target_role_counts[(role, condition, target_role)]
                for target_role in ("Crewmate", "Impostor", "none")
                if target_role_counts[(role, condition, target_role)]
            }
            for condition in conditions
        }
        for role in roles
    }
    return {
        "schema_version": "amongus.incentive-balance-audit.v1",
        "unique_agent_games": total,
        "roles": roles,
        "conditions": conditions,
        "plans": sorted(plans),
        "contingency_table": matrix,
        "cramers_v": cramers_v,
        "minimum_cell_count": min_cell,
        "gate_ready": bool(total and len(roles) >= 2 and len(conditions) >= 2 and not missing_cells),
        "underfilled_cells": missing_cells,
        "target_role_distribution": target_role_distribution,
        "note": "Cramer's V describes realized role/condition association; it is not a causal result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cell", type=int, default=5)
    args = parser.parse_args()
    report = audit(read_jsonl(args.input), args.min_cell)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} ({report['unique_agent_games']} agent-games)")


if __name__ == "__main__":
    main()
