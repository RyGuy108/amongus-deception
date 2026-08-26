#!/usr/bin/env python3
"""Aggregate Phase 4 stability, method agreement, and calibration evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.measurement.reliability import assess_method_agreement


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error


def calibration_summary(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = []
    for record in records:
        values = record.get("private_belief", {}).get("values") or {}
        roles = {
            player["name"]: player.get("role")
            for player in record.get("world_state_before", {}).get("players", [])
        }
        for player, probability in values.items():
            if player in roles:
                pairs.append((float(probability), 1.0 if roles[player] == "Impostor" else 0.0))
    bins = []
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        if upper == 1.0:
            members = [pair for pair in pairs if lower <= pair[0] <= upper]
        else:
            members = [pair for pair in pairs if lower <= pair[0] < upper]
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_prediction": sum(pair[0] for pair in members) / len(members) if members else None,
                "empirical_frequency": sum(pair[1] for pair in members) / len(members) if members else None,
            }
        )
    return {
        "prediction_count": len(pairs),
        "brier_score": (
            sum((prediction - label) ** 2 for prediction, label in pairs) / len(pairs)
            if pairs
            else None
        ),
        "bins": bins,
        "note": "Ground-truth role calibration; repeated turns are not statistically independent.",
    }


def analyze(records: list[Mapping[str, Any]], min_contexts: int = 30) -> Dict[str, Any]:
    reliability_analyses = []
    method_assessments = []
    for record in records:
        belief = record.get("private_belief", {})
        suite = belief.get("measurements", {}).get("reliability")
        if suite and suite.get("analysis"):
            reliability_analyses.append(suite["analysis"])
        triangulation = belief.get("triangulation", {})
        if len(triangulation.get("available_methods", [])) >= 2:
            adjusted = dict(triangulation)
            behavioral = belief.get("measurements", {}).get("behavioral_fork", {})
            if behavioral.get("scoring_note", "").startswith("heuristic"):
                adjusted["probability_l1_comparable"] = False
            method_assessments.append(assess_method_agreement(adjusted))

    stability_branches = Counter(
        analysis.get("gate", {}).get("branch", "unknown")
        for analysis in reliability_analyses
    )
    method_branches = Counter(item["branch"] for item in method_assessments)
    valid_samples = sum(item.get("valid_count", 0) for item in reliability_analyses)
    total_samples = sum(item.get("sample_count", 0) for item in reliability_analyses)
    stable_count = stability_branches["A_stable_self_report"]
    disagreement_count = method_branches["C_methods_disagree"]
    stable_rate = (
        stable_count / len(reliability_analyses) if reliability_analyses else None
    )
    disagreement_rate = (
        disagreement_count / len(method_assessments) if method_assessments else None
    )

    if len(reliability_analyses) < min_contexts:
        recommendation = "collect_more_contexts"
        reasons = [
            f"Only {len(reliability_analyses)} frozen contexts; gate requires {min_contexts}."
        ]
    else:
        if method_assessments and float(disagreement_rate or 0.0) > 0.20:
            recommendation = "C_investigate_belief_report_faithfulness"
            reasons = ["More than 20% of comparable contexts show cross-method disagreement."]
        elif float(stable_rate or 0.0) >= 0.80:
            recommendation = "A_use_elicitation_as_primary_with_auxiliary_checks"
            reasons = ["At least 80% of frozen contexts passed the stability gate."]
        else:
            recommendation = "B_emphasize_behavioral_and_activation_measures"
            reasons = ["Fewer than 80% of frozen contexts passed the stability gate."]

    return {
        "schema_version": "amongus.belief-reliability-report.v1",
        "record_count": len(records),
        "frozen_context_count": len(reliability_analyses),
        "sample_validity": {
            "valid": valid_samples,
            "total": total_samples,
            "valid_rate": valid_samples / total_samples if total_samples else None,
        },
        "stability_branches": dict(sorted(stability_branches.items())),
        "stability_rate": stable_rate,
        "method_agreement_contexts": len(method_assessments),
        "method_branches": dict(sorted(method_branches.items())),
        "method_disagreement_rate": disagreement_rate,
        "method_comparison_note": (
            "Behavioral-fork scores are heuristic ranks, so numeric probability L1 is retained "
            "only as a diagnostic; the cross-method gate uses top-suspect agreement."
        ),
        "calibration": calibration_summary(records),
        "decision_gate": {
            "minimum_contexts": min_contexts,
            "recommendation": recommendation,
            "reasons": reasons,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-contexts", type=int, default=30)
    args = parser.parse_args()
    report = analyze(list(read_jsonl(args.input)), args.min_contexts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} ({report['frozen_context_count']} frozen contexts)")


if __name__ == "__main__":
    main()
