#!/usr/bin/env python3
"""Aggregate Phase 7 belief and vote effects on claim-target listeners."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error


def _summary(values: list[float]) -> Dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
        "standard_error": (
            statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
        ),
    }


def analyze(
    records: list[Mapping[str, Any]],
    *,
    min_statements: int = 30,
    labels_validated: bool = False,
) -> Dict[str, Any]:
    expected_interventions = {"original", "neutral", "truthful_correction"}
    integrity_failures = Counter()
    eligible_records = []
    for record in records:
        failures = []
        if set(record.get("interventions", {})) != expected_interventions:
            failures.append("intervention_coverage")
        if not record.get("listeners"):
            failures.append("no_listeners")
        for listener in record.get("listeners", []):
            if set(listener.get("interventions", {})) != expected_interventions:
                failures.append("listener_intervention_coverage")
            if set(listener.get("presentation_order", [])) != expected_interventions:
                failures.append("presentation_order")
            if not listener.get("context_fingerprint"):
                failures.append("listener_context_fingerprint")
        if failures:
            integrity_failures.update(failures)
        else:
            eligible_records.append(record)

    effect_values: Dict[str, Dict[str, Dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    selection_labels = Counter()
    valid_listener_measurements = 0
    total_listener_measurements = 0
    replay_statuses = Counter()
    statement_sources = Counter()
    deceptive_scores_by_source: Dict[str, list[float]] = defaultdict(list)
    score_direction_by_source: Dict[str, str] = {}
    target_effect_source_ids = set()
    for record in eligible_records:
        source_id = (
            record.get("selection", {}).get("source_record_id")
            or record.get("influence_id")
        )
        selection_labels.update(record["selection"]["important_labels"])
        statement_sources[
            record["selection"].get("statement_source", "unspecified")
        ] += 1
        replay_statuses[record["outcome_replay"]["status"]] += 1
        for value in record.get("deceptive_influence_scores", {}).values():
            deceptive_scores_by_source[source_id].append(
                value["immediate_deceptive_influence_score"]
            )
            score_direction_by_source.setdefault(
                source_id, value.get("claim_direction", "unknown")
            )
        subjects = record["selection"].get("claim_subjects", [])
        for listener in record.get("listeners", []):
            total_listener_measurements += 1
            valid_listener_measurements += (
                listener["baseline_no_statement"].get("status") == "collected"
            )
            for intervention in listener["interventions"].values():
                total_listener_measurements += 1
                valid_listener_measurements += (
                    intervention["measurement"].get("status") == "collected"
                )
        for contrast_name, by_player in record.get("causal_contrasts", {}).items():
            for subject in subjects:
                values = by_player.get(subject)
                if not values:
                    continue
                effect_values[contrast_name]["target_belief_delta"][source_id].append(
                    values["mean_delta_impostor_probability"]
                )
                effect_values[contrast_name]["target_vote_delta"][source_id].append(
                    values["mean_delta_vote_probability"]
                )
        neutral_effects = record.get("causal_contrasts", {}).get(
            "original_minus_neutral", {}
        )
        if any(subject in neutral_effects for subject in subjects):
            target_effect_source_ids.add(
                source_id
            )

    deceptive_influence_scores = [
        statistics.fmean(values)
        for values in deceptive_scores_by_source.values()
        if values
    ]
    scores_by_direction: Dict[str, list[float]] = defaultdict(list)
    for source_id, values in deceptive_scores_by_source.items():
        if values:
            scores_by_direction[score_direction_by_source[source_id]].append(
                statistics.fmean(values)
            )

    validity_rate = (
        valid_listener_measurements / total_listener_measurements
        if total_listener_measurements
        else None
    )
    enough_data = len(eligible_records) >= min_statements
    target_effect_statement_count = len(target_effect_source_ids)
    enough_target_effects = target_effect_statement_count >= min_statements
    measurement_ready = validity_rate is not None and validity_rate >= 0.90
    gate_passed = (
        enough_data and enough_target_effects and measurement_ready and labels_validated
    )
    collection_ready = enough_data and enough_target_effects and measurement_ready
    reasons = []
    if not enough_data:
        reasons.append(f"Need at least {min_statements} eligible statements.")
    if not measurement_ready:
        reasons.append("Listener measurement validity must be at least 90%.")
    if not enough_target_effects:
        reasons.append(
            f"Need at least {min_statements} statements with a computable claim-target effect."
        )
    if not labels_validated:
        reasons.append("Candidate-deception selections require human validation.")
    return {
        "schema_version": "amongus.counterfactual-influence-report.v1",
        "statement_count": len(records),
        "integrity_valid_statement_count": len(eligible_records),
        "integrity_failure_counts": dict(sorted(integrity_failures.items())),
        "selection_labels": dict(sorted(selection_labels.items())),
        "statement_sources": dict(sorted(statement_sources.items())),
        "target_effect_statement_count": target_effect_statement_count,
        "listener_measurement_validity": {
            "valid": valid_listener_measurements,
            "total": total_listener_measurements,
            "valid_rate": validity_rate,
        },
        "claim_target_effects": {
            contrast: {
                metric: _summary(
                    [statistics.fmean(replicates) for replicates in by_source.values()]
                )
                for metric, by_source in sorted(metrics.items())
            }
            for contrast, metrics in sorted(effect_values.items())
        },
        "immediate_deceptive_influence_score": _summary(
            deceptive_influence_scores
        ),
        "direction_stratified_influence": {
            direction: _summary(values)
            for direction, values in sorted(scores_by_direction.items())
        },
        "outcome_replay_statuses": dict(sorted(replay_statuses.items())),
        "decision_gate": {
            "minimum_statements": min_statements,
            "labels_validated": labels_validated,
            "collection_and_measurement_ready": collection_ready,
            "passed": gate_passed,
            "reasons": reasons,
            "recommendation": (
                "estimate_deceptive_influence"
                if gate_passed
                else "collect_validate_or_repair_before_inference"
            ),
        },
        "scope_note": (
            "Listener belief/vote effects are immediate. Eventual game-outcome claims require "
            "a configured replay adapter and separate replay-quality validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-statements", type=int, default=30)
    parser.add_argument(
        "--labels-validated",
        action="store_true",
        help="Assert that included candidate statements completed human label review.",
    )
    args = parser.parse_args()
    report = analyze(
        list(read_jsonl(args.input)),
        min_statements=args.min_statements,
        labels_validated=args.labels_validated,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} ({report['statement_count']} statements)")


if __name__ == "__main__":
    main()
