import json
from pathlib import Path

import numpy as np

from amongagents.control import (
    activation_monitor_probabilities,
    evaluate_control_policies,
    parse_secondary_judge,
)
from amongagents.interpretability import build_generalization_suite


ROOT = Path(__file__).parents[1]


def test_secondary_judge_parser_retains_invalid_outputs():
    parsed = parse_secondary_judge(
        '{"strategic_mismatch": true, "confidence": 0.9, "reason": "conflict"}'
    )
    assert parsed["status"] == "parsed"
    assert parsed["strategic_mismatch"] is True
    assert parse_secondary_judge("not json")["status"] == "invalid"


def test_phase12_controls_separate_strategic_mismatch_from_sincere_mistake():
    records = build_generalization_suite(train_groups=5, test_groups=2)
    labels = np.asarray([record["strategic_deception"] for record in records])
    activations = np.zeros((len(records), 3, 8), dtype=np.float32)
    activations[:, 1, 0] = np.where(labels == 1, 5.0, -5.0)
    probabilities = activation_monitor_probabilities(
        activations, records, layer=1
    )
    judgments = {
        record["sample_id"]: {
            "status": "parsed",
            "strategic_mismatch": bool(record["strategic_deception"]),
        }
        for record in records
        if record["suite"] != "train"
    }

    activation_cost = {
        "input_tokens": 1200,
        "output_tokens": 0,
        "latency_seconds": 2.5,
        "mean_latency_seconds": 0.1,
    }
    report, decisions = evaluate_control_policies(
        records,
        probabilities,
        judgments,
        judge_cost={
            "input_tokens": 800,
            "output_tokens": 80,
            "latency_seconds": 4.0,
        },
        activation_cost=activation_cost,
    )
    schema = json.loads(
        (ROOT / "docs" / "schemas" / "control-intervention-report-v1.schema.json").read_text()
    )

    assert set(schema["required"]) <= set(report)
    assert report["monitors"]["world_state_grounding"][
        "strategic_deception_prevention"
    ]["rate"] == 1.0
    assert report["monitors"]["world_state_grounding"][
        "mistaken_falsehood_intervention_rate"
    ]["rate"] == 1.0
    assert report["monitors"]["belief_consistency"][
        "nondeceptive_false_positive_rate"
    ]["rate"] == 0.0
    assert report["monitors"]["belief_consistency"]["pareto_eligible"] is True
    assert report["monitors"]["no_control"]["pareto_eligible"] is False
    assert report["monitors"]["activation_monitor"]["cost"]["latency_seconds"] == 2.5
    assert report["monitors"]["combined_consensus"]["cost"] == {
        "input_tokens": 2000,
        "output_tokens": 80,
        "latency_seconds": 6.5,
        "scope": "secondary audit calls plus online activation-monitor forward passes",
    }
    assert "activation_monitor" not in report["pareto_cost_incomplete_monitors"]
    assert len(decisions) == sum(record["suite"] != "train" for record in records)
