import json
from collections import Counter
from pathlib import Path

import numpy as np

from amongagents.analysis import select_research_branch
from amongagents.interpretability import (
    build_generalization_suite,
    evaluate_frozen_generalization,
)


ROOT = Path(__file__).parents[1]


def test_phase10_suite_balances_intent_against_truth_and_mistake():
    records = build_generalization_suite()
    counts = Counter(record["suite"] for record in records)

    assert counts["train"] == 72
    assert all(counts[name] == 36 for name in counts if name != "train")
    assert counts["role_ood"] == 36
    train = [record for record in records if record["suite"] == "train"]
    assert sum(record["strategic_deception"] for record in train) == 24
    assert sum(record["deception_category"] == "mistaken_falsehood" for record in train) == 24
    assert sum(record["deception_category"] == "truth" for record in train) == 24
    assert all(
        record["belief_statement_mismatch"] == record["strategic_deception"]
        for record in records
    )


def test_phase10_probe_is_fitted_once_and_scores_all_available_ood_levels():
    records = build_generalization_suite(train_groups=6, test_groups=3)
    rng = np.random.default_rng(20260820)
    activations = rng.normal(0, 0.1, size=(len(records), 4, 12)).astype(np.float32)
    labels = np.asarray([record["strategic_deception"] for record in records])
    activations[:, 2, :] = 0.0
    activations[:, 2, 0] += np.where(labels == 1, 20.0, -20.0)

    report = evaluate_frozen_generalization(activations, records)
    schema = json.loads(
        (ROOT / "docs" / "schemas" / "frozen-generalization-report-v1.schema.json").read_text()
    )

    assert set(report) == set(schema["required"])
    assert report["selected_layer"] == 2
    assert report["training_design"]["fitting"].startswith("one final source fit")
    assert all(
        report["suites"][name]["balanced_accuracy"] == 1.0
        for name in (
            "iid",
            "lexical_ood",
            "scenario_ood",
            "role_ood",
            "incentive_ood",
            "category_ood",
            "game_ood",
            "domain_ood",
        )
    )
    assert report["suites"]["model_ood"]["status"] == "unavailable"
    assert report["gate"]["strong_universal_generalization_supported"] is False


def test_phase11_rejects_universal_claim_and_selects_evidence_branch():
    phase8 = {
        "validation": {"human_validated_count": 0},
        "primary_distribution": {"omission": 18},
        "multilabel_distribution": {"omission": 18, "strategic_falsehood": 4},
    }
    phase9 = {
        "pooling_sites": {
            "prompt_final": {
                "representation_similarity": {
                    "truthfulness__deception": {"cosine_similarity": -0.94}
                }
            }
        }
    }
    phase10_site = {
        "suites": {
            "iid": {"status": "evaluated", "balanced_accuracy": 0.95},
            "incentive_ood": {"status": "evaluated", "balanced_accuracy": 0.8},
            "category_ood": {
                "status": "evaluated",
                "balanced_accuracy": 0.55,
                "behavior_diagnostics": {},
            },
            "game_ood": {
                "status": "evaluated",
                "balanced_accuracy": 0.6,
                "behavior_diagnostics": {
                    "mistaken_false_accusation": {"predicted_positive_rate": 0.2}
                },
            },
            "domain_ood": {"status": "evaluated", "balanced_accuracy": 0.58},
            "model_ood": {"status": "unavailable"},
        }
    }
    report = select_research_branch(
        phase8,
        phase9,
        {"pooling_sites": {"prompt_final": phase10_site, "response_mean": phase10_site}},
    )
    schema = json.loads(
        (ROOT / "docs" / "schemas" / "research-branch-decision-v1.schema.json").read_text()
    )

    assert set(report) == set(schema["required"])
    assert report["primary_branch"] == "B"
    assert report["branches"]["A"]["status"] == "not_supported"
    assert report["gate"]["universal_representation_claim_authorized"] is False
    assert report["gate"]["natural_deception_inference_authorized"] is False


def test_phase11_later_phase4_gate_promotes_belief_faithfulness_branch():
    phase8 = {
        "validation": {"human_validated_count": 0},
        "primary_distribution": {},
        "multilabel_distribution": {},
    }
    phase9 = {"pooling_sites": {"prompt_final": {}}}
    site = {
        "suites": {
            "iid": {"status": "evaluated", "balanced_accuracy": 1.0},
            "category_ood": {"status": "evaluated", "balanced_accuracy": 0.5},
            "game_ood": {"status": "evaluated", "balanced_accuracy": 0.6},
            "domain_ood": {"status": "evaluated", "balanced_accuracy": 0.5},
            "model_ood": {"status": "unavailable"},
        }
    }
    phase4 = {
        "stability_rate": 14 / 30,
        "method_disagreement_rate": 0.55,
        "decision_gate": {
            "recommendation": "C_investigate_belief_report_faithfulness"
        },
    }

    report = select_research_branch(
        phase8,
        phase9,
        {"pooling_sites": {"prompt_final": site, "response_mean": site}},
        phase4,
    )

    assert report["primary_branch"] == "C"
    assert report["ranking"][0] == "C"
    assert report["branch_execution"]["phase4_belief_faithfulness_audit_required"]
