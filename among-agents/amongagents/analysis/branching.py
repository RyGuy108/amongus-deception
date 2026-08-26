"""Phase 11 evidence routing for the adaptive research program."""

from __future__ import annotations

from typing import Any, Mapping


BRANCHES = {
    "A": "transferable intent representation",
    "B": "environmental shortcut and construct audit",
    "C": "deception versus delusion",
    "D": "pragmatic deception",
}


def _score(report: Mapping[str, Any], suite: str) -> float | None:
    value = report.get("suites", {}).get(suite, {}).get("balanced_accuracy")
    return float(value) if value is not None else None


def _behavior_rate(report: Mapping[str, Any], prefix: str) -> float | None:
    rates = []
    for suite in report.get("suites", {}).values():
        if suite.get("status") != "evaluated":
            continue
        for name, diagnostics in suite.get("behavior_diagnostics", {}).items():
            if name.startswith(prefix):
                rates.append(float(diagnostics["predicted_positive_rate"]))
    return sum(rates) / len(rates) if rates else None


def _auroc(report: Mapping[str, Any], suite: str) -> float | None:
    value = report.get("suites", {}).get(suite, {}).get("auroc")
    return float(value) if value is not None else None


def select_research_branch(
    phase8: Mapping[str, Any],
    phase9: Mapping[str, Any],
    phase10: Mapping[str, Any],
    phase4: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank the four preregistered Phase 11 branches from collected evidence."""
    response = phase10.get("pooling_sites", {}).get("response_mean", phase10)
    prompt = phase10.get("pooling_sites", {}).get("prompt_final", {})
    iid = _score(response, "iid")
    game = _score(response, "game_ood")
    domain = _score(response, "domain_ood")
    category = _score(response, "category_ood")
    role = _score(response, "role_ood")
    game_auroc = _auroc(response, "game_ood")
    domain_auroc = _auroc(response, "domain_ood")
    category_auroc = _auroc(response, "category_ood")
    model_status = response.get("suites", {}).get("model_ood", {}).get("status")
    natural_validation = int(phase8.get("validation", {}).get("human_validated_count", 0))
    primary_counts = phase8.get("primary_distribution", {})
    multilabel = phase8.get("multilabel_distribution", {})
    pragmatic_count = sum(
        int(multilabel.get(name, 0))
        for name in ("omission", "equivocation", "misdirection", "pragmatic_distortion")
    )
    explicit_count = int(multilabel.get("strategic_falsehood", 0))
    mistaken_rate = _behavior_rate(response, "mistaken_false")
    p9_prompt = phase9.get("pooling_sites", {}).get("prompt_final", {})
    similarity = p9_prompt.get("representation_similarity", {}).get(
        "truthfulness__deception", {}
    ).get("cosine_similarity")
    similarity = abs(float(similarity)) if similarity is not None else None

    evaluated_universal = all(
        response.get("suites", {}).get(name, {}).get("status") == "evaluated"
        for name in ("incentive_ood", "game_ood", "model_ood", "domain_ood")
    )
    universal_floor = min(
        [value for value in (game, domain) if value is not None], default=0.0
    )
    a_ready = bool(
        evaluated_universal
        and universal_floor >= 0.75
        and natural_validation >= 30
    )
    a_score = 1.0 if a_ready else 0.0

    worst_transfer = min(
        [value for value in (game, domain, category) if value is not None], default=0.0
    )
    transfer_drop = max(0.0, (iid or 0.0) - worst_transfer)
    b_score = 0.0
    b_evidence = []
    if iid is not None and iid >= 0.75:
        b_score += 0.2
        b_evidence.append(f"source-matched IID calibration is strong ({iid:.3f})")
    if transfer_drop >= 0.15:
        b_score += 0.45
        b_evidence.append(
            f"worst transfer condition drops {transfer_drop:.3f} balanced-accuracy points"
        )
    if similarity is not None and similarity >= 0.8:
        b_score += 0.25
        b_evidence.append(
            f"Phase 9 truthfulness/deception directions are nearly collinear (|cos|={similarity:.3f})"
        )
    if model_status != "evaluated":
        b_score += 0.1
        b_evidence.append("model OOD remains unmeasured")
    calibration_shifts = [
        name
        for name, details in response.get("suites", {}).items()
        if details.get("frozen_threshold_calibration_shift")
    ]
    if calibration_shifts:
        b_evidence.append(
            "ranking survives better than the frozen threshold in "
            + ", ".join(sorted(calibration_shifts))
        )

    c_score = 0.0
    c_evidence = []
    if mistaken_rate is not None and mistaken_rate >= 0.4:
        c_score += 0.6
        c_evidence.append(
            f"mistaken-falsehood predicted-positive rate is {mistaken_rate:.3f}"
        )
    if similarity is not None and similarity >= 0.8:
        c_score += 0.3
        c_evidence.append("natural-trajectory probe is strongly entangled with truthfulness")
    mistaken_count = int(multilabel.get("mistaken_belief", 0))
    if mistaken_count:
        c_score += 0.1
        c_evidence.append(f"Phase 8 contains {mistaken_count} mistaken-belief candidates")
    phase4_recommendation = (
        phase4.get("decision_gate", {}).get("recommendation") if phase4 else None
    )
    phase4_stability_rate = phase4.get("stability_rate") if phase4 else None
    phase4_method_disagreement = (
        phase4.get("method_disagreement_rate") if phase4 else None
    )
    phase4_requires_c = phase4_recommendation == (
        "C_investigate_belief_report_faithfulness"
    )
    if phase4_requires_c:
        c_score = max(c_score, 1.0)
        c_evidence.append(
            "Phase 4's preregistered gate requires belief-report faithfulness investigation"
        )
        if phase4_stability_rate is not None:
            c_evidence.append(
                f"only {float(phase4_stability_rate):.3f} of frozen contexts passed stability"
            )
        if phase4_method_disagreement is not None:
            c_evidence.append(
                f"scale-aware cross-method disagreement rate is {float(phase4_method_disagreement):.3f}"
            )

    d_score = 0.0
    d_evidence = []
    if pragmatic_count > explicit_count:
        d_score += 0.4
        d_evidence.append(
            f"automatic pragmatic candidates outnumber explicit strategic candidates ({pragmatic_count} vs {explicit_count})"
        )
    if iid is not None and category is not None and iid - category >= 0.15:
        d_score += 0.45
        d_evidence.append(
            f"category OOD drops {iid - category:.3f} balanced-accuracy points from IID"
        )
    if int(primary_counts.get("omission", 0)):
        d_score += 0.1
        d_evidence.append(
            f"Phase 8 assigns {int(primary_counts.get('omission', 0))} primary omission candidates"
        )

    candidates = {
        "A": {
            "score": a_score,
            "status": "supported" if a_ready else "not_supported",
            "evidence": (
                ["all critical transfer levels pass and natural labels meet the validation floor"]
                if a_ready
                else [
                    "universal branch requires model OOD, >=0.75 critical-transfer floor, and at least 30 human-validated natural labels"
                ]
            ),
        },
        "B": {"score": b_score, "status": "candidate", "evidence": b_evidence},
        "C": {"score": c_score, "status": "candidate", "evidence": c_evidence},
        "D": {"score": d_score, "status": "candidate", "evidence": d_evidence},
    }
    selectable = ["A"] if a_ready else ["B", "C", "D"]
    primary = (
        "C"
        if phase4_requires_c
        else max(
            selectable,
            key=lambda key: (candidates[key]["score"], -selectable.index(key)),
        )
    )
    candidates[primary]["status"] = "selected"
    ranking = sorted(
        candidates,
        key=lambda key: (key == primary, candidates[key]["score"]),
        reverse=True,
    )
    plans = {
        "A": [
            "replicate across additional open-weight model families",
            "run cross-game causal activation interventions",
            "test whether steering the direction changes deceptive choices",
        ],
        "B": [
            "balance human-validated intent labels within role and incentive strata",
            "remove or counterbalance prompt, role, vocabulary, truth-value, and game-state cues",
            "separate rank transfer from threshold calibration under domain shift",
            "repeat frozen tests on a second model before interpreting the direction as transferable",
        ],
        "C": [
            "collect objective truth, stated belief, and confidence before every public claim",
            "match intentional falsehoods to equally false sincere mistakes",
            "test whether interventions separate belief accuracy from communicative intent",
        ],
        "D": [
            "human-validate omission, equivocation, implicature, and misdirection candidates",
            "add counterfactual full-disclosure controls for the same world state",
            "measure listener belief change from pragmatic acts rather than sentence truth alone",
        ],
    }
    return {
        "schema_version": "amongus.phase11-branch-decision.v1",
        "primary_branch": primary,
        "primary_branch_name": BRANCHES[primary],
        "ranking": ranking,
        "branches": candidates,
        "adaptive_plan": plans[primary],
        "decision_inputs": {
            "phase8_human_validated_count": natural_validation,
            "phase8_pragmatic_candidate_count": pragmatic_count,
            "phase8_explicit_strategic_candidate_count": explicit_count,
            "phase9_truthfulness_deception_absolute_cosine": similarity,
            "phase10_response_iid_balanced_accuracy": iid,
            "phase10_response_game_ood_balanced_accuracy": game,
            "phase10_response_domain_ood_balanced_accuracy": domain,
            "phase10_response_category_ood_balanced_accuracy": category,
            "phase10_response_category_ood_auroc": category_auroc,
            "phase10_response_role_ood_balanced_accuracy": role,
            "phase10_response_game_ood_auroc": game_auroc,
            "phase10_response_domain_ood_auroc": domain_auroc,
            "phase10_response_mistaken_falsehood_positive_rate": mistaken_rate,
            "phase10_prompt_iid_balanced_accuracy": _score(prompt, "iid") if prompt else None,
            "phase10_model_ood_status": model_status,
            "phase4_recommendation": phase4_recommendation,
            "phase4_stability_rate": phase4_stability_rate,
            "phase4_method_disagreement_rate": phase4_method_disagreement,
        },
        "branch_execution": {
            "completed_audits": [
                "truth-value-matched strategic, mistaken, and truthful controls",
                "statement-only and full-text lexical baselines",
                "paired semantic perturbation stability",
                "frozen-threshold versus threshold-free ranking comparison",
            ],
            "calibration_shift_suites": calibration_shifts,
            "statement_only_iid_balanced_accuracy": response.get(
                "text_shortcut_baselines", {}
            ).get("statement_only", {}).get("suites", {}).get("iid", {}).get(
                "balanced_accuracy"
            ),
            "natural_label_validation_still_required": natural_validation < 30,
            "phase4_belief_faithfulness_audit_required": phase4_requires_c,
        },
        "gate": {
            "phase11_decision_completed": True,
            "universal_representation_claim_authorized": a_ready,
            "natural_deception_inference_authorized": natural_validation >= 30,
        },
        "interpretation_boundary": (
            "Branch selection is a preregistered research decision, not confirmation of a deception mechanism. "
            "Automatic Phase 8 labels remain hypotheses until independent human review."
        ),
    }
