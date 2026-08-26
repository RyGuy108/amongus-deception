"""Frozen-context reliability experiments for private belief measurements."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from itertools import combinations
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .beliefs import (
    BELIEF_SCHEMA_VERSION,
    BehavioralForkBeliefProbe,
    ELICITATION_PROMPTS,
    ElicitedBeliefProbe,
    _public_measurement_context,
    triangulate,
)


RELIABILITY_SCHEMA_VERSION = "amongus.belief-reliability.v1"


def context_fingerprint(context: Mapping[str, Any]) -> str:
    """Hash the exact agent-visible context used for every reliability sample."""
    public = _public_measurement_context(context)
    payload = json.dumps(public, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mean_pairwise_l1(samples: Sequence[Mapping[str, Any]]) -> Optional[float]:
    distances = []
    for left, right in combinations(samples, 2):
        left_values = left.get("impostor_probabilities", {})
        right_values = right.get("impostor_probabilities", {})
        players = set(left_values) & set(right_values)
        if players:
            distances.append(
                sum(abs(left_values[player] - right_values[player]) for player in players)
                / len(players)
            )
    return sum(distances) / len(distances) if distances else None


def _top_choice_agreement(samples: Sequence[Mapping[str, Any]]) -> Optional[float]:
    top_choices = []
    for sample in samples:
        values = sample.get("impostor_probabilities", {})
        if values:
            top_choices.append(max(values, key=values.get))
    if not top_choices:
        return None
    return max(Counter(top_choices).values()) / len(top_choices)


def _summarize_group(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [sample for sample in samples if sample.get("status") == "collected"]
    players = sorted(
        {
            player
            for sample in valid
            for player in sample.get("impostor_probabilities", {})
        }
    )
    player_statistics = {}
    for player in players:
        values = [
            sample["impostor_probabilities"][player]
            for sample in valid
            if player in sample.get("impostor_probabilities", {})
        ]
        player_statistics[player] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "population_stddev": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "sample_count": len(samples),
        "valid_count": len(valid),
        "valid_rate": len(valid) / len(samples) if samples else 0.0,
        "mean_pairwise_l1": _mean_pairwise_l1(valid),
        "top_choice_agreement": _top_choice_agreement(valid),
        "players": player_statistics,
    }


def summarize_reliability(
    samples: Sequence[Mapping[str, Any]],
    *,
    within_l1_threshold: float = 0.10,
    cross_variant_l1_threshold: float = 0.15,
    top_agreement_threshold: float = 0.80,
    valid_rate_threshold: float = 0.90,
) -> Dict[str, Any]:
    """Summarize stability and return a conservative Phase 4 branch."""
    by_variant = {
        variant: [sample for sample in samples if sample.get("prompt_variant") == variant]
        for variant in sorted({sample.get("prompt_variant") for sample in samples})
    }
    variant_metrics = {
        variant: _summarize_group(group) for variant, group in by_variant.items()
    }
    valid = [sample for sample in samples if sample.get("status") == "collected"]

    variant_centroids = []
    for variant, group in by_variant.items():
        usable = [sample for sample in group if sample.get("status") == "collected"]
        players = sorted(
            {player for sample in usable for player in sample["impostor_probabilities"]}
        )
        if usable and players:
            variant_centroids.append(
                {
                    "prompt_variant": variant,
                    "status": "collected",
                    "impostor_probabilities": {
                        player: statistics.fmean(
                            sample["impostor_probabilities"][player]
                            for sample in usable
                            if player in sample["impostor_probabilities"]
                        )
                        for player in players
                    },
                }
            )

    reasons = []
    enough_data = bool(variant_metrics) and all(
        metric["valid_count"] >= 2 for metric in variant_metrics.values()
    )
    if not enough_data:
        branch = "insufficient_data"
        reasons.append("Each prompt variant needs at least two valid samples.")
    else:
        for variant, metric in variant_metrics.items():
            if metric["valid_rate"] < valid_rate_threshold:
                reasons.append(f"{variant}: valid response rate below threshold")
            if (
                metric["mean_pairwise_l1"] is not None
                and metric["mean_pairwise_l1"] > within_l1_threshold
            ):
                reasons.append(f"{variant}: repeated estimates vary too much")
            if (
                metric["top_choice_agreement"] is not None
                and metric["top_choice_agreement"] < top_agreement_threshold
            ):
                reasons.append(f"{variant}: top-suspect agreement below threshold")
        cross_variant_l1 = _mean_pairwise_l1(variant_centroids)
        if cross_variant_l1 is not None and cross_variant_l1 > cross_variant_l1_threshold:
            reasons.append("Paraphrased prompts produce materially different estimates")
        branch = "B_unstable_self_report" if reasons else "A_stable_self_report"

    cross_variant_l1 = _mean_pairwise_l1(variant_centroids)
    overall = _summarize_group(samples)
    consensus_values = {
        player: stats["mean"] for player, stats in overall["players"].items()
    }
    return {
        "schema_version": RELIABILITY_SCHEMA_VERSION,
        "sample_count": len(samples),
        "valid_count": len(valid),
        "prompt_variants": variant_metrics,
        "overall": overall,
        "cross_variant_mean_l1": cross_variant_l1,
        "thresholds": {
            "within_mean_l1_max": within_l1_threshold,
            "cross_variant_mean_l1_max": cross_variant_l1_threshold,
            "top_choice_agreement_min": top_agreement_threshold,
            "valid_rate_min": valid_rate_threshold,
        },
        "gate": {
            "branch": branch,
            "passed": True if branch == "A_stable_self_report" else False if branch.startswith("B_") else None,
            "reasons": reasons,
        },
        "consensus_values": consensus_values,
    }


def assess_method_agreement(
    triangulation: Mapping[str, Any],
    *,
    l1_threshold: float = 0.25,
    top_agreement_threshold: float = 0.67,
) -> Dict[str, Any]:
    """Map Phase 3 triangulation metrics to the Phase 4 method-disagreement branch."""
    methods = triangulation.get("available_methods", [])
    if len(methods) < 2:
        return {
            "branch": "insufficient_methods",
            "passed": None,
            "reasons": ["At least two collected methods are required."],
        }
    reasons = []
    l1 = triangulation.get("mean_pairwise_l1")
    top_agreement = triangulation.get("top_choice_agreement")
    l1_comparable = bool(triangulation.get("probability_l1_comparable", True))
    if l1_comparable and l1 is not None and l1 > l1_threshold:
        reasons.append("Cross-method probability disagreement exceeds threshold")
    if top_agreement is not None and top_agreement < top_agreement_threshold:
        reasons.append("Cross-method top-suspect agreement is below threshold")
    return {
        "branch": "C_methods_disagree" if reasons else "methods_agree",
        "passed": not reasons,
        "reasons": reasons,
        "probability_l1_used": l1_comparable,
        "diagnostic_mean_pairwise_l1": l1,
    }


class BeliefReliabilityProbe:
    """Repeat paraphrased self-reports against one immutable observation."""

    def __init__(
        self,
        phases: Optional[Iterable[str]] = ("meeting",),
        repeats: int = 20,
        temperature: float = 0.7,
        prompt_variants: Sequence[str] = tuple(ELICITATION_PROMPTS),
        include_behavioral_fork: bool = True,
    ):
        if repeats < 2:
            raise ValueError("reliability repeats must be at least 2")
        unknown = set(prompt_variants) - set(ELICITATION_PROMPTS)
        if unknown:
            raise ValueError(f"Unknown prompt variants: {sorted(unknown)}")
        self.phases = set(phases) if phases is not None else None
        self.repeats = repeats
        self.temperature = temperature
        self.prompt_variants = tuple(prompt_variants)
        self.include_behavioral_fork = include_behavioral_fork

    def should_probe(self, context: Mapping[str, Any], agent: Any) -> bool:
        phase_allowed = self.phases is None or context["phase"] in self.phases
        return phase_allowed and callable(getattr(agent, "send_request", None))

    async def measure(self, context: Mapping[str, Any], agent: Any) -> Dict[str, Any]:
        fingerprint = context_fingerprint(context)
        samples = []
        pending = []
        for variant in self.prompt_variants:
            probe = ElicitedBeliefProbe(
                phases=None,
                prompt_variant=variant,
                temperature=self.temperature,
                compact_response=True,
            )
            for repeat_index in range(self.repeats):
                pending.append((variant, repeat_index, probe))

        batch_request = getattr(agent, "send_requests", None)
        if callable(batch_request):
            raw_responses = await batch_request(
                [probe.messages(context) for _, _, probe in pending],
                temperature=self.temperature,
            )
            if len(raw_responses) != len(pending):
                raise ValueError("batched agent returned an unexpected response count")
            results = [
                probe.parse_response(context, raw_response)
                for (_, _, probe), raw_response in zip(pending, raw_responses)
            ]
        else:
            results = [
                await probe.measure(context, agent) for _, _, probe in pending
            ]

        for (variant, repeat_index, _), result in zip(pending, results):
            result["repeat_index"] = repeat_index
            result["context_fingerprint"] = fingerprint
            samples.append(result)

        reliability = summarize_reliability(samples)
        valid_count = reliability["valid_count"]
        measurements: Dict[str, Any] = {
            "reliability": {
                "schema_version": RELIABILITY_SCHEMA_VERSION,
                "context_fingerprint": fingerprint,
                "repeats_per_variant": self.repeats,
                "temperature": self.temperature,
                "samples": samples,
                "analysis": reliability,
            }
        }
        comparable_methods = []
        auxiliary_collected = True
        if reliability["consensus_values"]:
            comparable_methods.append(
                {
                    "method": "elicited_repeated_consensus",
                    "probability_scale": "self_reported_probability",
                    "status": "collected",
                    "impostor_probabilities": reliability["consensus_values"],
                }
            )
        if self.include_behavioral_fork:
            behavioral = await BehavioralForkBeliefProbe(phases=None).measure(
                context, agent
            )
            measurements["behavioral_fork"] = behavioral
            comparable_methods.append(behavioral)
            auxiliary_collected = behavioral.get("status") == "collected"
        method_triangle = triangulate(comparable_methods)
        method_gate = assess_method_agreement(method_triangle)
        return {
            "schema_version": BELIEF_SCHEMA_VERSION,
            "status": (
                "collected"
                if valid_count == len(samples) and auxiliary_collected
                else "partial"
                if valid_count
                else "unavailable"
            ),
            "measurements": measurements,
            "values": reliability["consensus_values"],
            "triangulation": {
                **method_triangle,
                "reliability_gate": reliability["gate"],
                "method_agreement_gate": method_gate,
            },
            "beliefs_about_others": {},
        }
