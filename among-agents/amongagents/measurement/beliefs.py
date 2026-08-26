"""Belief elicitation, behavioral forks, activation adapters, and triangulation."""

from __future__ import annotations

import inspect
import json
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


BELIEF_SCHEMA_VERSION = "amongus.belief-measurement.v1"

ELICITATION_PROMPTS = {
    "probability": (
        "Estimate the independent probability that each other living player is an impostor."
    ),
    "suspicion": (
        "Express how strongly you currently suspect each other living player of being an impostor, "
        "using 0 as no suspicion and 1 as maximal suspicion."
    ),
    "trust_inverse": (
        "For each other living player, estimate your distrust that they are a crewmate. "
        "Report that distrust directly as an impostor probability from 0 to 1."
    ),
}


def _public_measurement_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only information available to the measured agent."""
    speaker = context["observation"].get("self_state", {}).get("name")
    return {
        "phase": context["phase"],
        "round": context["round"],
        "players": [
            player["name"]
            for player in context["world_state"].get("players", [])
            if player.get("alive") and player["name"] != speaker
        ],
        "your_observation": context["observation"],
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:].lstrip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end < start:
        raise ValueError("response did not contain a JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("belief response must be a JSON object")
    return value


def _aliases(context: Mapping[str, Any]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for player in context["world_state"].get("players", []):
        canonical = player["name"]
        aliases[canonical.casefold()] = canonical
        if ":" in canonical:
            number, color = canonical.split(":", 1)
            aliases[number.strip().casefold()] = canonical
            aliases[color.strip().casefold()] = canonical
    return aliases


def _probabilities(
    raw: Mapping[str, Any], context: Mapping[str, Any]
) -> Dict[str, float]:
    aliases = _aliases(context)
    speaker = context["observation"].get("self_state", {}).get("name")
    normalized: Dict[str, float] = {}
    for name, value in raw.items():
        canonical = aliases.get(str(name).strip().casefold())
        if canonical is None:
            continue
        if canonical == speaker:
            continue
        try:
            probability = float(value)
        except (TypeError, ValueError):
            continue
        normalized[canonical] = min(1.0, max(0.0, probability))
    return dict(sorted(normalized.items()))


class ElicitedBeliefProbe:
    method = "elicited"

    def __init__(
        self,
        phases: Optional[Iterable[str]] = ("meeting",),
        prompt_variant: str = "probability",
        temperature: float = 0.0,
        compact_response: bool = False,
    ):
        if prompt_variant not in ELICITATION_PROMPTS:
            raise ValueError(f"Unknown elicitation prompt variant: {prompt_variant}")
        self.phases = set(phases) if phases is not None else None
        self.prompt_variant = prompt_variant
        self.temperature = temperature
        self.compact_response = compact_response

    def should_probe(self, context: Mapping[str, Any], agent: Any) -> bool:
        phase_allowed = self.phases is None or context["phase"] in self.phases
        return phase_allowed and callable(getattr(agent, "send_request", None))

    def messages(self, context: Mapping[str, Any]) -> list[dict[str, str]]:
        """Build the private query separately so repeated samples can be batched."""
        public_context = _public_measurement_context(context)
        if self.compact_response:
            compact_keys = [
                name.split(":", 1)[1].strip() if ":" in name else name
                for name in public_context["players"]
            ]
            response_shape = (
                json.dumps(
                    {
                        "impostor_probabilities": {
                            name: 0.0 for name in compact_keys
                        }
                    },
                    separators=(",", ":"),
                )
                + ". Include every shown key exactly once and no other player. "
            )
        else:
            response_shape = (
                '{"impostor_probabilities":{"player name":0.0},'
                '"event_beliefs":[],"beliefs_about_others":{},"confidence":0.0}. '
            )
        prompt = (
            "Privately estimate your current beliefs for research measurement. "
            "This response is not shown to players and will not be added to your game memory. "
            + ELICITATION_PROMPTS[self.prompt_variant]
            + " "
            "Use only the observation supplied below. Return JSON only with this shape: "
            + response_shape
            + "Each probability is an independent probability from 0 to 1; they need not sum to 1.\n"
            + json.dumps(public_context, sort_keys=True)
        )
        return [
            {
                "role": "system",
                "content": "You are answering a private measurement query, not taking a game action.",
            },
            {"role": "user", "content": prompt},
        ]

    def parse_response(
        self, context: Mapping[str, Any], raw_response: str
    ) -> Dict[str, Any]:
        """Parse one response using the same rules as the sequential measurement."""
        try:
            parsed = _extract_json_object(raw_response)
            probabilities = _probabilities(
                parsed.get("impostor_probabilities", {}), context
            )
            if not probabilities:
                raise ValueError("no recognized player probabilities")
            confidence = parsed.get("confidence")
            try:
                confidence = min(1.0, max(0.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = None
            return {
                "method": self.method,
                "probability_scale": "self_reported_probability",
                "prompt_variant": self.prompt_variant,
                "temperature": self.temperature,
                "response_schema": (
                    "impostor_probabilities_only"
                    if self.compact_response
                    else "full_belief_measurement"
                ),
                "status": "collected",
                "impostor_probabilities": probabilities,
                "event_beliefs": parsed.get("event_beliefs", []),
                "beliefs_about_others": parsed.get("beliefs_about_others", {}),
                "confidence": confidence,
                "raw_response": raw_response,
            }
        except (ValueError, json.JSONDecodeError) as error:
            return {
                "method": self.method,
                "probability_scale": "self_reported_probability",
                "prompt_variant": self.prompt_variant,
                "temperature": self.temperature,
                "response_schema": (
                    "impostor_probabilities_only"
                    if self.compact_response
                    else "full_belief_measurement"
                ),
                "status": "invalid_response",
                "impostor_probabilities": {},
                "error": str(error),
                "raw_response": raw_response,
            }

    async def measure(
        self, context: Mapping[str, Any], agent: Any
    ) -> Dict[str, Any]:
        raw_response = await agent.send_request(
            self.messages(context), temperature=self.temperature
        )
        return self.parse_response(context, raw_response)


class BehavioralForkBeliefProbe:
    """Measure action-implied beliefs with a throwaway forced-choice fork."""

    method = "behavioral_fork"

    def __init__(self, phases: Optional[Iterable[str]] = ("meeting",)):
        self.phases = set(phases) if phases is not None else None

    def should_probe(self, context: Mapping[str, Any], agent: Any) -> bool:
        phase_allowed = self.phases is None or context["phase"] in self.phases
        return phase_allowed and callable(getattr(agent, "send_request", None))

    async def measure(
        self, context: Mapping[str, Any], agent: Any
    ) -> Dict[str, Any]:
        public_context = _public_measurement_context(context)
        allowed_choices = [
            name.split(":", 1)[1].strip() if ":" in name else name
            for name in public_context["players"]
        ]
        prompt = (
            "This is a private counterfactual decision fork. It does not change the real game and "
            "will not be added to memory. Based only on the supplied observation, return JSON only: "
            '{"vote_for":"player","avoid":"player","trust":"player","follow":"player"}. '
            "Choose one living player for every field; choices may repeat. "
            f"Every value must be exactly one of these allowed colors: {json.dumps(allowed_choices)}. "
            "Do not choose yourself and do not return none.\n"
            + json.dumps(public_context, sort_keys=True)
        )
        messages = [
            {
                "role": "system",
                "content": "Make private forced-choice decisions without producing a game action.",
            },
            {"role": "user", "content": prompt},
        ]
        raw_responses = []
        last_error = "invalid behavioral response"
        aliases = _aliases(context)
        required = ("vote_for", "avoid", "trust", "follow")
        for attempt in range(2):
            raw_response = await agent.send_request(messages, temperature=0.0)
            raw_responses.append(raw_response)
            try:
                parsed = _extract_json_object(raw_response)
                choices = {}
                for key in required:
                    value = parsed.get(key)
                    canonical = aliases.get(str(value).strip().casefold())
                    if canonical in public_context["players"]:
                        choices[key] = canonical
                if len(choices) != len(required):
                    raise ValueError(
                        "every behavioral choice must name an allowed other player"
                    )
                players = public_context["players"]
                scores = {player: 0.5 for player in players}
                scores[choices["vote_for"]] += 0.3
                scores[choices["avoid"]] += 0.2
                scores[choices["trust"]] -= 0.25
                scores[choices["follow"]] -= 0.1
                probabilities = {
                    player: min(1.0, max(0.0, score))
                    for player, score in scores.items()
                }
                return {
                    "method": self.method,
                    "probability_scale": "heuristic_rank_score",
                    "status": "collected",
                    "choices": choices,
                    "impostor_probabilities": probabilities,
                    "scoring_note": "heuristic action-implied suspicion; not a calibrated probability",
                    "raw_response": raw_response,
                    "raw_responses": raw_responses,
                    "format_retry_count": attempt,
                }
            except (ValueError, json.JSONDecodeError) as error:
                last_error = str(error)
                if attempt == 0:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw_response},
                        {
                            "role": "user",
                            "content": (
                                "That answer used a missing or disallowed value. Correct only the "
                                "JSON format. Keep the same private decision, but replace invalid "
                                f"values with one of {json.dumps(allowed_choices)}. Every one of "
                                f"{json.dumps(required)} is required. Return JSON only."
                            ),
                        },
                    ]
        return {
            "method": self.method,
            "probability_scale": "heuristic_rank_score",
            "status": "invalid_response",
            "impostor_probabilities": {},
            "error": last_error,
            "raw_response": raw_responses[-1],
            "raw_responses": raw_responses,
            "format_retry_count": 1,
        }


class ActivationBeliefProbe:
    """Adapter for an open-weight activation decoder supplied by a GPU worker."""

    method = "activation"

    def __init__(
        self,
        decoder: Callable[..., Mapping[str, float]],
        phases: Optional[Iterable[str]] = None,
    ):
        self.decoder = decoder
        self.phases = set(phases) if phases is not None else None

    def should_probe(self, context: Mapping[str, Any], agent: Any) -> bool:
        return self.phases is None or context["phase"] in self.phases

    async def measure(
        self, context: Mapping[str, Any], agent: Any
    ) -> Dict[str, Any]:
        try:
            result = self.decoder(
                context=_public_measurement_context(context), agent=agent
            )
            if inspect.isawaitable(result):
                result = await result
            probabilities = _probabilities(result, context)
            if not probabilities:
                raise ValueError("activation decoder returned no recognized probabilities")
            return {
                "method": self.method,
                "probability_scale": "decoder_probability",
                "status": "collected",
                "impostor_probabilities": probabilities,
            }
        except Exception as error:  # decoder boundaries must become data, not corrupt a game
            return {
                "method": self.method,
                "probability_scale": "decoder_probability",
                "status": "decoder_error",
                "impostor_probabilities": {},
                "error": f"{type(error).__name__}: {error}",
            }


def triangulate(measurements: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    usable = [
        measurement
        for measurement in measurements
        if measurement.get("status") == "collected"
        and measurement.get("impostor_probabilities")
    ]
    if not usable:
        return {
            "status": "unavailable",
            "available_methods": [],
            "consensus_probabilities": {},
        }

    methods = [measurement["method"] for measurement in usable]
    probability_scales = {
        measurement["method"]: measurement.get("probability_scale", "unspecified")
        for measurement in usable
    }
    probability_l1_comparable = not any(
        scale == "heuristic_rank_score" for scale in probability_scales.values()
    )
    players = sorted(
        {
            player
            for measurement in usable
            for player in measurement["impostor_probabilities"]
        }
    )
    consensus = {}
    for player in players:
        values = [
            measurement["impostor_probabilities"][player]
            for measurement in usable
            if player in measurement["impostor_probabilities"]
        ]
        consensus[player] = sum(values) / len(values)

    top_by_method = {
        measurement["method"]: max(
            measurement["impostor_probabilities"],
            key=measurement["impostor_probabilities"].get,
        )
        for measurement in usable
    }
    top_counts = Counter(top_by_method.values())
    consensus_top = max(consensus, key=consensus.get)
    pairwise_l1: List[float] = []
    for index, left in enumerate(usable):
        for right in usable[index + 1 :]:
            common = set(left["impostor_probabilities"]) & set(
                right["impostor_probabilities"]
            )
            if common:
                pairwise_l1.append(
                    sum(
                        abs(
                            left["impostor_probabilities"][player]
                            - right["impostor_probabilities"][player]
                        )
                        for player in common
                    )
                    / len(common)
                )
    return {
        "status": "triangulated" if len(usable) > 1 else "single_method",
        "available_methods": methods,
        "probability_scales": probability_scales,
        "probability_l1_comparable": probability_l1_comparable,
        "consensus_probabilities": consensus,
        "consensus_top_suspect": consensus_top,
        "top_suspect_by_method": top_by_method,
        "top_choice_agreement": max(top_counts.values()) / len(usable),
        "mean_pairwise_l1": (
            sum(pairwise_l1) / len(pairwise_l1) if pairwise_l1 else None
        ),
    }


class CompositeBeliefProbe:
    def __init__(self, probes: Sequence[Any]):
        self.probes = list(probes)

    def should_probe(self, context: Mapping[str, Any], agent: Any) -> bool:
        return any(probe.should_probe(context, agent) for probe in self.probes)

    async def measure(
        self, context: Mapping[str, Any], agent: Any
    ) -> Dict[str, Any]:
        results = []
        for probe in self.probes:
            if probe.should_probe(context, agent):
                results.append(await probe.measure(context, agent))
        triangle = triangulate(results)
        collected = [result for result in results if result.get("status") == "collected"]
        other_beliefs = next(
            (
                result.get("beliefs_about_others", {})
                for result in collected
                if result.get("beliefs_about_others")
            ),
            {},
        )
        return {
            "schema_version": BELIEF_SCHEMA_VERSION,
            "status": (
                "collected"
                if len(collected) == len(results) and collected
                else "partial"
                if collected
                else "unavailable"
            ),
            "measurements": {result["method"]: result for result in results},
            "values": triangle.get("consensus_probabilities", {}),
            "triangulation": triangle,
            "beliefs_about_others": other_beliefs,
        }


def build_belief_probe(
    mode: str,
    *,
    reliability_repeats: int = 20,
    reliability_temperature: float = 0.7,
) -> Optional[Any]:
    # Imported lazily to keep the base probes independently reusable.
    from .reliability import BeliefReliabilityProbe

    modes = {
        "none": None,
        "elicited-meeting": [ElicitedBeliefProbe(("meeting",))],
        "triangulated-meeting": [
            ElicitedBeliefProbe(("meeting",)),
            BehavioralForkBeliefProbe(("meeting",)),
        ],
        "triangulated-all": [
            ElicitedBeliefProbe(None),
            BehavioralForkBeliefProbe(None),
        ],
        "reliability-meeting": BeliefReliabilityProbe(
            phases=("meeting",),
            repeats=reliability_repeats,
            temperature=reliability_temperature,
        ),
    }
    if mode not in modes:
        raise ValueError(f"Unknown belief probe mode: {mode}")
    probes = modes[mode]
    if probes is None:
        return None
    if isinstance(probes, BeliefReliabilityProbe):
        return probes
    return CompositeBeliefProbe(probes)
