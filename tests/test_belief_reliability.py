import asyncio
import json
from pathlib import Path

from amongagents.measurement.reliability import (
    BeliefReliabilityProbe,
    assess_method_agreement,
)


def make_context():
    return {
        "phase": "meeting",
        "round": 1,
        "world_state": {
            "players": [
                {"name": "Player 1: red", "alive": True, "role": "Crewmate"},
                {"name": "Player 2: blue", "alive": True, "role": "Impostor"},
                {"name": "Player 3: green", "alive": True, "role": "Crewmate"},
            ]
        },
        "observation": {
            "self_state": {"name": "Player 1: red", "location": "Cafeteria", "alive": True},
            "visible_players": ["Player 2: blue", "Player 3: green"],
            "private_role_knowledge": {"own_role": "Crewmate", "known_impostors": []},
            "location_information": "Current Location: Cafeteria",
            "observation_history": [],
            "action_history": [],
            "available_actions": [],
        },
    }


class FakeAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.summarization = "unchanged summary"
        self.processed_memory = "unchanged memory"

    async def send_request(self, messages, temperature=None):
        self.requests.append({"messages": messages, "temperature": temperature})
        return self.responses.pop(0)


class BatchAgent(FakeAgent):
    def __init__(self, responses):
        super().__init__(responses)
        self.batch_requests = []

    async def send_requests(self, message_batches, temperature=None):
        self.batch_requests.append(
            {"message_batches": message_batches, "temperature": temperature}
        )
        responses = list(self.responses[: len(message_batches)])
        del self.responses[: len(message_batches)]
        return responses


def response(blue, green):
    return json.dumps(
        {
            "impostor_probabilities": {"blue": blue, "green": green},
            "event_beliefs": [],
            "beliefs_about_others": {},
            "confidence": 0.8,
        }
    )


def test_reliability_probe_freezes_context_and_passes_stable_samples():
    agent = FakeAgent(
        [response(0.8, 0.2)] * 6
        + ['{"vote_for":"blue","avoid":"blue","trust":"green","follow":"green"}']
    )
    probe = BeliefReliabilityProbe(repeats=2, temperature=0.6)
    result = asyncio.run(probe.measure(make_context(), agent))
    suite = result["measurements"]["reliability"]
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "schemas"
            / "belief-reliability-v1.schema.json"
        ).read_text()
    )

    assert set(suite["analysis"]) == set(schema["required"])
    assert suite["analysis"]["gate"]["branch"] == "A_stable_self_report"
    assert result["triangulation"]["available_methods"] == [
        "elicited_repeated_consensus",
        "behavioral_fork",
    ]
    assert len({sample["context_fingerprint"] for sample in suite["samples"]}) == 1
    assert all(request["temperature"] == 0.6 for request in agent.requests[:6])
    assert agent.requests[-1]["temperature"] == 0.0
    prompts = "\n".join(
        message["content"]
        for request in agent.requests
        for message in request["messages"]
    )
    assert '"event_beliefs"' not in prompts
    assert '"beliefs_about_others"' not in prompts
    assert '"blue":0.0' in prompts
    assert '"green":0.0' in prompts
    assert '"role": "Impostor"' not in prompts
    assert agent.summarization == "unchanged summary"
    assert agent.processed_memory == "unchanged memory"


def test_reliability_probe_routes_unstable_estimates_to_branch_b():
    agent = FakeAgent([response(0.9, 0.1), response(0.1, 0.9)])
    probe = BeliefReliabilityProbe(
        repeats=2,
        prompt_variants=("probability",),
        include_behavioral_fork=False,
    )
    result = asyncio.run(probe.measure(make_context(), agent))
    gate = result["measurements"]["reliability"]["analysis"]["gate"]

    assert gate["branch"] == "B_unstable_self_report"
    assert gate["passed"] is False
    assert gate["reasons"]


def test_reliability_probe_uses_optional_batch_interface():
    agent = BatchAgent(
        [response(0.8, 0.2)] * 6
        + ['{"vote_for":"blue","avoid":"blue","trust":"green","follow":"green"}']
    )
    probe = BeliefReliabilityProbe(repeats=2, temperature=0.6)

    result = asyncio.run(probe.measure(make_context(), agent))

    assert result["measurements"]["reliability"]["analysis"]["valid_count"] == 6
    assert len(agent.batch_requests) == 1
    assert len(agent.batch_requests[0]["message_batches"]) == 6
    assert agent.batch_requests[0]["temperature"] == 0.6
    assert len(agent.requests) == 1  # the behavioral fork remains a separate method


def test_phase4_identifies_cross_method_disagreement():
    result = assess_method_agreement(
        {
            "available_methods": ["elicited", "behavioral_fork", "activation"],
            "mean_pairwise_l1": 0.5,
            "top_choice_agreement": 1 / 3,
        }
    )
    assert result["branch"] == "C_methods_disagree"
    assert result["passed"] is False


def test_phase4_does_not_gate_on_l1_for_heuristic_rank_scores():
    result = assess_method_agreement(
        {
            "available_methods": ["elicited", "behavioral_fork"],
            "mean_pairwise_l1": 0.5,
            "top_choice_agreement": 1.0,
            "probability_l1_comparable": False,
        }
    )

    assert result["branch"] == "methods_agree"
    assert result["probability_l1_used"] is False
