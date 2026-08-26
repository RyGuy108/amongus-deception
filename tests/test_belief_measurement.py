import asyncio
import json
from pathlib import Path

from amongagents.measurement.beliefs import (
    ActivationBeliefProbe,
    BehavioralForkBeliefProbe,
    CompositeBeliefProbe,
    ElicitedBeliefProbe,
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
            "observation_history": ["Player 2: blue moved away"],
            "action_history": [],
            "available_actions": [],
        },
    }


class FakeAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.summarization = "existing summary"
        self.processed_memory = "existing memory"

    async def send_request(self, messages, temperature=None):
        self.requests.append({"messages": messages, "temperature": temperature})
        return self.responses.pop(0)


def test_composite_probe_triangulates_without_leaking_hidden_world_roles():
    agent = FakeAgent(
        [
            '{"impostor_probabilities":{"blue":0.9,"green":0.1},"event_beliefs":[],"beliefs_about_others":{"green":{"blue":0.7}},"confidence":0.8}',
            '{"vote_for":"blue","avoid":"blue","trust":"green","follow":"green"}',
        ]
    )
    probe = CompositeBeliefProbe(
        [ElicitedBeliefProbe(), BehavioralForkBeliefProbe()]
    )
    result = asyncio.run(probe.measure(make_context(), agent))
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "schemas"
            / "belief-measurement-v1.schema.json"
        ).read_text()
    )

    assert set(result) == set(schema["required"])
    assert result["status"] == "collected"
    assert result["triangulation"]["status"] == "triangulated"
    assert result["triangulation"]["consensus_top_suspect"] == "Player 2: blue"
    assert result["triangulation"]["top_choice_agreement"] == 1.0
    assert result["beliefs_about_others"] == {"green": {"blue": 0.7}}
    assert all(request["temperature"] == 0.0 for request in agent.requests)
    assert agent.summarization == "existing summary"
    assert agent.processed_memory == "existing memory"
    prompts = "\n".join(
        message["content"]
        for request in agent.requests
        for message in request["messages"]
    )
    assert 'allowed colors: ["blue", "green"]' in prompts
    assert 'allowed colors: ["red"' not in prompts
    assert '"role": "Impostor"' not in prompts


def test_activation_adapter_receives_only_public_measurement_context():
    seen = {}

    def decoder(context, agent):
        seen.update(context)
        return {"blue": 0.8, "green": 0.2}

    result = asyncio.run(ActivationBeliefProbe(decoder).measure(make_context(), object()))
    assert result["status"] == "collected"
    assert result["impostor_probabilities"]["Player 2: blue"] == 0.8
    assert "world_state" not in seen
    assert seen["players"] == ["Player 2: blue", "Player 3: green"]


def test_behavioral_fork_repairs_disallowed_choices_once():
    agent = FakeAgent(
        [
            '{"vote_for":"red","avoid":"none","trust":"red","follow":"red"}',
            '{"vote_for":"blue","avoid":"blue","trust":"green","follow":"green"}',
        ]
    )

    result = asyncio.run(BehavioralForkBeliefProbe().measure(make_context(), agent))

    assert result["status"] == "collected"
    assert result["format_retry_count"] == 1
    assert len(result["raw_responses"]) == 2
    assert result["choices"]["vote_for"] == "Player 2: blue"


def test_invalid_private_response_becomes_measurement_status():
    agent = FakeAgent(["not json"])
    result = asyncio.run(ElicitedBeliefProbe().measure(make_context(), agent))
    assert result["status"] == "invalid_response"
    assert result["impostor_probabilities"] == {}
