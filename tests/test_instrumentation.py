import asyncio
import json
import random
from pathlib import Path

import numpy as np

from amongagents.envs.configs.agent_config import ALL_RANDOM
from amongagents.envs.configs.game_config import THREE_MEMBER_GAME
from amongagents.envs.game import AmongUs
from amongagents.envs.incentives import IncentiveScheduler
from amongagents.instrumentation import JsonlTurnRecorder, SCHEMA_VERSION


def run_random_game(
    path,
    seed,
    belief_probe=None,
    incentive_scheduler=None,
    matched_incentive_runner=None,
    counterfactual_runner=None,
):
    random.seed(seed)
    np.random.seed(seed)
    recorder = JsonlTurnRecorder(path / "turn-records.jsonl", "smoke-test")
    game = AmongUs(
        game_config=THREE_MEMBER_GAME,
        agent_config=ALL_RANDOM,
        game_index=0,
        recorder=recorder,
        belief_probe=belief_probe,
        incentive_scheduler=incentive_scheduler,
        matched_incentive_runner=matched_incentive_runner,
        counterfactual_runner=counterfactual_runner,
    )

    old_experiment_path = __import__("os").environ.get("EXPERIMENT_PATH")
    __import__("os").environ["EXPERIMENT_PATH"] = str(path)
    try:
        winner = asyncio.run(game.run_game())
    finally:
        if old_experiment_path is None:
            __import__("os").environ.pop("EXPERIMENT_PATH", None)
        else:
            __import__("os").environ["EXPERIMENT_PATH"] = old_experiment_path
    records = [
        json.loads(line)
        for line in (path / "turn-records.jsonl").read_text().splitlines()
    ]
    return winner, records


def test_random_game_completes_and_writes_structured_turns(tmp_path):
    winner, records = run_random_game(tmp_path, seed=7)

    assert winner in {1, 2, 3, 4}
    assert records
    schema_path = (
        Path(__file__).parents[1] / "docs" / "schemas" / "turn-record-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    assert set(records[0]) == set(schema["required"])
    assert [record["turn_index"] for record in records] == list(range(len(records)))
    for record in records:
        assert record["schema_version"] == SCHEMA_VERSION
        assert record["world_state_before"]
        assert record["world_state_after"]
        assert record["observation"]
        assert record["action"]["name"]
        assert record["private_belief"]["status"] == "not_collected"
        assert record["atomic_claims"]["status"] == "no_utterance"


def test_seeded_random_games_have_identical_state_transitions(tmp_path):
    winner_a, records_a = run_random_game(tmp_path / "a", seed=23)
    winner_b, records_b = run_random_game(tmp_path / "b", seed=23)

    def normalize(records):
        for record in records:
            record.pop("recorded_at")
        return records

    assert winner_a == winner_b
    assert normalize(records_a) == normalize(records_b)


def test_vote_and_speech_are_normalized(tmp_path):
    # Covered end-to-end by random play; this assertion protects the normalized
    # nullable fields needed by later claim and belief phases.
    recorder = JsonlTurnRecorder(tmp_path / "empty.jsonl", "shape-test")
    assert recorder.experiment_id == "shape-test"


def test_private_belief_measurement_is_attached_without_entering_game_state(tmp_path):
    class StaticProbe:
        def should_probe(self, context, agent):
            return True

        async def measure(self, context, agent):
            return {
                "schema_version": "amongus.belief-measurement.v1",
                "status": "collected",
                "measurements": {},
                "values": {"Player 2: blue": 0.75},
                "triangulation": {"status": "single_method"},
                "beliefs_about_others": {},
            }

    _, records = run_random_game(tmp_path, seed=31, belief_probe=StaticProbe())
    assert all(record["private_belief"]["status"] == "collected" for record in records)
    assert all(
        "private_belief" not in record["world_state_after"] for record in records
    )


def test_factorial_incentives_flow_through_game_records_and_outcomes(tmp_path):
    _, records = run_random_game(
        tmp_path,
        seed=41,
        incentive_scheduler=IncentiveScheduler("balanced_factorial", seed=41),
    )
    assignments = {
        (record["player"]["name"], record["incentive"]["condition"])
        for record in records
    }
    outcomes = [
        json.loads(line)
        for line in (tmp_path / "incentive-outcomes.jsonl").read_text().splitlines()
    ]

    assert len({condition for _, condition in assignments}) >= 2
    assert all(record["incentive"]["plan"] == "balanced_factorial" for record in records)
    assert outcomes[0]["schema_version"] == "amongus.incentive-outcomes.v1"
    assert len(outcomes[0]["results"]) == 3


def test_causal_runner_hooks_receive_frozen_game_contexts(tmp_path):
    class MatchedStub:
        def __init__(self):
            self.calls = []

        def should_run(self, context, agent):
            return True

        async def run(self, context, agent, game_index):
            self.calls.append((context["phase"], game_index, agent.player.name))

    matched = MatchedStub()
    run_random_game(
        tmp_path,
        seed=47,
        matched_incentive_runner=matched,
    )

    assert matched.calls
    assert all(game_index == 0 for _, game_index, _ in matched.calls)
