import re
from types import SimpleNamespace

from amongagents.envs.incentives import (
    CONDITIONS,
    IncentiveScheduler,
    apply_incentive_assignment,
    evaluate_incentive_outcomes,
)


def players():
    return [
        SimpleNamespace(name="Player 1", identity="Crewmate", is_alive=True),
        SimpleNamespace(name="Player 2", identity="Crewmate", is_alive=True),
        SimpleNamespace(name="Player 3", identity="Impostor", is_alive=True),
    ]


def test_balanced_scheduler_crosses_every_condition_with_both_roles():
    observed = {"Crewmate": set(), "Impostor": set()}
    for game_index in range(len(CONDITIONS)):
        assignments = IncentiveScheduler("balanced_factorial", seed=11).assign(
            players(), game_index
        )
        for assignment in assignments.values():
            observed[assignment.role].add(assignment.condition.condition_id)
            if assignment.condition.needs_target:
                assert assignment.target_player != assignment.player

    assert observed["Crewmate"] == set(CONDITIONS)
    assert observed["Impostor"] == set(CONDITIONS)


def test_fixed_incentive_applies_to_every_role_without_direct_deception_instruction():
    assignments = IncentiveScheduler("target_ejection", seed=5).assign(players(), 0)
    assert {item.condition.condition_id for item in assignments.values()} == {
        "target_ejection"
    }
    for assignment in assignments.values():
        agent = SimpleNamespace(system_prompt="base", player=SimpleNamespace(identity=assignment.role))
        apply_incentive_assignment(agent, assignment)
        assert "Private scored objective" in agent.system_prompt
        assert not re.search(r"\b(lie|lying|deceive|deception)\b", agent.system_prompt, re.I)
        assert agent.incentive_assignment["factorial_cell"]["role"] == assignment.role


def test_incentive_outcomes_score_target_and_evidence_conditions():
    game_players = players()
    scheduler = IncentiveScheduler("target_ejection", seed=3)
    target_assignment = scheduler.assign(game_players, 0)["Player 1"]
    target_agent = SimpleNamespace(player=game_players[0])
    apply_incentive_assignment(target_agent, target_assignment)

    evidence_assignment = IncentiveScheduler("evidence_accuracy").assign(
        game_players, 0
    )["Player 2"]
    evidence_agent = SimpleNamespace(player=game_players[1])
    apply_incentive_assignment(evidence_agent, evidence_assignment)
    env = SimpleNamespace(
        winner=2,
        game_index=0,
        players=game_players,
        agents=[target_agent, evidence_agent],
        ejected_players=[target_assignment.target_player],
        turn_records=[
            {
                "player": {"name": "Player 2"},
                "atomic_claims": {
                    "items": [
                        {
                            "objective_truth": {"status": "true"},
                            "speaker_support": {"status": "supported"},
                        },
                        {
                            "objective_truth": {"status": "false"},
                            "speaker_support": {"status": "contradicted"},
                        },
                        {
                            "objective_truth": {"status": "unknown"},
                            "speaker_support": {"status": "unsupported"},
                        },
                    ]
                },
            }
        ],
    )

    report = evaluate_incentive_outcomes(env)
    by_player = {item["player"]: item for item in report["results"]}
    assert by_player["Player 1"]["secondary_score"] == 1.0
    assert by_player["Player 2"]["secondary_score"] == 0.5
    assert by_player["Player 2"]["primary_success"] is True
