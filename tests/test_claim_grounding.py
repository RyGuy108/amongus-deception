import json
import importlib.util
from pathlib import Path

from amongagents.analysis import GroundingPipeline


SAMPLER_PATH = Path(__file__).parents[1] / "scripts" / "build_claim_validation_set.py"
SAMPLER_SPEC = importlib.util.spec_from_file_location("claim_sampler", SAMPLER_PATH)
SAMPLER = importlib.util.module_from_spec(SAMPLER_SPEC)
assert SAMPLER_SPEC.loader is not None
SAMPLER_SPEC.loader.exec_module(SAMPLER)


def make_record(utterance, speaker_role="Crewmate", known_impostors=None):
    return {
        "game_index": 2,
        "turn_index": 4,
        "player": {"name": "Player 1: red", "role": speaker_role},
        "utterance": utterance,
        "world_state_before": {
            "players": [
                {"name": "Player 1: red", "role": speaker_role, "alive": True, "location": "Electrical"},
                {"name": "Player 2: blue", "role": "Impostor", "alive": True, "location": "MedBay"},
                {"name": "Player 3: green", "role": "Crewmate", "alive": True, "location": "Electrical"},
                {"name": "Player 4: yellow", "role": "Crewmate", "alive": False, "location": "MedBay"},
            ],
            "activity_history": [
                {
                    "timestep": 3,
                    "phase": "task",
                    "player": "Player 2: blue",
                    "action": {
                        "name": "KILL",
                        "text": "KILL Player 4: yellow",
                        "target_player": "Player 4: yellow",
                    },
                }
            ],
        },
        "observation": {
            "self_state": {"name": "Player 1: red", "location": "Electrical", "alive": True},
            "visible_players": ["Player 3: green"],
            "private_role_knowledge": {
                "own_role": speaker_role,
                "known_impostors": known_impostors or [],
            },
            "location_information": "Current Location: Electrical; Players: Player 3: green",
            "observation_history": ["Player 2: blue KILL Player 4: yellow"],
            "action_history": [],
            "available_actions": [],
        },
    }


def claims_by_predicate(annotation):
    return {claim["predicate"]: claim for claim in annotation["items"]}


def test_extracts_and_separately_grounds_world_and_observation_evidence():
    record = make_record(
        "I am in Electrical with green. Blue is the impostor. I saw blue kill yellow."
    )
    annotation = GroundingPipeline().annotate_record(record)
    claims = claims_by_predicate(annotation)
    self_location = next(
        claim
        for claim in annotation["items"]
        if claim["predicate"] == "located_at" and claim["subject"] == "Player 1: red"
    )

    assert annotation["status"] == "completed"
    assert self_location["objective_truth"]["status"] == "true"
    assert self_location["speaker_support"]["status"] == "supported"
    assert claims["role_is_impostor"]["objective_truth"]["status"] == "true"
    assert claims["role_is_impostor"]["speaker_support"]["status"] == "unsupported"
    assert claims["killed"]["objective_truth"]["status"] == "true"
    assert claims["killed"]["speaker_support"]["status"] == "supported"
    assert claims["killed"]["preliminary_label"] == "grounded_truth"


def test_known_self_role_can_identify_candidate_misrepresentation():
    record = make_record(
        "Red is not the impostor.",
        speaker_role="Impostor",
        known_impostors=["Player 1: red", "Player 2: blue"],
    )
    annotation = GroundingPipeline().annotate_record(record)
    claim = annotation["items"][0]

    assert claim["predicate"] == "role_is_not_impostor"
    assert claim["objective_truth"]["status"] == "false"
    assert claim["speaker_support"]["status"] == "contradicted"
    assert claim["preliminary_label"] == "candidate_misrepresentation"


def test_absence_claim_remains_unknown_without_complete_observation_window():
    annotation = GroundingPipeline().annotate_record(make_record("I never saw blue."))
    claim = annotation["items"][0]
    assert claim["predicate"] == "observed_absence"
    assert claim["objective_truth"]["status"] == "unknown"
    assert claim["speaker_support"]["status"] == "unknown"


def test_private_belief_is_compared_with_public_role_claim():
    record = make_record("Blue is the impostor.")
    record["private_belief"] = {
        "status": "collected",
        "values": {"Player 2: blue": 0.1, "Player 3: green": 0.4},
    }
    claim = GroundingPipeline().annotate_record(record)["items"][0]
    schema = json.loads(
        (Path(__file__).parents[1] / "docs" / "schemas" / "atomic-claim-v1.schema.json").read_text()
    )
    assert set(claim) == set(schema["required"])
    assert claim["belief_alignment"] == {
        "status": "computed",
        "private_impostor_probability": 0.1,
        "public_implied_probability": 1.0,
        "absolute_divergence": 0.9,
        "direction": "public_more_suspicious",
    }


def test_validation_sampling_round_robins_across_claim_strata():
    rows = []
    for index, (claim_type, label) in enumerate(
        [("role", "grounded_truth")] * 4
        + [("event", "unsupported_falsehood")] * 4
        + [("location", "unverified")] * 4
    ):
        rows.append(({}, {"claim_type": claim_type, "preliminary_label": label, "id": index}))
    selected = SAMPLER.stratified_sample(rows, target=6, seed=11)
    selected_strata = [
        (claim["claim_type"], claim["preliminary_label"]) for _, claim in selected
    ]
    assert len(selected) == 6
    assert len(set(selected_strata)) == 3
