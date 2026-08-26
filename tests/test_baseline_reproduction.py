import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "reproduce_baseline.py"
SPEC = importlib.util.spec_from_file_location("reproduce_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_update_elo_is_zero_sum():
    winner, loser = MODULE.update_elo(1500, 1500)
    assert winner == 1516
    assert loser == 1484
    assert winner + loser == 3000


def test_role_specific_elos_preserve_notebook_update_semantics():
    game = {
        "winner": 1,
        "Player 1": {"identity": "Impostor", "model": "model/a"},
        "Player 2": {"identity": "Crewmate", "model": "model/b"},
        "Player 3": {"identity": "Crewmate", "model": "model/b"},
    }
    deception, detection, counts = MODULE.role_specific_elos([game])
    assert deception["model/a"] == 1516
    assert detection["model/b"] == 1484
    assert counts["model/b"] == {"wins": 0, "games": 2}
