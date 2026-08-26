import random

import numpy as np

from amongagents.envs.configs.agent_config import ALL_LLM
from amongagents.envs.configs.game_config import SEVEN_MEMBER_GAME
from amongagents.envs.game import AmongUs


def test_every_impostor_prompt_receives_complete_teammate_list(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_PATH", str(tmp_path))
    random.seed(43)
    np.random.seed(43)
    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        agent_config=ALL_LLM,
        game_index=0,
    )
    game.initialize_game()

    assert len(game.list_of_impostors) == SEVEN_MEMBER_GAME["num_impostors"]
    for agent in game.agents:
        if agent.player.identity == "Impostor":
            assert str(game.list_of_impostors) in agent.system_prompt
