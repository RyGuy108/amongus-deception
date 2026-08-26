# !/usr/bin/env python3
# usage: main.py [-h] [--name NAME]

import os
import sys
import asyncio
import random
import numpy as np

from typing import Optional, List

sys.path.append(os.path.join(os.path.abspath("."), "among-agents"))

import argparse
import datetime
import subprocess

from amongagents.envs.configs.agent_config import ALL_LLM
from amongagents.envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME, FIVE_MEMBER_GAME
from amongagents.envs.configs.map_config import map_coords
from amongagents.envs.game import AmongUs
from amongagents.envs.incentives import build_incentive_scheduler
from amongagents.experiments import (
    CounterfactualInfluenceExperiment,
    MatchedIncentiveExperiment,
)
from amongagents.instrumentation import JsonlTurnRecorder
from amongagents.measurement import build_belief_probe
from amongagents.UI.MapUI import MapUI
from dotenv import load_dotenv

from utils import setup_experiment

ROOT_PATH = os.path.abspath(".")
LOGS_PATH = os.path.join(ROOT_PATH, "expt-logs")
ASSETS_PATH = os.path.join(ROOT_PATH, "among-agents", "amongagents", "assets")
BLANK_MAP_IMAGE = os.path.join(ASSETS_PATH, "blankmap.png")

load_dotenv()

DATE = datetime.datetime.now().strftime("%Y-%m-%d")
COMMIT_HASH = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)

BIG_LIST_OF_MODELS: List[str] = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3.7-sonnet:thinking",
    "anthropic/claude-3.7-sonnet",
    "openai/o3",
    "openai/o4-mini-high",
    "openai/gpt-4o",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-distill-llama-70b",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.0-flash-001",
    "google/gemma-3-4b-it",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct",
    "x-ai/grok-3-beta",
    "microsoft/phi-4",
]

ARGS = {
    "game_config": SEVEN_MEMBER_GAME,
    "include_human": False,
    "test": False,
    "personality": False,
    "agent_config": {
        "Impostor": "LLM",
        "Crewmate": "LLM",
        "IMPOSTOR_LLM_CHOICES": BIG_LIST_OF_MODELS,
        "CREWMATE_LLM_CHOICES": BIG_LIST_OF_MODELS,
    },
    "UI": False,
    "seed": None,
    "belief_probe_mode": "none",
    "belief_reliability_repeats": 20,
    "belief_reliability_temperature": 0.7,
    "incentive_plan": "role_default",
    "matched_incentive_forks": False,
    "matched_incentive_repeats": 1,
    "matched_incentive_temperature": 0.0,
    "counterfactual_influence": False,
    "counterfactual_max_listeners": 3,
    "counterfactual_temperature": 0.0,
}

async def multiple_games(experiment_name=None, num_games=1, rate_limit=50):
    experiment_name = setup_experiment(experiment_name, LOGS_PATH, DATE, COMMIT_HASH, ARGS)
    recorder = JsonlTurnRecorder(
        os.path.join(os.environ["EXPERIMENT_PATH"], "turn-records.jsonl"),
        experiment_id=experiment_name,
    )
    belief_probe = build_belief_probe(
        ARGS["belief_probe_mode"],
        reliability_repeats=ARGS["belief_reliability_repeats"],
        reliability_temperature=ARGS["belief_reliability_temperature"],
    )
    matched_incentive_runner = (
        MatchedIncentiveExperiment(
            os.path.join(
                os.environ["EXPERIMENT_PATH"], "matched-incentive-trials.jsonl"
            ),
            experiment_id=experiment_name,
            repeats=ARGS["matched_incentive_repeats"],
            temperature=ARGS["matched_incentive_temperature"],
        )
        if ARGS["matched_incentive_forks"]
        else None
    )
    counterfactual_runner = (
        CounterfactualInfluenceExperiment(
            os.path.join(
                os.environ["EXPERIMENT_PATH"], "counterfactual-influence.jsonl"
            ),
            experiment_id=experiment_name,
            max_listeners=ARGS["counterfactual_max_listeners"],
            temperature=ARGS["counterfactual_temperature"],
        )
        if ARGS["counterfactual_influence"]
        else None
    )
    ui = MapUI(BLANK_MAP_IMAGE, map_coords, debug=False) if ARGS["UI"] else None
    with open(os.path.join(os.environ["EXPERIMENT_PATH"], "experiment-details.txt"), "a") as experiment_file:
        experiment_file.write(f"\nExperiment args: {ARGS}\n")

    semaphore = asyncio.Semaphore(rate_limit)

    async def run_limited_game(game_index):
        async with semaphore:
            if ARGS.get("tournament_style") == "1on1":
                # Randomly select one model for each role for this specific game
                game_config = ARGS["agent_config"].copy()
                game_config["CREWMATE_LLM_CHOICES"] = [random.choice(BIG_LIST_OF_MODELS)]
                game_config["IMPOSTOR_LLM_CHOICES"] = [random.choice(BIG_LIST_OF_MODELS)]
            else:
                game_config = ARGS["agent_config"]
                
            game = AmongUs(
                game_config=ARGS["game_config"],
                include_human=ARGS["include_human"],
                test=ARGS["test"],
                personality=ARGS["personality"],
                agent_config=game_config,
                UI=ui,
                game_index=game_index,
                recorder=recorder,
                belief_probe=belief_probe,
                incentive_scheduler=build_incentive_scheduler(
                    ARGS["incentive_plan"], ARGS["seed"]
                ),
                matched_incentive_runner=matched_incentive_runner,
                counterfactual_runner=counterfactual_runner,
            )
            await game.run_game()

    tasks = [run_limited_game(i) for i in range(1, num_games+1)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AmongUs experiment.")
    parser.add_argument("--name", type=str, default=None, help="Optional name for the experiment.")
    parser.add_argument("--num_games", type=int, default=2, help="Number of games to run.")
    parser.add_argument("--display_ui", type=bool, default=False, help="Display UI.")
    parser.add_argument("--crewmate_llm", type=str, default=None, help="Crewmate LLM model.")
    parser.add_argument("--impostor_llm", type=str, default=None, help="Impostor LLM model.")
    parser.add_argument("--streamlit", type=bool, default=False, help="Streamlit.")
    parser.add_argument("--tournament_style", type=str, default="random", help="random or 1on1.")
    parser.add_argument("--seed", type=int, default=None, help="Seed Python and NumPy randomness.")
    parser.add_argument(
        "--belief-probe",
        choices=[
            "none",
            "elicited-meeting",
            "triangulated-meeting",
            "triangulated-all",
            "reliability-meeting",
        ],
        default="none",
        help="Private belief measurement mode; non-none modes add model calls.",
    )
    parser.add_argument(
        "--belief-reliability-repeats",
        type=int,
        default=20,
        help="Repeated samples per paraphrase in the opt-in Phase 4 reliability probe.",
    )
    parser.add_argument(
        "--belief-reliability-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for Phase 4 repeated belief elicitation.",
    )
    parser.add_argument(
        "--incentive-plan",
        choices=[
            "role_default",
            "balanced_factorial",
            "target_ejection",
            "target_protection",
            "evidence_accuracy",
        ],
        default="role_default",
        help="Private scored incentive assignment; balanced_factorial crosses role and incentive.",
    )
    parser.add_argument(
        "--matched-incentive-forks",
        action="store_true",
        help="Run Phase 6 matched utterance forks during eligible meeting turns.",
    )
    parser.add_argument(
        "--matched-incentive-repeats",
        type=int,
        default=1,
        help="Samples per incentive condition for each frozen Phase 6 state.",
    )
    parser.add_argument(
        "--matched-incentive-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Phase 6 matched forks.",
    )
    parser.add_argument(
        "--counterfactual-influence",
        action="store_true",
        help="Run Phase 7 listener-response interventions for candidate falsehoods.",
    )
    parser.add_argument(
        "--counterfactual-max-listeners",
        type=int,
        default=3,
        help="Maximum frozen listeners measured per eligible statement.",
    )
    parser.add_argument(
        "--counterfactual-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Phase 7 listener-response probes.",
    )
    args = parser.parse_args()
    if args.belief_reliability_repeats < 2:
        parser.error("--belief-reliability-repeats must be at least 2")
    if args.belief_reliability_temperature < 0:
        parser.error("--belief-reliability-temperature must be non-negative")
    if args.matched_incentive_repeats < 1:
        parser.error("--matched-incentive-repeats must be positive")
    if args.matched_incentive_temperature < 0:
        parser.error("--matched-incentive-temperature must be non-negative")
    if args.counterfactual_max_listeners < 1:
        parser.error("--counterfactual-max-listeners must be positive")
    if args.counterfactual_temperature < 0:
        parser.error("--counterfactual-temperature must be non-negative")
    if args.num_games > 1 or args.display_ui == False:
        ARGS["UI"] = False
    if args.crewmate_llm:
        ARGS["agent_config"]["CREWMATE_LLM_CHOICES"] = [args.crewmate_llm]
    if args.impostor_llm:
        ARGS["agent_config"]["IMPOSTOR_LLM_CHOICES"] = [args.impostor_llm]
    ARGS["tournament_style"] = args.tournament_style
    ARGS["seed"] = args.seed
    ARGS["belief_probe_mode"] = args.belief_probe
    ARGS["belief_reliability_repeats"] = args.belief_reliability_repeats
    ARGS["belief_reliability_temperature"] = args.belief_reliability_temperature
    ARGS["incentive_plan"] = args.incentive_plan
    ARGS["matched_incentive_forks"] = args.matched_incentive_forks
    ARGS["matched_incentive_repeats"] = args.matched_incentive_repeats
    ARGS["matched_incentive_temperature"] = args.matched_incentive_temperature
    ARGS["counterfactual_influence"] = args.counterfactual_influence
    ARGS["counterfactual_max_listeners"] = args.counterfactual_max_listeners
    ARGS["counterfactual_temperature"] = args.counterfactual_temperature
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    asyncio.run(multiple_games(experiment_name=args.name, num_games=args.num_games))
