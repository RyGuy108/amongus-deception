import json
from functools import reduce
from typing import TYPE_CHECKING, List, Dict
import os
import platform
import sys
from datetime import datetime, timezone

if TYPE_CHECKING:
    from pandas import DataFrame


def setup_experiment(experiment_name, LOGS_PATH, DATE, COMMIT_HASH, DEFAULT_ARGS):
    """Set up experiment directory and files with an index-based system."""
    
    os.makedirs(LOGS_PATH, exist_ok=True)

    # Find the next available index for the current date
    next_index = 0
    while os.path.exists(os.path.join(LOGS_PATH, f"{DATE}_exp_{next_index}")):
        next_index += 1
    
    # Create the experiment name with the next index
    experiment_name = f"{DATE}_exp_{next_index}"
    
    experiment_path = os.path.join(LOGS_PATH, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    
    # delete everything in the experiment path
    for file in os.listdir(experiment_path):
        os.remove(os.path.join(experiment_path, file))

    with open(
        os.path.join(experiment_path, "experiment-details.txt"), "w"
    ) as experiment_file:
        experiment_file.write(f"Experiment {experiment_path}\n")
        experiment_file.write(f"Date: {DATE}\n")
        experiment_file.write(f"Commit: {COMMIT_HASH}\n")
        experiment_file.write(f"Experiment args: {DEFAULT_ARGS}\n")
        experiment_file.write(f"Path of executable file: {os.path.abspath(__file__)}\n")
        experiment_file.write(f"Experiment index: {next_index}\n")

    manifest = {
        "schema_version": "amongus.experiment-manifest.v1",
        "experiment_id": experiment_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": COMMIT_HASH,
        "experiment_index": next_index,
        "arguments": DEFAULT_ARGS,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "credentials_present": {
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
        },
        "artifacts": {
            "turn_records": "turn-records.jsonl",
            "legacy_agent_log": "agent-logs.json",
            "legacy_compact_agent_log": "agent-logs-compact.json",
            "summary": "summary.json",
            "incentive_outcomes": "incentive-outcomes.jsonl",
            "belief_reliability_report": "belief-reliability-report.json",
            "matched_incentive_trials": "matched-incentive-trials.jsonl",
            "matched_incentive_report": "matched-incentive-report.json",
            "counterfactual_influence": "counterfactual-influence.jsonl",
            "counterfactual_influence_report": "counterfactual-influence-report.json",
            "taxonomy_annotations": "phase8/taxonomy-annotations.jsonl",
            "taxonomy_report": "phase8/taxonomy-report.json",
            "taxonomy_human_validation": "phase8/taxonomy-human-validation.csv",
            "mechanistic_probe_report": "phase9/mechanistic-probe-report.json",
            "activations": "phase9/activations.npz",
        },
    }
    with open(os.path.join(experiment_path, "experiment.json"), "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")

    os.environ["EXPERIMENT_PATH"] = experiment_path
    os.environ["STREAMLIT"] = str(DEFAULT_ARGS.get("streamlit", False))
    os.environ["EXPERIMENT_INDEX"] = str(next_index)
    
    return experiment_name

def load_game_summary(filepath: str) -> "DataFrame":
    import pandas as pd

    # Read each line of the JSONL file
    with open(filepath, 'r') as file:
        data = [json.loads(line.strip()) for line in file]
    
    # Extract Game, Winner, and Winner Reason
    games_summary = [
        {
            "Game": game_id,
            "Winner": game_details.get("winner"),
            "Winner Reason": game_details.get("winner_reason")
        }
        for entry in data
        for game_id, game_details in entry.items()
    ]
    
    # Create DataFrame
    return pd.DataFrame(games_summary)

def read_jsonl_as_json(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def load_agent_logs_df(path: str) -> "DataFrame":
    from pandas import DataFrame, json_normalize

    df: DataFrame = json_normalize(read_jsonl_as_json(path))
    
    action_cols = [
        "interaction.response.Action",
        "interaction.response.Action.action",
        "interaction.response.SPEAK Strategy.action",
        "interaction.response.ACTION",
        "interaction.response.Thinking Process.action",
    ]
    
    thinking_cols = [
        "interaction.response.Thinking Process",
        "interaction.response.Thinking Process.thought",
        "interaction.response.SPEAK Strategy.thought",
        "interaction.response.SPEAK Strategy",
        "interaction.response",
        "interaction.response.Action.thought",
    ]
    

    df["action"] = reduce(
        lambda x, y: x.combine_first(df[y]) if y in df else x,
        action_cols,
        df.assign(action=None)["action"]  # Start with a column of None
    )
    
    df["thought"] = reduce(
        lambda x, y: x.combine_first(df[y]) if y in df else x,
        thinking_cols,
        df.assign(thought=None)["thought"]  # Start with a column of None
    )
    
    df = df.drop(columns=(action_cols + thinking_cols), errors='ignore')

    return df
