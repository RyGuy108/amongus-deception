#!/usr/bin/env python3
"""Add non-target listener replicates until Phase 7 has enough target effects."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))
sys.path.insert(0, str(ROOT))

from amongagents.experiments import CounterfactualInfluenceExperiment  # noqa: E402
from scripts.collect_causal_dataset import ResearchAgent, _context_for  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def has_target_effect(record: dict) -> bool:
    effects = record.get("causal_contrasts", {}).get("original_minus_neutral", {})
    return any(
        subject in effects for subject in record["selection"].get("claim_subjects", [])
    )


async def repair(args: argparse.Namespace) -> dict:
    model = f"local:{args.model.resolve()}"
    matched = read_jsonl(args.matched)
    by_match = {}
    for record in matched:
        by_match.setdefault(record["match_id"], record)
    influence = read_jsonl(args.influence)
    target_effect_source_ids = {
        record["selection"].get("source_record_id") or record["influence_id"]
        for record in influence
        if has_target_effect(record)
    }
    usable = len(target_effect_source_ids)
    used = {
        (
            record["selection"].get("source_record_id"),
            listener["listener"],
        )
        for record in influence
        for listener in record.get("listeners", [])
    }
    attempts = 0
    by_source = {}
    for record in influence:
        source_id = record["selection"].get("source_record_id") or record["influence_id"]
        by_source.setdefault(source_id, record)
    candidates = [
        record
        for source_id, record in by_source.items()
        if source_id not in target_effect_source_ids
    ]
    runner = CounterfactualInfluenceExperiment(
        args.influence,
        experiment_id=influence[0]["experiment_id"],
        max_listeners=1,
        temperature=0.0,
    )
    for original_record in candidates:
        source_id = original_record["selection"].get("source_record_id")
        match_id = str(source_id).split(":", 1)[0]
        matched_record = by_match[match_id]
        frozen = matched_record["frozen_context"]
        speaker_context = {
            "phase": matched_record["phase"],
            "round": matched_record["round"],
            "world_state": frozen["world_state"],
            "observation": frozen["observation"],
            "private_belief": frozen.get("private_belief"),
            "model_metadata": matched_record["model_metadata"],
            "state_source": matched_record["state_source"],
        }
        players = frozen["world_state"]["players"]
        impostors = [player["name"] for player in players if player["role"] == "Impostor"]
        speaker_name = matched_record["speaker"]["name"]
        speaker_index = next(
            index for index, player in enumerate(players) if player["name"] == speaker_name
        )
        target = original_record["selection"]["claim_subjects"][0]
        scenario = int(original_record["game_index"]) - 1
        speaker = ResearchAgent(
            speaker_name,
            matched_record["speaker"]["role"],
            model,
            impostors,
        )
        listener_candidates = [
            (index, player)
            for index, player in enumerate(players)
            if index != speaker_index
            and player["name"] != target
            and player["role"] == "Crewmate"
            and (source_id, player["name"]) not in used
        ]
        for listener_index, listener_state in listener_candidates:
            if usable >= args.minimum_target_effects:
                break
            listener_context = _context_for(
                players,
                listener_index,
                scenario,
                designated_target=target,
                state_source="controlled_schema_scenario_recipient_replicate",
            )
            listener_context["model_metadata"]["model"] = model
            listener = ResearchAgent(
                listener_state["name"],
                listener_state["role"],
                model,
                impostors,
            )
            before = len(read_jsonl(args.influence))
            influence_id = await runner.run(
                speaker_context,
                speaker,
                [(listener, listener_context)],
                original_record["interventions"]["original"],
                game_index=original_record["game_index"],
                statement_source=(
                    original_record["selection"].get("statement_source", "unknown")
                    + "_recipient_replicate"
                ),
                source_record_id=source_id,
            )
            attempts += 1
            if influence_id:
                latest = read_jsonl(args.influence)[before:]
                if any(has_target_effect(record) for record in latest):
                    target_effect_source_ids.add(source_id)
                    usable = len(target_effect_source_ids)
                used.add((source_id, listener_state["name"]))
            print(
                json.dumps(
                    {
                        "attempts": attempts,
                        "target_effect_statements": usable,
                        "listener": listener_state["name"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if source_id in target_effect_source_ids:
                break
        if usable >= args.minimum_target_effects:
            break
    result = {
        "schema_version": "amongus.phase7-recipient-repair.v1",
        "attempts": attempts,
        "target_effect_statement_count": usable,
        "minimum_target_effects": args.minimum_target_effects,
        "passed": usable >= args.minimum_target_effects,
    }
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--influence", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-target-effects", type=int, default=30)
    args = parser.parse_args()
    result = asyncio.run(repair(args))
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
