#!/usr/bin/env python3
"""Collect the Phase 4 frozen-context reliability gate with a local model."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))
sys.path.insert(0, str(ROOT))

from amongagents.measurement import build_belief_probe  # noqa: E402
from scripts.collect_causal_dataset import (  # noqa: E402
    COLORS,
    ResearchAgent,
    _context_for,
    _players,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _config_hash(model: Path) -> str:
    payload = json.loads((model / "config.json").read_text())
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


async def collect(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "belief-reliability-contexts.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(
            "collection output already exists; choose a new --output directory or --resume"
        )
    records = _read_jsonl(records_path)
    existing_scenarios = {
        int(record["scenario_index"])
        for record in records
        if "scenario_index" in record
    }
    scenario = max(
        args.start_scenario,
        max(existing_scenarios, default=args.start_scenario - 1) + 1,
    )
    initial_scenario = scenario
    model = f"local:{args.model.resolve()}"
    probe = build_belief_probe(
        "reliability-meeting",
        reliability_repeats=args.repeats,
        reliability_temperature=args.temperature,
    )
    role_counts = Counter(
        str(record.get("speaker", {}).get("role", "unknown")) for record in records
    )
    status_counts = Counter(
        str(record.get("private_belief", {}).get("status", "unknown"))
        for record in records
    )

    while len(records) < args.target_contexts and scenario < args.max_states:
        speaker_index = scenario % len(COLORS)
        speaker_role = "Impostor" if scenario % 3 == 0 else "Crewmate"
        players = _players(scenario, speaker_index, speaker_role)
        desired_target_role = "Impostor" if scenario % 2 == 0 else "Crewmate"
        designated_target = next(
            player["name"]
            for index, player in enumerate(players)
            if index != speaker_index and player["role"] == desired_target_role
        )
        context = _context_for(
            players,
            speaker_index,
            scenario,
            designated_target=designated_target,
        )
        context["model_metadata"]["model"] = model
        impostors = [
            player["name"] for player in players if player["role"] == "Impostor"
        ]
        agent = ResearchAgent(
            players[speaker_index]["name"],
            speaker_role,
            model,
            impostors,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        started = time.perf_counter()
        belief = await probe.measure(context, agent)
        elapsed = time.perf_counter() - started
        record = {
            "schema_version": "amongus.phase4-controlled-context.v1",
            "experiment_id": args.experiment_id,
            "context_id": hashlib.sha256(
                f"{args.experiment_id}:{scenario}".encode()
            ).hexdigest()[:24],
            "scenario_index": scenario,
            "game_index": scenario + 1,
            "speaker": {
                "name": players[speaker_index]["name"],
                "role": speaker_role,
            },
            "model_metadata": context["model_metadata"],
            "state_source": "controlled_schema_scenario",
            "world_state_before": context["world_state"],
            "observation": context["observation"],
            "private_belief": belief,
            "measurement_elapsed_seconds": elapsed,
        }
        _append(records_path, record)
        records.append(record)
        role_counts[speaker_role] += 1
        status_counts[str(belief.get("status", "unknown"))] += 1
        scenario += 1
        print(
            json.dumps(
                {
                    "contexts": len(records),
                    "scenario": scenario,
                    "roles": dict(sorted(role_counts.items())),
                    "statuses": dict(sorted(status_counts.items())),
                    "last_elapsed_seconds": round(elapsed, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = {
        "schema_version": "amongus.phase4-collection-manifest.v1",
        "experiment_id": args.experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "identifier": model,
            "config_sha256": _config_hash(args.model),
        },
        "protocol": {
            "frozen_contexts": len(records),
            "roles": dict(sorted(role_counts.items())),
            "statuses": dict(sorted(status_counts.items())),
            "repeats_per_variant": args.repeats,
            "prompt_variants": ["probability", "suspicion", "trust_inverse"],
            "elicitation_response_schema": "impostor_probabilities_only",
            "temperature": args.temperature,
            "behavioral_fork": True,
            "behavioral_format_retries": 1,
            "scenario_start_this_run": initial_scenario,
            "scenario_stop_exclusive": scenario,
            "seed": args.seed,
            "sampling_reproducibility": "best_effort_local_backend",
            "local_batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
        "gate_target": {
            "minimum_frozen_contexts": args.target_contexts,
            "minimum_role_coverage": ["Crewmate", "Impostor"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "artifacts": {"contexts": records_path.name},
    }
    (args.output / "collection-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="phase4-belief-reliability-local-v1")
    parser.add_argument("--target-contexts", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--allow-underpowered-pilot",
        action="store_true",
        help="Allow fewer than 30 contexts for runtime/format checks only.",
    )
    parser.add_argument("--start-scenario", type=int, default=0)
    parser.add_argument("--max-states", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.target_contexts < 30 and not args.allow_underpowered_pilot:
        parser.error(
            "--target-contexts must be at least 30 for the Phase 4 gate; "
            "use --allow-underpowered-pilot only for runtime/format checks"
        )
    if args.repeats < 20:
        parser.error("--repeats must be at least 20 for the preregistered protocol")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_new_tokens < 32:
        parser.error("--max-new-tokens must be at least 32")
    if args.start_scenario < 0 or args.max_states <= args.start_scenario:
        parser.error("scenario bounds are invalid")
    if args.target_contexts > args.max_states - args.start_scenario:
        parser.error("scenario capacity is smaller than --target-contexts")
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass
    manifest = asyncio.run(collect(args))
    print(json.dumps(manifest["protocol"], sort_keys=True))


if __name__ == "__main__":
    main()
