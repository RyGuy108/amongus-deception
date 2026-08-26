#!/usr/bin/env python3
"""Measure online forward-pass cost for the frozen Phase 12 activation monitor."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.models import LocalTransformersRuntime  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _round_robin_sample(
    records: list[dict[str, Any]], target: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["suite"])].append(record)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < min(target, len(records)):
        progressed = False
        for suite in sorted(groups):
            if groups[suite] and len(selected) < target:
                selected.append(groups[suite].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


async def measure(args: argparse.Namespace) -> dict[str, Any]:
    records = _read_jsonl(args.phase10_dir / "controlled-generalization-dataset.jsonl")
    test_records = [record for record in records if record["suite"] != "train"]
    selected = _round_robin_sample(test_records, args.sample_size, args.seed)
    if not selected:
        raise ValueError("no non-training records available for latency measurement")
    phase10 = json.loads(
        (args.phase10_dir / "frozen-generalization-report.json").read_text()
    )
    layer = int(phase10["pooling_sites"]["response_mean"]["selected_layer"])
    runtime = LocalTransformersRuntime(f"local:{args.model.resolve()}")

    warmup = selected[0]
    await runtime.pooled_activations(
        [
            {"role": "system", "content": warmup["system_prompt"]},
            {"role": "user", "content": warmup["user_prompt"]},
        ],
        warmup["continuation"],
    )

    calls = []
    for index, record in enumerate(selected, 1):
        messages = [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["user_prompt"]},
        ]
        started = time.perf_counter()
        pooled = await runtime.pooled_activations(messages, record["continuation"])
        latency = time.perf_counter() - started
        calls.append(
            {
                "sample_id": record["sample_id"],
                "suite": record["suite"],
                "latency_seconds": latency,
                "prompt_tokens": pooled["prompt_tokens"],
                "response_tokens": pooled["response_tokens"],
            }
        )
        if index % 12 == 0 or index == len(selected):
            print(f"activation latency progress: {index}/{len(selected)}", flush=True)

    latencies = [call["latency_seconds"] for call in calls]
    total_tokens = [call["prompt_tokens"] + call["response_tokens"] for call in calls]
    population = len(test_records)
    mean_latency = statistics.fmean(latencies)
    mean_tokens = statistics.fmean(total_tokens)
    return {
        "schema_version": "amongus.activation-monitor-cost.v1",
        "model": f"local:{args.model.resolve()}",
        "activation_layer": layer,
        "measurement_design": {
            "warmup_discarded": True,
            "stratified_by_suite": True,
            "measured_call_count": len(calls),
            "population_call_count": population,
            "seed": args.seed,
            "device": runtime.device,
        },
        "input_tokens": int(round(mean_tokens * population)),
        "output_tokens": 0,
        "latency_seconds": mean_latency * population,
        "mean_latency_seconds": mean_latency,
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "measured_total_latency_seconds": sum(latencies),
        "estimated_population_total_latency_seconds": mean_latency * population,
        "mean_forward_tokens": mean_tokens,
        "scope": (
            "stratified local online forward-pass estimate scaled to all non-training "
            "Phase 12 messages; generation and rewrite costs excluded"
        ),
        "calls": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase10-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.sample_size < 8:
        parser.error("--sample-size must be at least 8 to cover every test suite")
    report = asyncio.run(measure(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "mean_latency_seconds": report["mean_latency_seconds"],
                "p95_latency_seconds": report["p95_latency_seconds"],
                "estimated_population_total_latency_seconds": report[
                    "estimated_population_total_latency_seconds"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
