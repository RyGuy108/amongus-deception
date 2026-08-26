#!/usr/bin/env python3
"""Extract Phase 9 activations and run group-safe concept/confound probes."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.interpretability import build_probe_labels, run_probe_suite  # noqa: E402
from amongagents.models import LocalTransformersRuntime  # noqa: E402


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


async def extract(records, model: str):
    runtime = LocalTransformersRuntime(model)
    prompt_vectors = []
    response_vectors = []
    metadata = []
    for index, record in enumerate(records, 1):
        model_input = record.get("model_input", {})
        if not model_input.get("system_prompt") or not model_input.get("user_prompt"):
            continue
        messages = [
            {"role": "system", "content": model_input["system_prompt"]},
            {"role": "user", "content": model_input["user_prompt"]},
        ]
        activation = await runtime.pooled_activations(messages, record["utterance"])
        prompt_vectors.append(activation["prompt_final"])
        response_vectors.append(activation["response_mean"])
        metadata.append(
            {
                "sample_index": len(metadata),
                "source_line": index,
                "match_id": record["match_id"],
                "condition": record["incentive"]["condition"],
                "speaker": record["speaker"],
                "prompt_tokens": activation["prompt_tokens"],
                "response_tokens": activation["response_tokens"],
                "device": activation["device"],
                "labels": build_probe_labels(record),
            }
        )
    if not prompt_vectors:
        raise ValueError("no activation-compatible collected records were found")
    return np.stack(prompt_vectors), np.stack(response_vectors), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--reuse-activations",
        action="store_true",
        help="Reuse activations.npz and activation-metadata.jsonl in the output directory.",
    )
    args = parser.parse_args()
    records = [
        record for record in read_jsonl(args.input) if record.get("status") == "collected"
    ]
    if args.max_samples:
        records = records[: args.max_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = f"local:{args.model.resolve()}"
    activation_path = args.output_dir / "activations.npz"
    metadata_path = args.output_dir / "activation-metadata.jsonl"
    if args.reuse_activations:
        cached = np.load(activation_path)
        prompt = cached["prompt_final"]
        response = cached["response_mean"]
        metadata = list(read_jsonl(metadata_path))
    else:
        prompt, response, metadata = asyncio.run(extract(records, model))
        np.savez_compressed(
            activation_path,
            prompt_final=prompt.astype(np.float16),
            response_mean=response.astype(np.float16),
        )
        with metadata_path.open("w", encoding="utf-8") as stream:
            for item in metadata:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
    labels = [item["labels"] for item in metadata]
    groups = [item["match_id"] for item in metadata]
    report = {
        "schema_version": "amongus.phase9-report.v1",
        "model": model,
        "pooling_sites": {
            "prompt_final": run_probe_suite(
                prompt.astype(np.float32), labels, groups, seed=args.seed
            ),
            "response_mean": run_probe_suite(
                response.astype(np.float32), labels, groups, seed=args.seed
            ),
        },
        "activation_artifact": "activations.npz",
        "metadata_artifact": "activation-metadata.jsonl",
        "human_validated_deception_labels": False,
    }
    (args.output_dir / "mechanistic-probe-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        site: {
            task: details.get("best_layer_metrics", {}).get("balanced_accuracy", {}).get("mean")
            for task, details in suite["probes"].items()
        }
        for site, suite in report["pooling_sites"].items()
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
