#!/usr/bin/env python3
"""Run Phase 10 controlled-intent transfer tests without OOD retraining."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "among-agents"))

from amongagents.interpretability import (  # noqa: E402
    build_generalization_suite,
    evaluate_frozen_generalization,
)
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
        messages = [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["user_prompt"]},
        ]
        activation = await runtime.pooled_activations(messages, record["continuation"])
        prompt_vectors.append(activation["prompt_final"])
        response_vectors.append(activation["response_mean"])
        metadata.append(
            {
                **record,
                "activation": {
                    "prompt_tokens": activation["prompt_tokens"],
                    "response_tokens": activation["response_tokens"],
                    "device": activation["device"],
                },
            }
        )
        if index % 24 == 0 or index == len(records):
            print(f"activation progress: {index}/{len(records)}", flush=True)
    return np.stack(prompt_vectors), np.stack(response_vectors), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-groups", type=int, default=12)
    parser.add_argument("--test-groups", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--reuse-activations",
        action="store_true",
        help="Reuse the activation and metadata artifacts in the output directory.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    activation_path = args.output_dir / "activations.npz"
    metadata_path = args.output_dir / "controlled-generalization-dataset.jsonl"
    if args.reuse_activations:
        cached = np.load(activation_path)
        prompt = cached["prompt_final"]
        response = cached["response_mean"]
        records = list(read_jsonl(metadata_path))
    else:
        records = build_generalization_suite(
            train_groups=args.train_groups, test_groups=args.test_groups
        )
        model = f"local:{args.model.resolve()}"
        prompt, response, records = asyncio.run(extract(records, model))
        np.savez_compressed(
            activation_path,
            prompt_final=prompt.astype(np.float16),
            response_mean=response.astype(np.float16),
        )
        with metadata_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    model = f"local:{args.model.resolve()}"
    suites = Counter(record["suite"] for record in records)
    report = {
        "schema_version": "amongus.phase10-report.v1",
        "model": model,
        "sample_count": len(records),
        "suite_counts": dict(sorted(suites.items())),
        "controlled_label_design": {
            "positive": "public statement conflicts with stated private belief and advances the stated objective",
            "negative_controls": [
                "objectively false but belief-aligned mistaken statement",
                "objectively true and belief-aligned statement",
            ],
            "truth_value_balanced_against_intent": True,
            "natural_phase8_labels_used_for_training": False,
        },
        "pooling_sites": {
            "prompt_final": evaluate_frozen_generalization(
                prompt.astype(np.float32), records, seed=args.seed
            ),
            "response_mean": evaluate_frozen_generalization(
                response.astype(np.float32), records, seed=args.seed
            ),
        },
        "coverage": {
            "lexical_ood": "evaluated",
            "scenario_ood": "evaluated",
            "role_ood": "evaluated",
            "incentive_ood": "evaluated",
            "deception_category_ood": "evaluated",
            "game_ood": "evaluated",
            "model_ood": "unavailable",
            "domain_ood": "evaluated",
        },
        "artifacts": {
            "dataset": metadata_path.name,
            "activations": activation_path.name,
        },
        "primary_pooling_site": "response_mean",
        "interpretation_boundary": (
            "The controlled labels operationalize intentional belief-statement mismatch. "
            "They test frozen transfer but do not validate natural trajectory labels."
        ),
    }
    report_path = args.output_dir / "frozen-generalization-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        site: {
            suite: details.get("balanced_accuracy")
            for suite, details in result["suites"].items()
        }
        for site, result in report["pooling_sites"].items()
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
