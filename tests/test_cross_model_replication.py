import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def _load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CROSS_MODEL = _load_script("analyze_cross_model_replication")


def _write_model(directory, hidden_size):
    records = []
    labels = []
    for suite, count in (("train", 12), ("iid", 8), ("lexical_ood", 8)):
        for index in range(count):
            label = index % 2
            labels.append(label)
            records.append(
                {
                    "sample_id": f"{suite}-{index}",
                    "suite": suite,
                    "group_id": f"{suite}-group-{index // 2}",
                    "strategic_deception": label,
                    "deception_category": "explicit_falsehood" if label else "truth",
                    "belief_statement_mismatch": label,
                }
            )
    activations = np.zeros((len(records), 2, hidden_size), dtype=np.float32)
    activations[:, 1, 0] = [4.0 if label else -4.0 for label in labels]
    directory.mkdir()
    with (directory / "controlled-generalization-dataset.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    np.savez_compressed(directory / "activations.npz", response_mean=activations)
    (directory / "frozen-generalization-report.json").write_text(
        json.dumps(
            {
                "model": f"test-model-{hidden_size}",
                "pooling_sites": {"response_mean": {"selected_layer": 1}},
            }
        )
    )


def test_cross_model_replication_supports_incompatible_hidden_sizes(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_model(reference, hidden_size=4)
    _write_model(candidate, hidden_size=7)

    report = CROSS_MODEL.analyze(reference, candidate)

    assert report["coordinate_design"]["reference_hidden_size"] == 4
    assert report["coordinate_design"]["candidate_hidden_size"] == 7
    assert report["coordinate_design"]["direct_weight_transfer"] is False
    assert report["replicated_suite_count"] == 2
    assert report["gate"]["all_suites_replicated"] is True
