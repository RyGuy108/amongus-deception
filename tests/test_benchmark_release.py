import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILD_RELEASE = _load_script("build_benchmark_release")


def test_release_id_rejects_path_segments():
    assert BUILD_RELEASE.release_id("amongus-v0.2.0") == "amongus-v0.2.0"
    with pytest.raises(Exception):
        BUILD_RELEASE.release_id("../unsafe")


def test_release_gate_status_uses_packet_manifest_and_report(tmp_path):
    manifest = {
        "packets": {
            "claims": {"item_count": 307},
            "taxonomy": {"item_count": 120},
        }
    }
    (tmp_path / "validation-manifest.json").write_text(json.dumps(manifest))

    assert BUILD_RELEASE.validation_gate_status(
        tmp_path, "claims", 400, minimum=300
    ) == (
        "pending human validation; 307 items prepared; minimum 300 met; "
        "stretch target 400"
    )

    (tmp_path / "claims-validation-report.json").write_text(
        json.dumps({"gate": {"passed": True}})
    )
    assert BUILD_RELEASE.validation_gate_status(
        tmp_path, "claims", 400, minimum=300
    ) == (
        "validated; 307 items resolved"
    )
    assert BUILD_RELEASE.validation_gate_status(tmp_path, "control", 120) == (
        "pending human validation; 0 of target 120 items prepared"
    )


def test_release_gate_status_reads_phase4_decision(tmp_path):
    (tmp_path / "belief-reliability-report.json").write_text(
        json.dumps(
            {
                "frozen_context_count": 30,
                "decision_gate": {
                    "recommendation": "C_investigate_belief_report_faithfulness"
                },
            }
        )
    )

    assert BUILD_RELEASE.belief_reliability_gate_status(tmp_path) == (
        "completed on 30 contexts; C_investigate_belief_report_faithfulness"
    )
