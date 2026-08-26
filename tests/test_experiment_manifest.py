import json
from pathlib import Path

from utils import setup_experiment


def test_setup_experiment_writes_safe_structured_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    args = {"seed": 17, "agent_config": {"Crewmate": "Random", "Impostor": "Random"}}
    name = setup_experiment(
        experiment_name="ignored-by-upstream-indexing",
        LOGS_PATH=str(tmp_path),
        DATE="2026-08-18",
        COMMIT_HASH="abc123",
        DEFAULT_ARGS=args,
    )

    manifest_path = Path(__import__("os").environ["EXPERIMENT_PATH"]) / "experiment.json"
    manifest = json.loads(manifest_path.read_text())
    assert name == "2026-08-18_exp_0"
    assert manifest["repository_commit"] == "abc123"
    assert manifest["arguments"] == args
    assert manifest["credentials_present"] == {"openai": False, "openrouter": False}
    assert manifest["artifacts"]["turn_records"] == "turn-records.jsonl"
