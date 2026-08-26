"""Non-invasive, structured turn logging for deception experiments.

The recorder intentionally distinguishes facts that are available now from fields
that later phases measure. Beliefs carry an explicit ``not_collected`` status when
probing is disabled, and claims retain conservative unknown states when the logged
evidence is insufficient.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from amongagents.analysis import GroundingPipeline


SCHEMA_VERSION = "amongus.turn-record.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_index(env: Any) -> Optional[int]:
    if env.current_phase != "meeting":
        return None
    return env.game_config["discussion_rounds"] - env.discussion_rounds_left


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "name"):
        return getattr(value, "name")
    return str(value)


def serialize_action(action: Any) -> Dict[str, Any]:
    """Return a stable JSON representation for any game action."""
    record: Dict[str, Any] = {
        "class": action.__class__.__name__,
        "name": getattr(action, "name", action.__class__.__name__),
        "text": str(action),
        "current_location": getattr(action, "current_location", None),
    }
    optional_attributes = {
        "new_location": "new_location",
        "message": "message",
    }
    for output_name, attribute_name in optional_attributes.items():
        if hasattr(action, attribute_name):
            record[output_name] = _jsonable(getattr(action, attribute_name))
    if hasattr(action, "other_player"):
        record["target_player"] = action.other_player.name
    if hasattr(action, "task"):
        record["task"] = {
            "name": action.task.name,
            "location": action.task.location,
            "type": action.task.task_type,
        }
    return record


def snapshot_observation(player: Any, env: Any) -> Dict[str, Any]:
    """Capture exactly what the acting player can inspect before acting."""
    action_history = []
    for item in player.action_history:
        serialized = {
            "timestep": item["timestep"],
            "phase": item["phase"],
            "action": serialize_action(item["action"]),
        }
        if "round" in item:
            serialized["round"] = item["round"]
        action_history.append(serialized)

    return {
        "self_state": {
            "name": player.name,
            "location": player.location,
            "alive": player.is_alive,
        },
        "visible_players": [
            other.name
            for other in env.map.get_players_in_room(
                player.location, include_new_deaths=True
            )
            if other != player
        ],
        "private_role_knowledge": {
            "own_role": player.identity,
            "known_impostors": (
                list(env.list_of_impostors) if player.identity == "Impostor" else []
            ),
        },
        "location_information": player.location_info,
        "observation_history": list(player.observation_history),
        "action_history": action_history,
        "available_actions": [serialize_action(action) for action in player.available_actions],
    }


def _task_snapshot(task: Any) -> Dict[str, Any]:
    return {
        "name": task.name,
        "location": task.location,
        "type": task.task_type,
        "remaining_steps": task.duration,
        "completed": bool(task.is_completed or task.duration <= 0),
    }


def snapshot_world(env: Any) -> Dict[str, Any]:
    """Capture ground-truth state without mutating the environment."""
    players = []
    for player in env.players:
        players.append(
            {
                "name": player.name,
                "color": player.color,
                "role": player.identity,
                "alive": player.is_alive,
                "location": player.location,
                "reported_death": player.reported_death,
                "tasks": [_task_snapshot(task) for task in player.tasks],
                "kill_cooldown": getattr(player, "kill_cooldown", None),
            }
        )

    eligible_task_records = [
        task
        for task in env.task_assignment.assigned_tasks
        if getattr(task, "assigned_player", None) is not None
        and task.assigned_player.is_alive
    ]
    completed_tasks = sum(
        1
        for task in eligible_task_records
        if task.is_completed or task.duration <= 0
    )
    votes = {
        getattr(player, "name", str(player)): count
        for player, count in env.votes.items()
    }
    activity_history = []
    for event in env.activity_log:
        activity_record = {
            "timestep": event["timestep"],
            "phase": event["phase"],
            "player": getattr(event["player"], "name", str(event["player"])),
            "action": serialize_action(event["action"]),
        }
        if "round" in event:
            activity_record["round"] = event["round"]
        activity_history.append(activity_record)
    return {
        "timestep": env.timestep,
        "phase": env.current_phase,
        "round": _round_index(env),
        "players": players,
        "votes": votes,
        "vote_intentions": dict(env.vote_info_one_round),
        "task_progress": {
            "completed": completed_tasks,
            "eligible": len(eligible_task_records),
        },
        "activity_history": activity_history,
        "important_events": _jsonable(env.important_activity_log),
    }


def _model_metadata(agent: Any) -> Dict[str, Any]:
    model = getattr(agent, "model", agent.__class__.__name__)
    provider = model.split("/", 1)[0] if "/" in model else None
    return {
        "agent_class": agent.__class__.__name__,
        "model": model,
        "provider": provider,
        "temperature": getattr(agent, "temperature", None),
    }


def _incentive_snapshot(agent: Any) -> Dict[str, Any]:
    assignment = getattr(agent, "incentive_assignment", None)
    if assignment:
        return _jsonable(assignment)
    role = agent.player.identity
    default_objective = (
        "eliminate_crewmates_or_reach_parity"
        if role == "Impostor"
        else "complete_tasks_and_eject_impostors"
    )
    return {
        "schema_version": "amongus.incentive-assignment.v1",
        "plan": "role_default",
        "condition": getattr(agent, "incentive_condition", "role_default"),
        "factorial_cell": {"role": role, "incentive_condition": "role_default"},
        "primary_objective": getattr(agent, "primary_objective", default_objective),
        "secondary_objectives": list(
            getattr(agent, "secondary_objectives", [])
        ),
        "target_player": None,
        "target_role_ground_truth": None,
        "reward_contract": {"primary_weight": 1.0, "secondary_weight": 0.0},
        "private_instruction": None,
    }


class JsonlTurnRecorder:
    """Append schema-versioned turn records to one JSONL file."""

    _path_locks: Dict[Path, threading.Lock] = defaultdict(threading.Lock)

    def __init__(
        self,
        path: str | Path,
        experiment_id: str,
        claim_pipeline: Optional[GroundingPipeline] = None,
    ):
        self.path = Path(path)
        self.experiment_id = experiment_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._turn_sequences: Dict[int, int] = defaultdict(int)
        self.claim_pipeline = claim_pipeline or GroundingPipeline()

    def capture_context(self, env: Any, agent: Any) -> Dict[str, Any]:
        """Capture pre-action state before an agent call can change anything."""
        observation = snapshot_observation(agent.player, env)
        observation["agent_memory"] = {
            "summarization": getattr(agent, "summarization", None),
            "processed_memory": getattr(agent, "processed_memory", None),
            "condensed_memory": getattr(agent, "condensed_memory", None),
        }
        return {
            "captured_at": _utc_now(),
            "phase": env.current_phase,
            "round": _round_index(env),
            "world_state": snapshot_world(env),
            "observation": observation,
            "incentive": _incentive_snapshot(agent),
            "model_metadata": _model_metadata(agent),
        }

    def record_turn(
        self,
        *,
        env: Any,
        agent: Any,
        action: Any,
        context: Mapping[str, Any],
        observation_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        game_index = int(env.game_index)
        turn_index = self._turn_sequences[game_index]
        self._turn_sequences[game_index] += 1
        action_record = serialize_action(action)
        action_name = str(action_record["name"]).upper()
        utterance = action_record.get("message") if action_name == "SPEAK" else None
        vote_target = action_record.get("target_player") if action_name == "VOTE" else None

        record = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "agent_turn",
            "recorded_at": _utc_now(),
            "experiment_id": self.experiment_id,
            "game_index": game_index,
            "turn_index": turn_index,
            "timestep": context["world_state"]["timestep"],
            "phase": context["phase"],
            "round": context["round"],
            "player": {
                "name": agent.player.name,
                "color": agent.player.color,
                "role": agent.player.identity,
                "personality": agent.player.personality,
            },
            "world_state_before": context["world_state"],
            "observation": context["observation"],
            "incentive": context["incentive"],
            "private_belief": context.get(
                "private_belief", {"status": "not_collected", "values": None}
            ),
            "utterance": utterance,
            "atomic_claims": {"status": "pending", "items": []},
            "action": action_record,
            "vote": {"target": vote_target} if vote_target else None,
            "other_agent_beliefs": context.get(
                "other_agent_beliefs",
                {"status": "not_collected", "values": None},
            ),
            "model_metadata": context["model_metadata"],
            "observation_location": observation_location or None,
            "world_state_after": snapshot_world(env),
        }
        record["atomic_claims"] = self.claim_pipeline.annotate_record(record)

        payload = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        with self._path_locks[self.path]:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(payload)
                output.flush()
        return record
