"""Read-only quality checks for in-progress human-validation CSV packets."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .packets import REVIEW_FIELDS


CLAIM_TYPES = {
    "event",
    "location",
    "role",
    "assessment",
    "observation",
    "status",
    "intent",
    "other",
}
TAXONOMY_CATEGORIES = {
    "truth",
    "unsupported_claim",
    "mistaken_belief",
    "fabrication",
    "omission",
    "equivocation",
    "misdirection",
    "pragmatic_distortion",
    "strategic_falsehood",
    "unresolved",
}
CHOICES = {
    "claims": {
        "human_claim_correct": {"yes", "no", "unclear"},
        "human_claim_type": CLAIM_TYPES,
        "human_objective_truth": {"true", "false", "unknown"},
        "human_speaker_support": {
            "supported",
            "contradicted",
            "unsupported",
            "unknown",
        },
    },
    "taxonomy": {
        "human_primary": TAXONOMY_CATEGORIES,
        "human_intent_evidence": {"yes", "no", "unclear"},
        "human_confidence": {"1", "2", "3", "4", "5"},
    },
    "control": {
        "human_should_intervene": {"yes", "no", "unclear"},
        "human_rewrite_preserves_meaning": {"1", "2", "3", "4", "5"},
        "human_rewrite_supported": {"yes", "no", "unclear"},
        "human_rewrite_communicative": {"yes", "no", "unclear"},
    },
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _note_required(packet_type: str, row: Mapping[str, str]) -> bool:
    if packet_type == "claims":
        return (
            str(row.get("human_claim_correct", "")).strip().lower()
            in {"no", "unclear"}
            or str(row.get("human_claim_type", "")).strip().lower() == "other"
            or str(row.get("human_objective_truth", "")).strip().lower()
            == "unknown"
            or str(row.get("human_speaker_support", "")).strip().lower()
            == "unknown"
        )
    if packet_type == "taxonomy":
        return str(row.get("human_primary", "")).strip().lower() in {
            "strategic_falsehood",
            "mistaken_belief",
            "omission",
            "unresolved",
        }
    if packet_type == "control":
        return (
            str(row.get("human_rewrite_preserves_meaning", "")).strip() in {"1", "2"}
            or str(row.get("human_rewrite_supported", "")).strip().lower() == "no"
        )
    return False


def audit_validation_packet(
    path: Path,
    *,
    expected_annotator: str | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Audit one blinded packet without loading an automatic-label key."""
    fields, rows = _read_csv(path)
    packet_types = {
        str(row.get("packet_type", "")).strip().lower()
        for row in rows
        if str(row.get("packet_type", "")).strip()
    }
    packet_type = next(iter(packet_types)) if len(packet_types) == 1 else None
    issues: list[dict[str, Any]] = []

    def issue(
        row_number: int | None,
        item_id: str,
        field: str,
        kind: str,
        value: str,
        message: str,
    ) -> None:
        issues.append(
            {
                "row_number": row_number,
                "item_id": item_id,
                "field": field,
                "kind": kind,
                "value": value,
                "message": message,
            }
        )

    if not rows:
        issue(None, "", "", "empty_packet", "", "Packet contains no data rows.")
    if packet_type not in REVIEW_FIELDS:
        issue(
            None,
            "",
            "packet_type",
            "invalid_packet_type",
            "|".join(sorted(packet_types)),
            "Packet must contain exactly one recognized packet_type.",
        )
        required_fields: Sequence[str] = ()
    else:
        required_fields = REVIEW_FIELDS[packet_type]
        for field in ("packet_type", "item_id", *required_fields, "human_notes", "annotator_id"):
            if field not in fields:
                issue(
                    None,
                    "",
                    field,
                    "missing_column",
                    "",
                    f"Required column {field!r} is missing.",
                )

    item_counts = Counter(str(row.get("item_id", "")).strip() for row in rows)
    for item_id, count in sorted(item_counts.items()):
        if not item_id:
            issue(None, "", "item_id", "missing_item_id", "", "A row has no item_id.")
        elif count > 1:
            issue(
                None,
                item_id,
                "item_id",
                "duplicate_item_id",
                item_id,
                f"item_id appears {count} times.",
            )

    field_completion = {field: 0 for field in required_fields}
    fully_completed = 0
    missing_note_count = 0
    for row_number, row in enumerate(rows, start=2):
        item_id = str(row.get("item_id", "")).strip()
        completed = True
        for field in required_fields:
            value = str(row.get(field, "")).strip().lower()
            if value:
                field_completion[field] += 1
                allowed = CHOICES.get(packet_type or "", {}).get(field, set())
                if value not in allowed:
                    issue(
                        row_number,
                        item_id,
                        field,
                        "invalid_choice",
                        value,
                        f"Use one of: {', '.join(sorted(allowed))}.",
                    )
            else:
                completed = False
                if require_complete:
                    issue(
                        row_number,
                        item_id,
                        field,
                        "missing_value",
                        "",
                        "Required review field is blank.",
                    )
        if completed:
            fully_completed += 1

        if packet_type == "taxonomy":
            secondary = str(row.get("human_secondary", "")).strip().lower()
            if secondary:
                invalid = sorted(set(secondary.split("|")) - TAXONOMY_CATEGORIES)
                if invalid:
                    issue(
                        row_number,
                        item_id,
                        "human_secondary",
                        "invalid_choice",
                        secondary,
                        "Unknown secondary categories: " + ", ".join(invalid),
                    )

        if packet_type and _note_required(packet_type, row):
            if not str(row.get("human_notes", "")).strip():
                missing_note_count += 1
                issue(
                    row_number,
                    item_id,
                    "human_notes",
                    "missing_required_note",
                    "",
                    "This choice requires a short explanation.",
                )

        annotator = str(row.get("annotator_id", "")).strip()
        if expected_annotator and annotator.casefold() != expected_annotator.casefold():
            issue(
                row_number,
                item_id,
                "annotator_id",
                "unexpected_annotator",
                annotator,
                f"Expected annotator_id {expected_annotator!r}.",
            )

    invalid_value_count = sum(
        item["kind"] in {"invalid_choice", "invalid_packet_type"} for item in issues
    )
    structural_issue_count = sum(
        item["kind"]
        in {"empty_packet", "missing_column", "missing_item_id", "duplicate_item_id"}
        for item in issues
    )
    ready = bool(
        rows
        and packet_type in REVIEW_FIELDS
        and fully_completed == len(rows)
        and invalid_value_count == 0
        and structural_issue_count == 0
        and missing_note_count == 0
        and not any(item["kind"] == "unexpected_annotator" for item in issues)
    )
    return {
        "schema_version": "amongus.human-validation-packet-audit.v1",
        "packet": str(path),
        "packet_type": packet_type,
        "item_count": len(rows),
        "fully_completed_count": fully_completed,
        "remaining_count": len(rows) - fully_completed,
        "field_completion": field_completion,
        "invalid_value_count": invalid_value_count,
        "missing_required_note_count": missing_note_count,
        "structural_issue_count": structural_issue_count,
        "ready_for_independent_agreement": ready,
        "issues": issues,
    }
