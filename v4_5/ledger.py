"""Append-only evidence ledger helpers."""

from __future__ import annotations

from .models import EvidenceLedgerEntry, ToolObservation


def observation_entry(observation: ToolObservation) -> EvidenceLedgerEntry:
    kind = "observed" if observation.status == "success" else "tool_error"
    payload = observation.result.model_dump() if observation.result is not None else {
        "status": observation.status, "limitations": observation.limitations,
    }
    return EvidenceLedgerEntry(
        evidence_id=f"EV_{observation.observation_id}", task_id=observation.task_id,
        kind=kind, source=observation.tool_name, observation_id=observation.observation_id,
        payload=payload,
    )

