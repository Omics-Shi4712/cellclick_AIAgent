"""Append-only evidence and target bookkeeping for V5."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .models import EvidenceLedgerEntry, ToolObservation, VerificationTarget


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def make_target(*, target_id: str, task_id: str, source_target_id: str, cell_barcodes: Iterable[str],
                supported_parent_ids: Iterable[str] = (), reason: str) -> VerificationTarget:
    cells = list(dict.fromkeys(str(x) for x in cell_barcodes))
    return VerificationTarget(target_id=target_id, task_id=task_id, source_target_id=source_target_id,
                              cell_barcodes=cells, cell_set_fingerprint=fingerprint(cells),
                              supported_parent_ids=list(supported_parent_ids), reason=reason)


def observation_entry(observation: ToolObservation, *, evidence_id: str | None = None) -> EvidenceLedgerEntry:
    eid = evidence_id or f"EV_{observation.observation_id}"
    kind = "observed" if observation.status == "success" else "tool_error"
    return EvidenceLedgerEntry(evidence_id=eid, task_id=observation.task_id, target_id=observation.target_id,
                               kind=kind, source=observation.tool_name,
                               observation_id=observation.observation_id,
                               payload=observation.result or {"status": observation.status},
                               usable=observation.status == "success")


def derived_entry(*, evidence_id: str, task_id: str, target_id: str, source: str,
                  payload: dict[str, Any], parent_evidence_ids: Iterable[str]) -> EvidenceLedgerEntry:
    return EvidenceLedgerEntry(evidence_id=evidence_id, task_id=task_id, target_id=target_id, kind="derived",
                               source=source, payload=payload, parent_evidence_ids=list(parent_evidence_ids))


def snapshot_id(entries: Iterable[EvidenceLedgerEntry]) -> str:
    return "ES_" + fingerprint([entry.model_dump(mode="json") for entry in entries])[:20]
