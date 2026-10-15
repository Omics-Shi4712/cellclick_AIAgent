"""Program-controlled verification helpers shared by V5 graph implementations."""
from __future__ import annotations

from typing import Iterable

from .ledger import observation_entry, snapshot_id
from .models import EvidenceLedgerEntry, ToolObservation, ToolRequest, V5State, VerificationReport
from .tools import ToolRegistry


def dispatch_requests(state: V5State, requests: Iterable[ToolRequest], registry: ToolRegistry,
                      context: dict) -> list[ToolObservation]:
    observations: list[ToolObservation] = []
    for request in requests:
        question = next((item for item in state.questions if item.question_id == request.question_id), None)
        if question is None or question.status not in {"open", "answered"} or request.target_id not in {target.target_id for target in state.verification_targets}:
            observations.append(ToolObservation(
                observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                task_id=state.task.task_id, target_id=request.target_id, tool_name=request.tool_name,
                status="protocol_error", parameter_hash="question_or_target_invalid",
                limitations=["tool_request_requires_open_question_and_registered_target"],
            ))
            continue
        if request.task_id != state.task.task_id:
            observations.append(ToolObservation(
                observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                task_id=state.task.task_id, target_id=request.target_id, tool_name=request.tool_name,
                status="protocol_error", parameter_hash="task_id_mismatch",
                limitations=["task_id_mismatch"],
            ))
            continue
        if state.budget.get("tool_requests", 0) <= 0:
            observations.append(ToolObservation(
                observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                task_id=request.task_id, target_id=request.target_id, tool_name=request.tool_name,
                status="protocol_error", parameter_hash="budget_exhausted",
                limitations=["tool_request_budget_exhausted"],
            ))
            continue
        state.budget["tool_requests"] -= 1
        observation = registry.dispatch(request, context)
        observations.append(observation)
        state.evidence_store.append(observation_entry(observation))
        question.status = "answered" if observation.status == "success" else "unanswerable"
        question.evidence_ids.append(state.evidence_store[-1].evidence_id)
    return observations


def current_evidence_snapshot(state: V5State) -> str:
    return snapshot_id(state.evidence_store)


def report(*, state: V5State, report_id: str, target_id: str, scope: str,
           candidate_status: dict[str, str], supporting: Iterable[str] = (),
           conflicting: Iterable[str] = (), revision_requests: list[dict] | None = None,
           unresolved: Iterable[str] = (), stop_reason: str) -> VerificationReport:
    value = VerificationReport(report_id=report_id, target_id=target_id, scope=scope,
                               candidate_status=candidate_status,
                               supporting_evidence_ids=list(supporting),
                               conflicting_evidence_ids=list(conflicting),
                               revision_requests=revision_requests or [],
                               unresolved_questions=list(unresolved), stop_reason=stop_reason)
    state.verification_reports.append(value)
    return value
