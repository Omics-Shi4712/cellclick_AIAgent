"""CellClick V4.5 independent four-agent Annotation Boost workflow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from cellclick.llm import JsonChatClient, close_json_client
from cellclick.retry import retry_json

from .expression import collect_initial_evidence
from .ledger import observation_entry
from .models import (A1Result, A2Result, A4DecisionResult, A4ObservedSupportResult,
                     A4Result, A4SubtypeEvidenceResult, CandidateRegistry, DiagnosisResult,
                     EvidenceLedgerEntry, EvidenceReviewResult, FinalSummaryResult,
                     QueryPlanResult, TaskInput, ToolObservation)
from .prompts import (A1_SYSTEM, A2_SYSTEM, A3_DIAGNOSIS_SYSTEM, A3_EVIDENCE_REVIEW_SYSTEM,
                      A3_FINAL_SUMMARY_SYSTEM, A3_QUERY_PLAN_SYSTEM, A4_DECISION_SYSTEM,
                      A4_OBSERVED_SUPPORT_SYSTEM, A4_SUBTYPE_EVIDENCE_SYSTEM,
                      PROMPT_VERSION, user)
from .tools import execute_tool, query_fingerprint

PROTOCOL_VERSION = "v4.5"
ANNOTATION_CANDIDATES = 5
MAX_ROUNDS = 5
REQUEST_RETRIES = 2
MAX_PARALLEL_CLUSTERS = 1
MAX_TOOL_REQUESTS = 2
MAX_UNIQUE_QUERY_GENES = 200
MAX_NO_PROGRESS_ROUNDS = 1
WORKFLOW_VERSION = "marker-statistics-four-agent-boost-loop-v4.5-independent-staged-a4-evidence-decision"


def build_graph(**kwargs: Any) -> None:
    del kwargs
    raise NotImplementedError("v4.5 uses annotate_cluster")


def _schema(model: Any) -> dict[str, Any]:
    return model.model_json_schema()


def _marker_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _validate_ids(value: list[str], allowed: set[str], field: str) -> list[str]:
    for item in value:
        if item not in allowed:
            raise ValueError(f"{field} cites unknown evidence id: {item}")
    return value


async def _complete(client: JsonChatClient, *, system: str, payload: dict[str, Any], model: Any,
                    response_model: Any, temperature: float, retries: int,
                    validator: Callable[[dict[str, Any]], Any], strict: bool = False,
                    stage: str | None = None, attempt_log: list[dict[str, Any]] | None = None) -> tuple[Any, str]:
    raw, text = await retry_json(
        client, system=system, user=user(payload), schema=_schema(response_model),
        model=model, temperature=temperature, retries=retries, validator=validator,
        strict=strict, stage=stage, attempt_log=attempt_log,
    )
    return response_model.model_validate(raw), text


def _write_trace(work_dir: Path | None, task_id: str, trace: dict[str, Any]) -> None:
    if work_dir is None:
        return
    path = work_dir / "traces" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def _run(*, task: TaskInput, client: JsonChatClient, model: str, temperature: float, retries: int,
               work_dir: Path | None, expression_file: str | Path, expression_manifest: str | Path) -> dict[str, Any]:
    packet, selection, quality = collect_initial_evidence(
        task, expression_file=expression_file, manifest_file=expression_manifest, work_dir=work_dir,
    )
    evidence_id_by_marker = [f"E_{task.task_id}_{index:03d}" for index, _ in enumerate(task.ordered_markers, 1)]
    evidence_ids = set(evidence_id_by_marker)
    evidence_by_gene = dict(zip(task.ordered_markers, evidence_id_by_marker))
    base = {
        "task_id": task.task_id, "dataset": task.dataset, "species": task.species,
        "tissue": task.tissue, "ordered_markers": task.ordered_markers,
    }

    def validate_a1(value: dict[str, Any]) -> A1Result:
        result = A1Result.model_validate(value)
        if result.task_id != task.task_id or result.selection_id != selection.selection_id:
            raise ValueError("A1 result does not match task or selection")
        _validate_ids(result.evidence_ids, evidence_ids, "A1 evidence_ids")
        return result

    a1, a1_raw = await _complete(
        client, system=A1_SYSTEM, payload={**base, "evidence_packet": packet.model_dump(),
                                           "evidence_id_by_gene": evidence_by_gene,
                                           "selection": selection.model_dump(), "quality": quality.model_dump()},
        model=model, response_model=A1Result, temperature=temperature, retries=retries,
        validator=validate_a1,
    )

    def validate_a2(value: dict[str, Any]) -> A2Result:
        result = A2Result.model_validate(value)
        if result.task_id != task.task_id:
            raise ValueError("A2 result does not match task")
        for candidate in result.candidates[:ANNOTATION_CANDIDATES]:
            _validate_ids(candidate.observed_support, evidence_ids, "candidate observed_support")
            _validate_ids(candidate.subtype_evidence, evidence_ids, "candidate subtype_evidence")
        return result

    a2, a2_raw = await _complete(
        client, system=A2_SYSTEM, payload={**base, "a1_result": a1.model_dump(),
                                           "evidence_packet": packet.model_dump(), "selection": selection.model_dump(),
                                           "evidence_id_by_gene": evidence_by_gene},
        model=model, response_model=A2Result, temperature=temperature, retries=retries,
        validator=validate_a2,
    )
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(a2.candidates[:ANNOTATION_CANDIDATES], 1):
        candidate.hypothesis_id = f"H{index:02d}"
        candidates.append(candidate.model_dump())
    registry = CandidateRegistry(hypotheses=candidates)
    hypothesis_ids = {candidate["hypothesis_id"] for candidate in candidates}
    sibling_comparisons = [
        {"comparison_id": f"CMP_{candidate['hypothesis_id']}",
         "hypothesis_id": candidate["hypothesis_id"], "candidate_label": candidate["label"],
         "competing_label": candidate["competing_label"]}
        for candidate in candidates
    ]
    sibling_comparison_ids = {item["comparison_id"] for item in sibling_comparisons}
    ledger: list[EvidenceLedgerEntry] = []
    observations: list[Any] = []
    a3_results: list[dict[str, Any]] = []
    a3_raw_responses: list[str] = []
    a3_attempts: list[dict[str, Any]] = []
    boost_history: list[dict[str, Any]] = []
    queried_genes: set[str] = set()
    seen_requests: set[str] = set()
    query_history: list[dict[str, Any]] = []
    no_progress_rounds = 0
    stop_reason = "boost_budget_exhausted"

    async def run_phase(*, phase: str, system: str, response_model: Any,
                        payload: dict[str, Any], validator: Callable[[dict[str, Any]], Any]) -> Any:
        phase_attempts: list[dict[str, Any]] = []
        try:
            result, raw = await _complete(
                client, system=system, payload=payload, model=model, response_model=response_model,
                temperature=temperature, retries=retries, validator=validator, strict=True,
                stage=phase, attempt_log=phase_attempts,
            )
        except Exception as error:
            a3_attempts.extend(phase_attempts)
            _write_trace(work_dir, task.task_id, {
                **base, "protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION,
                "workflow_version": WORKFLOW_VERSION, "failure": {
                    "phase": phase, "error_type": type(error).__name__, "error": str(error),
                }, "a3_results": a3_results, "attempts": a3_attempts,
                "raw_responses": a3_raw_responses,
            })
            raise
        a3_attempts.extend(phase_attempts)
        a3_raw_responses.append(raw)
        record = {"phase": phase, **result.model_dump()}
        a3_results.append(record)
        return result

    def validate_diagnosis(value: dict[str, Any]) -> DiagnosisResult:
        result = DiagnosisResult.model_validate(value)
        if result.task_id != task.task_id:
            raise ValueError(f"diagnosis task_id mismatch: expected {task.task_id!r}, got {result.task_id!r}")
        for assessment in result.candidate_assessments:
            hypothesis_id = assessment.get("hypothesis_id") if isinstance(assessment, dict) else None
            if hypothesis_id is not None and hypothesis_id not in hypothesis_ids:
                raise ValueError("diagnosis assessment cites an unknown hypothesis")
        return result

    diagnosis = await run_phase(
        phase="diagnosis", system=A3_DIAGNOSIS_SYSTEM, response_model=DiagnosisResult,
        payload={**base, "current_stage": "diagnosis", "a1_result": a1.model_dump(),
                 "a2_result": {**a2.model_dump(), "candidates": candidates},
                 "evidence_packet": packet.model_dump(), "candidate_registry": registry.model_dump(),
                 "evidence_ids": sorted(evidence_ids)}, validator=validate_diagnosis,
    )
    boost_history.append({"cycle": 0, "stage": "diagnosis", "next_action": diagnosis.next_action,
                          "candidate_assessments": [item.model_dump() for item in diagnosis.candidate_assessments],
                          "unresolved_questions": diagnosis.unresolved_questions,
                          "limitations": diagnosis.limitations,
                          "summary": diagnosis.summary})
    final_a3: EvidenceReviewResult | FinalSummaryResult | None = None
    final_summary: FinalSummaryResult | None = None
    completed_cycles = 0
    for cycle in range(1, MAX_ROUNDS + 1):
        def validate_query_plan(value: dict[str, Any]) -> QueryPlanResult:
            result = QueryPlanResult.model_validate(value)
            if result.task_id != task.task_id:
                raise ValueError(f"query_plan task_id mismatch: expected {task.task_id!r}, got {result.task_id!r}")
            for request in result.tool_requests:
                if request.task_id != task.task_id:
                    raise ValueError("query_plan tool request does not match task")
                _validate_ids(request.hypothesis_ids, hypothesis_ids, "query_plan hypothesis_ids")
                if hypothesis_ids and not request.hypothesis_ids:
                    raise ValueError("query_plan tool requests require a named hypothesis")
            return result

        plan = await run_phase(
            phase="query_plan", system=A3_QUERY_PLAN_SYSTEM, response_model=QueryPlanResult,
            payload={**base, "current_stage": "query_plan", "boost_cycle": cycle,
                     "diagnosis": diagnosis.model_dump(), "a1_result": a1.model_dump(),
                     "a2_result": {**a2.model_dump(), "candidates": candidates},
                     "candidate_registry": registry.model_dump(), "sibling_comparisons": sibling_comparisons,
                     "evidence_ledger": [item.model_dump() for item in ledger],
                     "evidence_ids": sorted(evidence_ids), "boost_history": boost_history,
                     "executed_queries": query_history},
            validator=validate_query_plan,
        )
        if len(plan.tool_requests) > MAX_TOOL_REQUESTS:
            raise ValueError("query_plan tool request budget exceeded")
        cycle_observations = []
        duplicate_requests = 0
        for request in plan.tool_requests:
            request_hash = query_fingerprint(request)
            if request_hash in seen_requests:
                duplicate_requests += 1
                observation = ToolObservation(
                    observation_id=f"OBS_DUPLICATE_{cycle}_{len(cycle_observations) + 1}",
                    request_id=request.request_id, task_id=task.task_id,
                    tool_name=request.tool_name, status="protocol_error",
                    parameter_hash=request_hash,
                    limitations=["duplicate_request_skipped"],
                )
                observations.append(observation)
                cycle_observations.append(observation)
                ledger.append(observation_entry(observation))
                continue
            seen_requests.add(request_hash)
            query_history.append({"tool_name": request.tool_name,
                                  "comparison": request.arguments.comparison,
                                  "genes": list(request.arguments.genes),
                                  "parameter_hash": request_hash})
            genes = {_marker_key(gene) for gene in request.arguments.genes}
            if len(queried_genes | genes) > MAX_UNIQUE_QUERY_GENES:
                raise ValueError("A3 unique gene query budget exceeded")
            queried_genes.update(genes)
            observation = execute_tool(task, request, expression_file=expression_file,
                                       expression_manifest=expression_manifest)
            observations.append(observation)
            cycle_observations.append(observation)
            ledger.append(observation_entry(observation))
            if observation.status == "success":
                evidence_ids.add(f"EV_{observation.observation_id}")
        boost_history.append({"cycle": cycle, "stage": "query_plan", "next_action": plan.next_action,
                              "tool_requests": [item.model_dump() for item in plan.tool_requests],
                              "unresolved_questions": plan.unresolved_questions,
                              "limitations": plan.limitations,
                              "summary": plan.summary})

        def validate_review(value: dict[str, Any]) -> EvidenceReviewResult:
            result = EvidenceReviewResult.model_validate(value)
            if result.task_id != task.task_id:
                raise ValueError(f"evidence_review task_id mismatch: expected {task.task_id!r}, got {result.task_id!r}")
            for assessment in result.candidate_assessments:
                hypothesis_id = assessment.get("hypothesis_id") if isinstance(assessment, dict) else None
                if hypothesis_id is not None and hypothesis_id not in hypothesis_ids:
                    raise ValueError("evidence_review assessment cites an unknown hypothesis")
            return result

        review = await run_phase(
            phase="evidence_review", system=A3_EVIDENCE_REVIEW_SYSTEM, response_model=EvidenceReviewResult,
            payload={**base, "current_stage": "evidence_review", "boost_cycle": cycle,
                     "diagnosis": diagnosis.model_dump(), "query_plan": plan.model_dump(),
                     "candidate_registry": registry.model_dump(), "sibling_comparisons": sibling_comparisons,
                     "evidence_ledger": [item.model_dump() for item in ledger],
                     "tool_observations": [item.model_dump() for item in observations],
                     "evidence_ids": sorted(evidence_ids), "boost_history": boost_history,
                     "executed_queries": query_history},
            validator=validate_review,
        )
        completed_cycles = cycle
        if duplicate_requests == len(plan.tool_requests):
            no_progress_rounds += 1
        else:
            no_progress_rounds = 0
        successful_observations = sum(1 for item in observations if item.status == "success")
        finish_allowed = bool(
            review.next_action == "finish" and successful_observations > 0
            and review.candidate_assessments
        )
        boost_history.append({"cycle": cycle, "stage": "evidence_review", "next_action": review.next_action,
                              "candidate_assessments": [item.model_dump() for item in review.candidate_assessments],
                              "unresolved_questions": review.unresolved_questions,
                              "limitations": review.limitations,
                              "observation_ids": [item.observation_id for item in cycle_observations],
                              "stop_reason": review.stop_reason, "finish_allowed": finish_allowed,
                              "summary": review.summary})
        if finish_allowed:
            final_a3 = review
            stop_reason = review.stop_reason
            break
        if no_progress_rounds >= MAX_NO_PROGRESS_ROUNDS:
            stop_reason = "no_progress_due_to_duplicate_queries"
            break

    if final_a3 is None:
        def validate_final_summary(value: dict[str, Any]) -> FinalSummaryResult:
            result = FinalSummaryResult.model_validate(value)
            if result.task_id != task.task_id:
                raise ValueError(f"final_summary task_id mismatch: expected {task.task_id!r}, got {result.task_id!r}")
            if result.recommended_hypothesis_id is not None and result.recommended_hypothesis_id not in hypothesis_ids:
                raise ValueError("final_summary recommends an unknown hypothesis")
            for assessment in result.candidate_assessments:
                hypothesis_id = assessment.hypothesis_id
                if hypothesis_id not in hypothesis_ids:
                    raise ValueError("final_summary assessment cites an unknown hypothesis")
            return result

        final_summary = await run_phase(
            phase="final_summary", system=A3_FINAL_SUMMARY_SYSTEM, response_model=FinalSummaryResult,
            payload={**base, "current_stage": "final_summary", "diagnosis": diagnosis.model_dump(),
                     "a1_result": a1.model_dump(), "a2_result": {**a2.model_dump(), "candidates": candidates},
                     "candidate_registry": registry.model_dump(), "sibling_comparisons": sibling_comparisons,
                     "evidence_ledger": [item.model_dump() for item in ledger],
                     "tool_observations": [item.model_dump() for item in observations],
                     "evidence_ids": sorted(evidence_ids), "boost_history": boost_history,
                     "executed_queries": query_history, "stop_reason": stop_reason},
            validator=validate_final_summary,
        )
        final_a3 = final_summary

    common_record = {"a1": a1.model_dump(), "a2": {**a2.model_dump(), "candidates": candidates},
                     "a3": final_a3.model_dump() if final_a3 is not None else None,
                     "a3_final_summary": final_summary.model_dump() if final_summary is not None else None,
                     "a3_results": a3_results, "candidate_registry": registry.model_dump(),
                     "evidence_ledger": [item.model_dump() for item in ledger],
                     "tool_observations": [item.model_dump() for item in observations],
                     "sibling_comparisons": sibling_comparisons, "boost_history": boost_history}
    a4_attempts: list[dict[str, Any]] = []

    async def run_a4_phase(*, phase: str, system: str, response_model: Any,
                           payload: dict[str, Any], validator: Callable[[dict[str, Any]], Any]) -> tuple[Any, str]:
        phase_attempts: list[dict[str, Any]] = []
        try:
            result, raw = await _complete(
                client, system=system, payload=payload, model=model, response_model=response_model,
                temperature=temperature, retries=retries, validator=validator, strict=True,
                stage=phase, attempt_log=phase_attempts,
            )
        except Exception as error:
            a4_attempts.extend(phase_attempts)
            _write_trace(work_dir, task.task_id, {
                **base, "protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION,
                "workflow_version": WORKFLOW_VERSION, "failure": {
                    "phase": phase, "error_type": type(error).__name__, "error": str(error),
                }, "a3_results": a3_results, "a4_attempts": a4_attempts,
            })
            raise
        a4_attempts.extend(phase_attempts)
        return result, raw

    def validate_observed_support(value: dict[str, Any]) -> A4ObservedSupportResult:
        result = A4ObservedSupportResult.model_validate(value)
        if result.task_id != task.task_id:
            raise ValueError("A4 observed-support result does not match task")
        _validate_ids(result.observed_support_ids, evidence_ids, "A4 observed_support_ids")
        return result

    a4_observed, a4_observed_raw = await run_a4_phase(
        phase="a4_observed_support", system=A4_OBSERVED_SUPPORT_SYSTEM,
        response_model=A4ObservedSupportResult,
        payload={**base, "current_stage": "a4_observed_support", "a1_result": a1.model_dump(),
                 "a2_result": {**a2.model_dump(), "candidates": candidates},
                 "a3_result": final_a3.model_dump(), "a3_final_summary":
                 final_summary.model_dump() if final_summary is not None else None,
                 "a3_results": a3_results, "candidate_registry": registry.model_dump(),
                 "evidence_ledger": [item.model_dump() for item in ledger],
                 "tool_observations": [item.model_dump() for item in observations],
                 "evidence_ids": sorted(evidence_ids)}, validator=validate_observed_support,
    )

    allowed_subtype_ids = set(a4_observed.observed_support_ids)

    def validate_subtype_evidence(value: dict[str, Any]) -> A4SubtypeEvidenceResult:
        result = A4SubtypeEvidenceResult.model_validate(value)
        if result.task_id != task.task_id:
            raise ValueError("A4 subtype-evidence result does not match task")
        _validate_ids(result.subtype_evidence_ids, allowed_subtype_ids, "A4 subtype_evidence_ids")
        return result

    a4_subtype, a4_subtype_raw = await run_a4_phase(
        phase="a4_subtype_evidence", system=A4_SUBTYPE_EVIDENCE_SYSTEM,
        response_model=A4SubtypeEvidenceResult,
        payload={**base, "current_stage": "a4_subtype_evidence",
                 "allowed_observed_support_ids": a4_observed.observed_support_ids,
                 "candidate_registry": registry.model_dump(), "a3_result": final_a3.model_dump(),
                 "a3_final_summary": final_summary.model_dump() if final_summary is not None else None},
        validator=validate_subtype_evidence,
    )

    accepted_labels = {str(candidate.get("label")) for candidate in candidates} | {
        str(candidate.get("parent_label")) for candidate in candidates
    }
    subtype_labels = {str(candidate.get("label")) for candidate in candidates
                      if candidate.get("granularity") == "subtype"}

    def validate_decision(value: dict[str, Any]) -> A4DecisionResult:
        result = A4DecisionResult.model_validate(value)
        if result.task_id != task.task_id:
            raise ValueError("A4 final-decision result does not match task")
        _validate_ids(result.sibling_comparison_ids, sibling_comparison_ids, "A4 sibling_comparison_ids")
        if result.label not in accepted_labels:
            raise ValueError("A4 label is not present in the candidate registry")
        if (result.granularity == "subtype" or result.label in subtype_labels) and not a4_subtype.subtype_evidence_ids:
            raise ValueError("A4 subtype label requires frozen subtype evidence")
        if result.action == "accepted" and result.granularity == "subtype" and not result.sibling_comparison_ids:
            raise ValueError("accepted subtype requires sibling comparison")
        return result

    a4_decision, a4_decision_raw = await run_a4_phase(
        phase="a4_decision", system=A4_DECISION_SYSTEM, response_model=A4DecisionResult,
        payload={**base, "current_stage": "a4_decision", "a1_result": a1.model_dump(),
                 "a2_result": {**a2.model_dump(), "candidates": candidates},
                 "a3_result": final_a3.model_dump(), "a3_final_summary":
                 final_summary.model_dump() if final_summary is not None else None,
                 "candidate_registry": registry.model_dump(), "sibling_comparisons": sibling_comparisons,
                 "frozen_observed_support_ids": a4_observed.observed_support_ids,
                 "frozen_subtype_evidence_ids": a4_subtype.subtype_evidence_ids},
        validator=validate_decision,
    )
    a4 = A4Result(**a4_decision.model_dump(), observed_support_ids=a4_observed.observed_support_ids,
                  subtype_evidence_ids=a4_subtype.subtype_evidence_ids)
    decision_history = [{"stage": "a2", "status": "candidate_generated",
                         "labels": [candidate.get("label") for candidate in candidates]}
                        ]
    for item in a3_results:
        labels = [assessment.get("hypothesis_id") for assessment in item.get("candidate_assessments", [])]
        statuses = [assessment.get("status") for assessment in item.get("candidate_assessments", [])]
        decision_history.append({"stage": item.get("phase"), "status": statuses or ["reviewed"],
                                 "hypotheses": labels})
    decision_history.append({"stage": "a4", "status": a4.decision_status, "label": a4.label,
                             "confidence": a4.confidence, "action": a4.action,
                             "unresolved_conflicts": list(a4.unresolved_conflicts),
                             "rationale": a4.rationale})
    record = {"status": a4.decision_status,
              "prediction": a4.label,
              "decision_status": a4.decision_status,
              "decision_history": decision_history,
              "stop_reason": stop_reason, **common_record,
              "a3": final_a3.model_dump(), "a4": a4.model_dump(),
                              "call_counts": {"a1": 1, "a2": 1, "a3": len(a3_results), "a4": 3,
                              "total": 5 + len(a3_results)}}
    _write_trace(work_dir, task.task_id, {**base, "protocol_version": PROTOCOL_VERSION,
                                           "prompt_version": PROMPT_VERSION, "workflow_version": WORKFLOW_VERSION,
                                           "evidence_packet": packet.model_dump(), "selection": selection.model_dump(),
                                           "quality": quality.model_dump(), "final_record": record,
                                           "raw_responses": {"a1": a1_raw, "a2": a2_raw,
                                                             "a3": a3_raw_responses,
                                                             "a4": {"observed_support": a4_observed_raw,
                                                                    "subtype_evidence": a4_subtype_raw,
                                                                    "decision": a4_decision_raw}},
                                           "attempts": {"a3": a3_attempts, "a4": a4_attempts}})
    return {"status": record["status"], "round_count": completed_cycles, "final_record": record}


def annotate_cluster(*, cluster_id: str, dataset: str, species: str, tissue: str, markers: list[str],
                     client: JsonChatClient, model: str, temperature: float, retries: int,
                     work_dir: Path | None = None, prompt_version: str = PROMPT_VERSION,
                     protocol_version: str = PROTOCOL_VERSION, expression_file: str | Path | None = None,
                     expression_manifest: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    del kwargs
    if prompt_version != PROMPT_VERSION or protocol_version != PROTOCOL_VERSION:
        raise ValueError("CellClick v4.5 requires protocol_version=v4.5 and prompt_version=v4.5")
    if not expression_file or not expression_manifest:
        raise ValueError("CellClick V4 requires expression_file and expression_manifest")
    task = TaskInput(task_id=cluster_id, cluster_id=cluster_id, dataset=dataset, species=species,
                     tissue=tissue, ordered_markers=markers)

    async def run() -> dict[str, Any]:
        try:
            return await _run(task=task, client=client, model=model, temperature=temperature, retries=retries,
                              work_dir=work_dir, expression_file=expression_file,
                              expression_manifest=expression_manifest)
        finally:
            await close_json_client(client)

    return asyncio.run(run())

