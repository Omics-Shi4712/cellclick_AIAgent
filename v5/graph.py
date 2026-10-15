"""Single program-controlled V5.1 execution graph."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from cellclick.llm import JsonChatClient, close_json_client
from cellclick.retry import retry_json
from .expression import cluster_expression_evidence
from .ledger import make_target, snapshot_id
from .models import (AdjudicationResponse, ArgumentDraftResponse, CandidateDraft, CandidateDraftResponse, CandidateProposal,
    CandidateRegistry, EvidenceLedgerEntry, FinalAnnotation, MarkerAudit, MessageEnvelope,
    ReferenceComparisonResult, TaskInput, ToolRequest, ToulminArgument, V5State, VerificationQuestion,
    VerifierResponse)
from .prompts import ADJUDICATOR_SYSTEM, MAJOR_TYPE_SYSTEM, MARKER_AUDIT_SYSTEM, PROMPT_VERSION, SUBTYPE_SYSTEM, VERIFIER_SYSTEM
from .tools import build_tool_registry
from .verifier import dispatch_requests, report

PROTOCOL_VERSION = "v5"
ANNOTATION_CANDIDATES, MAX_ROUNDS, REQUEST_RETRIES, MAX_PARALLEL_CLUSTERS = 5, 4, 2, 1
WORKFLOW_VERSION = "v5.3-reference-evidence-auto-recheck-audited-fallback"

# Labels that describe an expression program or technical state rather than
# an identifiable cell type.  They are filtered before entering the registry
# so later verifier/adjudicator stages cannot select them as identities.
_NON_CELL_TYPE_TERMS = (
    "ribosomal", "ribosome", "housekeeping", "mitochondrial", "stress",
    "interferon response", "interferon-responsive", "cycling", "cell cycle",
    "low quality", "low-quality", "doublet", "ambient", "cycling cells",
)


def _is_biological_cell_type(label: str) -> bool:
    normalized = " ".join(label.casefold().strip().split())
    return bool(normalized) and not any(term in normalized for term in _NON_CELL_TYPE_TERMS)


def _initial_evidence(task: TaskInput, expression: dict[str, Any]) -> list[EvidenceLedgerEntry]:
    result = [EvidenceLedgerEntry(evidence_id=f"E_MARKER_{i:03d}", task_id=task.task_id, target_id=task.cluster_id,
        kind="gene", source="submitted_marker", payload={"gene": gene}) for i, gene in enumerate(task.ordered_markers, 1)]
    result.extend(EvidenceLedgerEntry(evidence_id=f"E_EXPR_{i:03d}", task_id=task.task_id, target_id=task.cluster_id,
        kind="derived", source="expression_summary", payload=row) for i, row in enumerate(expression.get("top_differential_genes", []), 1))
    return result


def _reference_evidence(task: TaskInput, packet: dict[str, Any]) -> EvidenceLedgerEntry | None:
    hits = packet.get("hits") if isinstance(packet, dict) else None
    if not isinstance(hits, list) or not hits:
        return None
    bounded_hits = []
    for hit in hits:
        if not isinstance(hit, dict) or not isinstance(hit.get("label"), str):
            continue
        bounded_hits.append({key: hit.get(key) for key in (
            "rank", "label", "cl_id", "score", "gene_score",
            "matched_input_markers", "matched_gene_count", "matched_query_fraction",
        )})
    if not bounded_hits:
        return None
    return EvidenceLedgerEntry(
        evidence_id="E_REFERENCE_CELLMARKER_001", task_id=task.task_id, target_id=task.cluster_id,
        kind="reference", source="initial_cellmarker_retrieval",
        payload={
            "retrieval_version": packet.get("retrieval_version"),
            "reference_sha256": packet.get("reference_sha256"),
            "query_tissue": packet.get("query_tissue"),
            "hits": bounded_hits,
        },
    )


def _register(drafts: CandidateDraftResponse, *, source: str, version: int, parent: CandidateProposal | None = None,
              trusted_labels: list[str] | None = None) -> CandidateRegistry:
    start = 20 if parent else 1
    proposals = [d for d in drafts.candidates if _is_biological_cell_type(d.label)]
    # Include reliable reference hits even when the LLM omitted them from its
    # initial shortlist.  This prevents a real supported type (e.g. epithelial
    # cell) from becoming unverifiable merely because candidate generation
    # focused on an incorrect lineage.
    existing = {d.label.casefold().strip() for d in proposals}
    trusted_proposals: list[CandidateDraft] = []
    for label in trusted_labels or []:
        if _is_biological_cell_type(label) and label.casefold().strip() not in existing:
            trusted_proposals.append(CandidateDraft(label=label.strip(), granularity="major"))
            existing.add(label.casefold().strip())
    proposals = trusted_proposals + proposals
    # Preserve the parent even if a provider repeats its label verbatim; only
    # newly proposed children are filtered for non-cell-type descriptions.
    # Scope is owned by the workflow. Do not let a major-stage model response
    # hide candidates from verification by misclassifying them as subtypes.
    stage_granularity = "subtype" if parent else "major"
    values = [CandidateProposal(candidate_id=f"H{i:02d}", label=d.label.strip(), granularity=stage_granularity,
        parent_id=parent.candidate_id if parent else None, provisional=True, source=source)
        for i, d in enumerate(proposals[:ANNOTATION_CANDIDATES], start)]
    if parent: values.insert(0, parent)
    ids = [v.candidate_id for v in values]
    return CandidateRegistry(version=version, candidates=values, parent_by_id={v.candidate_id: v.parent_id for v in values},
        competing_by_id={v.candidate_id: [x for x in ids if x not in {v.candidate_id, v.parent_id}] for v in values})


async def _complete(client: JsonChatClient, state: V5State, *, system: str, recipient: str, stage: str,
                    payload: dict[str, Any], model: str, temperature: float, retries: int, response_type: Any,
                    semantic_validator: Any | None = None) -> Any:
    envelope = MessageEnvelope(message_id=f"M_{len(state.messages)+1:04d}", task_id=state.task.task_id,
        target_id=state.task.cluster_id, sender="program", recipient=recipient, message_type=stage,
        candidate_registry_version=state.candidate_registry.version, evidence_snapshot_id=snapshot_id(state.evidence_store), payload=payload)
    state.messages.append(envelope)
    def validate(value: dict[str, Any]) -> Any:
        parsed = response_type.model_validate(value)
        if semantic_validator is not None:
            semantic_validator(parsed)
        return parsed
    raw, _ = await retry_json(client, system=system, user=envelope.model_dump_json(), schema=response_type.model_json_schema(),
        model=model, temperature=temperature, retries=retries, validator=validate, strict=True, stage=stage)
    state.transition_history.append({"stage": stage, "message_id": envelope.message_id, "evidence_snapshot_id": envelope.evidence_snapshot_id})
    return validate(raw)


def _status_map(response: VerifierResponse) -> dict[str, str]:
    return {item.candidate_id: item.status for item in response.candidate_status}


def _normalize_argument_drafts(drafts: ArgumentDraftResponse, state: V5State, *, allowed: set[str]) -> ArgumentDraftResponse:
    """Bind model-proposed arguments to program-owned candidate/evidence IDs."""
    candidates = {v.candidate_id for v in state.candidate_registry.candidates}
    evidence = {v.evidence_id for v in state.evidence_store if v.usable}
    normalized = []
    for draft in drafts.arguments:
        if draft.hypothesis_id not in candidates or draft.hypothesis_id not in allowed:
            continue
        bound_evidence = list(dict.fromkeys(item for item in draft.grounds_evidence_ids if item in evidence))
        if not bound_evidence:
            continue
        normalized.append(draft.model_copy(update={"grounds_evidence_ids": bound_evidence}))
    return ArgumentDraftResponse(arguments=normalized)


def _normalize_verifier_response(response: VerifierResponse, *, allowed: set[str], evidence: set[str]) -> VerifierResponse:
    """Convert a verifier suggestion into a complete bounded program record."""
    statuses = {}
    for item in response.candidate_status:
        if item.candidate_id in allowed and item.candidate_id not in statuses:
            statuses[item.candidate_id] = item.status
    candidate_status = [
        {"candidate_id": candidate_id, "status": statuses.get(candidate_id, "insufficient_evidence")}
        for candidate_id in sorted(allowed)
    ]
    requests = []
    for request in response.tool_requests[:2]:
        hypothesis_ids = list(dict.fromkeys(item for item in request.hypothesis_ids if item in allowed))
        if hypothesis_ids:
            requests.append(request.model_copy(update={"hypothesis_ids": hypothesis_ids}))
    action = response.action
    if action == "request_tools" and not requests:
        action = "unresolved"
    return VerifierResponse.model_validate({
        **response.model_dump(),
        "action": action,
        "candidate_status": candidate_status,
        "supporting_evidence_ids": list(dict.fromkeys(item for item in response.supporting_evidence_ids if item in evidence)),
        "conflicting_evidence_ids": list(dict.fromkeys(item for item in response.conflicting_evidence_ids if item in evidence)),
        "tool_requests": requests,
    })


def _normalize_adjudication(response: AdjudicationResponse, *, evidence: set[str], arguments: set[str]) -> AdjudicationResponse:
    annotations = [item.model_copy(update={
        "evidence_ids": list(dict.fromkeys(value for value in item.evidence_ids if value in evidence)),
    }) for item in response.population_annotations]
    return AdjudicationResponse.model_validate({
        **response.model_dump(),
        "supporting_evidence_ids": list(dict.fromkeys(item for item in response.supporting_evidence_ids if item in evidence)),
        "accepted_argument_ids": list(dict.fromkeys(item for item in response.accepted_argument_ids if item in arguments)),
        "population_annotations": [item.model_dump() for item in annotations],
    })


def _candidate_label(value: str | None, state: V5State, *, granularity: str | None = None) -> str | None:
    """Resolve internal candidate handles and canonicalize returned labels."""
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    candidates = [item for item in state.candidate_registry.candidates
                  if granularity is None or item.granularity == granularity]
    by_id = {item.candidate_id: item.label for item in candidates}
    if normalized in by_id:
        return by_id[normalized]
    by_label = {" ".join(item.label.casefold().split()): item.label for item in candidates}
    return by_label.get(" ".join(normalized.casefold().split()))


def _all_uncertain(response: VerifierResponse) -> bool:
    return bool(response.candidate_status) and all(
        item.status in {"uncertain", "insufficient_evidence"}
        for item in response.candidate_status
    )


def _reference_fallback(state: V5State, *, allowed: set[str]) -> tuple[str, dict[str, Any]] | None:
    """Select a dominant reference hit only when its margin is auditable."""
    entry = next((item for item in state.evidence_store
                  if item.source == "initial_cellmarker_retrieval" and item.usable), None)
    hits = entry.payload.get("hits", []) if entry else []
    if not isinstance(hits, list) or not hits:
        return None
    candidates = {" ".join(item.label.casefold().split()): item for item in state.candidate_registry.candidates
                  if item.candidate_id in allowed}
    ranked = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        candidate = candidates.get(" ".join(str(hit.get("label", "")).casefold().split()))
        score = hit.get("score")
        if candidate is not None and isinstance(score, (int, float)):
            ranked.append((float(score), candidate))
    ranked.sort(key=lambda item: (-item[0], item[1].candidate_id))
    if not ranked:
        return None
    top_score, top = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    gap = top_score - second_score
    # Require both useful absolute support and a non-trivial margin. This keeps
    # fallback deterministic without turning every ambiguous reference list into
    # a forced prediction.
    if top_score < 0.45 or gap < 0.05:
        return None
    return top.label, {
        "source_evidence_id": entry.evidence_id, "candidate_id": top.candidate_id,
        "top_score": top_score, "second_score": second_score, "score_gap": gap,
    }


def _arguments(drafts: ArgumentDraftResponse, state: V5State) -> list[ToulminArgument]:
    candidates = {v.candidate_id for v in state.candidate_registry.candidates}; evidence = {v.evidence_id for v in state.evidence_store if v.usable}
    values = []
    for i, d in enumerate(drafts.arguments, len(state.arguments)+1):
        if d.hypothesis_id not in candidates or not set(d.grounds_evidence_ids) <= evidence: raise ValueError("argument cites unknown candidate or evidence")
        values.append(ToulminArgument(argument_id=f"ARG_{i:03d}", revision=state.rounds["major"] + state.rounds["subtype"] + 1,
            target_id=state.task.cluster_id, hypothesis_id=d.hypothesis_id, claim_type=d.claim_type, claim=d.claim,
            grounds_evidence_ids=d.grounds_evidence_ids, qualifier_strength=d.qualifier_strength,
            supported_granularity=d.supported_granularity, limitations=d.limitations, rebuttals=d.rebuttals))
    return values


async def _verify(state: V5State, *, scope: str, client: JsonChatClient, model: str, temperature: float, retries: int, tools: Any, context: dict[str, Any]):
    candidates = [v for v in state.candidate_registry.candidates if v.granularity == scope]
    target = make_target(target_id=f"T_{scope.upper()}_{len(state.verification_targets)+1:03d}", task_id=state.task.task_id,
        source_target_id=state.task.cluster_id, cell_barcodes=[], supported_parent_ids=[v.parent_id for v in candidates if v.parent_id], reason=f"{scope}_verification")
    state.verification_targets.append(target)
    question = VerificationQuestion(question_id=f"Q_{scope.upper()}_{len(state.questions)+1:03d}",
        description=f"Which {scope} candidate is best supported by trusted evidence?", hypothesis_ids=[v.candidate_id for v in candidates],
        affected_level=scope, queryable=True, blocks_acceptance_of=[v.candidate_id for v in candidates])
    state.questions.append(question)
    allowed = {v.candidate_id for v in candidates}
    latest = None; latest_report = None
    for cycle in range(1, MAX_ROUNDS + 1):
        state.rounds[scope] = cycle
        drafts = await _complete(client, state, system=VERIFIER_SYSTEM, recipient="Verifier", stage=f"{scope}_arguments", model=model, temperature=temperature, retries=retries,
            response_type=ArgumentDraftResponse, payload={"task": state.task.model_dump(), "scope": scope, "candidate_registry": state.candidate_registry.model_dump(), "evidence_ledger": [v.model_dump() for v in state.evidence_store]},
            semantic_validator=None)
        drafts = _normalize_argument_drafts(drafts, state, allowed=allowed)
        state.arguments.extend(_arguments(drafts, state))
        evidence = {v.evidence_id for v in state.evidence_store}
        # Expose only candidates at the level currently being verified.  The
        # full registry may also contain subtype candidates; including those
        # invites the verifier to return statuses for out-of-scope hypotheses,
        # which correctly fail the strict semantic validation below.
        scoped_registry = {
            "version": state.candidate_registry.version,
            "candidates": [v.model_dump() for v in candidates],
        }
        latest = await _complete(client, state, system=VERIFIER_SYSTEM, recipient="Verifier", stage=f"verify_{scope}", model=model, temperature=temperature, retries=retries,
            response_type=VerifierResponse, payload={"task": state.task.model_dump(), "scope": scope,
                "allowed_candidate_ids": sorted(allowed), "candidate_registry": scoped_registry,
                "arguments": [v.model_dump() for v in state.arguments], "questions": [question.model_dump()],
                "evidence_ledger": [v.model_dump() for v in state.evidence_store]},
            semantic_validator=None)
        latest = _normalize_verifier_response(latest, allowed=allowed, evidence=evidence)
        status_map = _status_map(latest)
        latest_report = report(state=state, report_id=f"VR_{scope.upper()}_{cycle:02d}", target_id=target.target_id, scope=scope,
            candidate_status=status_map, supporting=latest.supporting_evidence_ids, conflicting=latest.conflicting_evidence_ids,
            revision_requests=[{"request": item} for item in latest.revision_requests], unresolved=latest.unresolved_questions, stop_reason=latest.stop_reason)
        if scope == "major" and cycle == 1 and latest.action != "request_tools" and _all_uncertain(latest):
            request = ToolRequest(
                request_id="REQ_MAJOR_01_AUTO", task_id=state.task.task_id, question_id=question.question_id,
                target_id=target.target_id, tool_name="query_cellmarker",
                hypothesis_ids=sorted(allowed), purpose="resolve_all_uncertain_major_candidates",
                discriminating_question="Which major candidate has the strongest marker-reference support?",
                arguments={"genes": state.task.ordered_markers},
            )
            dispatch_requests(state, [request], tools, context)
            continue
        if latest.action != "request_tools": return latest, latest_report
        requests = [ToolRequest(request_id=f"REQ_{scope.upper()}_{cycle:02d}_{index:02d}",
            task_id=state.task.task_id, question_id=question.question_id, target_id=target.target_id,
            tool_name=draft.tool_name, hypothesis_ids=draft.hypothesis_ids,
            purpose=draft.purpose, discriminating_question=draft.discriminating_question,
            arguments=draft.arguments.model_dump()) for index, draft in enumerate(latest.tool_requests, 1)]
        observations = dispatch_requests(state, requests, tools, context)
        for observation in observations:
            if observation.tool_name == "compare_reference" and observation.status == "success" and observation.result:
                value = observation.result
                state.reference_comparisons.append(ReferenceComparisonResult(comparison_id=f"RC_{len(state.reference_comparisons)+1:03d}", target_id=target.target_id,
                    reference_version=value["reference_version"], calculation_mode=value["calculation_mode"], score_matrix_handle=value["score_matrix_handle"], scores=value["scores"], decompositions=value["decompositions"], valid_markers=value["valid_markers"], coverage=value["coverage"], limitations=value.get("limitations", [])))
    return latest, latest_report


async def _workflow(state: V5State, *, client: JsonChatClient, model: str, temperature: float, retries: int, context: dict[str, Any]) -> V5State:
    audit = await _complete(client, state, system=MARKER_AUDIT_SYSTEM, recipient="MarkerAudit", stage="marker_audit", model=model, temperature=temperature, retries=retries, response_type=MarkerAudit, payload={"task": state.task.model_dump(), "evidence_ledger": [v.model_dump() for v in state.evidence_store]})
    state.marker_audit = audit
    trusted_labels = context.get("trusted_major_labels", [])
    major = await _complete(client, state, system=MAJOR_TYPE_SYSTEM, recipient="MajorType", stage="major_candidates", model=model, temperature=temperature, retries=retries, response_type=CandidateDraftResponse, payload={"task": state.task.model_dump(), "marker_audit": audit.model_dump(), "trusted_reference_candidates": trusted_labels, "evidence_ledger": [v.model_dump() for v in state.evidence_store]})
    state.candidate_registry = _register(major, source="MajorType", version=1, trusted_labels=trusted_labels)
    tools = build_tool_registry(context)
    major_reply, major_report = await _verify(state, scope="major", client=client, model=model, temperature=temperature, retries=retries, tools=tools, context=context)
    major_supported = [v for v in state.candidate_registry.candidates if v.granularity == "major" and major_report.candidate_status.get(v.candidate_id) == "supported"]
    subtype_report = None
    if major_reply.action == "advance_subtype" and major_supported:
        parent = major_supported[0]
        subtype = await _complete(client, state, system=SUBTYPE_SYSTEM, recipient="SubType", stage="subtype_candidates", model=model, temperature=temperature, retries=retries, response_type=CandidateDraftResponse, payload={"task": state.task.model_dump(), "parent_candidate": parent.model_dump(), "evidence_ledger": [v.model_dump() for v in state.evidence_store]})
        state.candidate_registry = _register(subtype, source="SubType", version=2, parent=parent)
        _, subtype_report = await _verify(state, scope="subtype", client=client, model=model, temperature=temperature, retries=retries, tools=tools, context=context)
    evidence = {v.evidence_id for v in state.evidence_store}; arguments = {v.argument_id for v in state.arguments}
    decision = await _complete(client, state, system=ADJUDICATOR_SYSTEM, recipient="Adjudicator", stage="adjudication", model=model, temperature=temperature, retries=retries, response_type=AdjudicationResponse, payload={"task": state.task.model_dump(), "candidate_registry": state.candidate_registry.model_dump(), "arguments": [v.model_dump() for v in state.arguments], "verification_reports": [v.model_dump() for v in state.verification_reports], "questions": [v.model_dump() for v in state.questions], "evidence_ledger": [v.model_dump() for v in state.evidence_store]})
    decision = _normalize_adjudication(decision, evidence=evidence, arguments=arguments)
    subtype_labels = {v.label for v in state.candidate_registry.candidates if v.granularity == "subtype" and subtype_report and subtype_report.candidate_status.get(v.candidate_id) == "supported"}
    status = decision.annotation_status
    fallback = None
    fallback_audit = None
    decision_identity = _candidate_label(decision.identity, state, granularity="major")
    decision_subtype = _candidate_label(decision.subtype, state, granularity="subtype")
    if status == "accepted" and decision_subtype not in subtype_labels: status = "major_accepted_subtype_unresolved" if major_supported else "unresolved"
    if status == "major_accepted_subtype_unresolved" and not major_supported: status = "unresolved"
    supported_major_labels = {item.label for item in major_supported}
    identity = decision_identity if decision_identity in supported_major_labels else None
    identity = identity or (major_supported[0].label if status == "major_accepted_subtype_unresolved" and major_supported else None)

    # If subtype evidence is unavailable, retain the highest-proportion
    # biological major annotation.  This is deliberately label-agnostic: the
    # candidate prompt filters technical expression programs, and the lookup
    # below prevents one from being promoted if it still appears in a model
    # response.  The proportion is a ranking signal, not a hard confidence
    # threshold.
    if status == "unresolved" and decision.population_annotations:
        major_labels = {candidate.label for candidate in state.candidate_registry.candidates if candidate.granularity == "major"}
        biological_annotations = [item for item in decision.population_annotations
                                   if item.fraction is not None and item.label in major_labels
                                   and _is_biological_cell_type(item.label)]
        if biological_annotations:
            dominant = max(biological_annotations, key=lambda item: item.fraction)
            status = "major_accepted_subtype_unresolved"
            identity = dominant.label
    if status == "unresolved":
        fallback = _reference_fallback(state, allowed={item.candidate_id for item in state.candidate_registry.candidates
                                                       if item.granularity == "major"})
        if fallback is not None:
            identity, fallback_audit = fallback
            status = "major_accepted_subtype_unresolved"
            decision.stop_reason = "deterministic_reference_fallback"
    # ``AdjudicationResponse`` stores typed ``PopulationAnnotation`` models,
    # while the persisted ``FinalAnnotation`` contract intentionally exposes
    # these entries as plain dictionaries.  Serialize explicitly here rather
    # than passing model instances into the stricter list[dict] field.
    state.final_result = FinalAnnotation(execution_status="success", annotation_status=status, identity=identity, subtype=decision_subtype if status == "accepted" else None, accepted_level="subtype" if status == "accepted" else ("major" if status == "major_accepted_subtype_unresolved" else ("composition" if status == "mixture" else "none")), cluster_summary=decision.cluster_summary, population_annotations=[item.model_dump() for item in decision.population_annotations], unresolved_questions=decision.unresolved_questions, supporting_evidence_ids=decision.supporting_evidence_ids, accepted_argument_ids=decision.accepted_argument_ids, fallback_audit=fallback_audit if fallback is not None else None, stop_reason=decision.stop_reason)
    return state


def build_graph(*, client: JsonChatClient, model: str, temperature: float, retries: int, work_dir: Path | None = None):
    """Compile the only V5.1 execution path."""
    async def execute(values: dict[str, Any]) -> dict[str, Any]:
        state = V5State.model_validate(values["state"])
        try:
            completed = await _workflow(state, client=client, model=model, temperature=temperature,
                                        retries=retries, context=values["tool_context"])
        except Exception as error:
            if work_dir:
                work_dir.mkdir(parents=True, exist_ok=True)
                failure = {
                    "protocol_version": PROTOCOL_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "workflow_version": WORKFLOW_VERSION,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "trace": _trace(state),
                }
                (work_dir / f"{state.task.cluster_id}.v5_failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            raise
        return {"state": completed.model_dump()}
    graph = StateGraph(dict); graph.add_node("evidence_workflow", execute); graph.add_edge(START, "evidence_workflow"); graph.add_edge("evidence_workflow", END)
    return graph.compile()


def _trace(state: V5State) -> dict[str, Any]:
    return {"task": state.task.model_dump(), "marker_audit": state.marker_audit.model_dump() if state.marker_audit else None, "candidate_registry": state.candidate_registry.model_dump(), "evidence_ledger": [v.model_dump() for v in state.evidence_store], "verification_targets": [v.model_dump() for v in state.verification_targets], "questions": [v.model_dump() for v in state.questions], "messages": [v.model_dump() for v in state.messages], "arguments": [v.model_dump() for v in state.arguments], "verification_reports": [v.model_dump() for v in state.verification_reports], "reference_comparisons": [v.model_dump() for v in state.reference_comparisons], "transition_history": state.transition_history, "rounds": state.rounds, "final_record": state.final_result.model_dump() if state.final_result else None}


def annotate_cluster(*, cluster_id: str, dataset: str, species: str, tissue: str, markers: list[str], client: JsonChatClient, model: str, temperature: float, retries: int, work_dir: Path | None = None, prompt_version: str = PROMPT_VERSION, protocol_version: str = PROTOCOL_VERSION, expression_file: str | Path | None = None, expression_manifest: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    marker_path, top_k, comparison = kwargs.pop("cellmarker_reference_path", ""), kwargs.pop("cellmarker_top_k", 5), kwargs.pop("comparison_reference_path", None)
    if kwargs: raise TypeError(f"unexpected V5 parameters: {', '.join(sorted(kwargs))}")
    if prompt_version != PROMPT_VERSION or protocol_version != PROTOCOL_VERSION: raise ValueError(f"CellClick V5 requires protocol_version=v5 and prompt_version={PROMPT_VERSION}")
    if expression_file is None or expression_manifest is None: raise ValueError("CellClick V5 requires expression evidence")
    if not marker_path: raise ValueError("CellClick V5 requires cellmarker_reference_path")
    task = TaskInput(task_id=cluster_id, cluster_id=cluster_id, dataset=dataset, species=species, tissue=tissue, ordered_markers=markers)
    expression = cluster_expression_evidence(expression_file=expression_file, manifest_file=expression_manifest, source_row_id=cluster_id)
    state = V5State(task=task, evidence_store=_initial_evidence(task, expression), budget={"tool_requests": 8, "revisions": 2})
    trusted_labels: list[str] = []
    reference_packet: dict[str, Any] = {}
    try:
        from cellclick.reference import MarkerReferenceRetriever
        reference_packet = MarkerReferenceRetriever.from_tsv(Path(marker_path)).retrieve(
            markers, tissue, top_k=top_k).prompt_dict()
        hits = reference_packet.get("hits", [])
        trusted_labels = [str(hit.get("label", "")).strip() for hit in hits if isinstance(hit, dict) and hit.get("label")]
    except Exception:
        # Candidate generation remains available if reference retrieval is
        # unavailable; the tool call later reports the capability limitation.
        trusted_labels = []
    reference_entry = _reference_evidence(task, reference_packet)
    if reference_entry is not None:
        state.evidence_store.append(reference_entry)
    context = {"expression_file": str(Path(expression_file).resolve()), "expression_manifest": str(Path(expression_manifest).resolve()), "source_row_id": cluster_id, "tissue": tissue, "cellmarker_reference_path": marker_path, "cellmarker_top_k": top_k, "comparison_reference_path": comparison, "trusted_major_labels": trusted_labels}
    async def run():
        try: return V5State.model_validate((await build_graph(client=client, model=model, temperature=temperature, retries=retries, work_dir=work_dir).ainvoke({"state": state.model_dump(), "tool_context": context}))["state"])
        finally: await close_json_client(client)
    completed = asyncio.run(run())
    trace = _trace(completed)
    if work_dir:
        work_dir.mkdir(parents=True, exist_ok=True); (work_dir / f"{cluster_id}.v5_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final = trace["final_record"] or {}; final.update({"protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION, "workflow_version": WORKFLOW_VERSION})
    return {"status": final.get("annotation_status", "failed"), "round_count": sum(completed.rounds.values()),
            "fallback": final.get("stop_reason") == "deterministic_reference_fallback",
            "final_record": final, "trace": trace}
