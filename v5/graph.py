"""Program-controlled V5.9 execution graph."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph
from cellclick.llm import JsonChatClient, close_json_client
from cellclick.retry import retry_json
from .context import build_llm_context
from .expression import marker_expression_evidence
from .ledger import snapshot_id
from .models import (AdjudicationResponse, CandidateDraft, CandidateProposal, MajorCandidateResponse,
                     CandidateRegistry, EvidenceLedgerEntry, FinalAnnotation, MarkerAudit,
                     MessageEnvelope, SubtypeCandidateResponse, TaskInput, V5State)
from .prompts import (ADJUDICATOR_SYSTEM, MAJOR_TYPE_SYSTEM, MARKER_AUDIT_SYSTEM,
                      PROMPT_VERSION, SUBTYPE_SYSTEM)
from .tools import build_tool_registry
from .verifier import verify
from .candidate_registry import register_major, register_subtype

PROTOCOL_VERSION = "v5"
ANNOTATION_CANDIDATES, MAX_ROUNDS, REQUEST_RETRIES, MAX_PARALLEL_CLUSTERS = 5, 6, 2, 2
WORKFLOW_VERSION = "v5.9-deduplicated-llm-context-cosine-summary"


@lru_cache(maxsize=16)
def _cached_file_sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: str) -> str:
    resolved = Path(path).resolve()
    from .runtime import CURRENT
    runtime = CURRENT.get()
    if runtime and str(resolved) in runtime.resources.hashes:
        return runtime.resources.hashes[str(resolved)]
    info = resolved.stat()
    return _cached_file_sha256(str(resolved), info.st_size, info.st_mtime_ns)

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
    rows = expression.get("genes", [])
    by_gene = {str(row.get("gene")): row for row in rows if isinstance(row, dict)}
    for i, gene in enumerate(task.ordered_markers, 1):
        row = by_gene.get(gene, {"gene": gene, "status": "unavailable", "available": False})
        result.append(EvidenceLedgerEntry(evidence_id=f"E_EXPR_{i:03d}", task_id=task.task_id,
            target_id=task.cluster_id, kind="derived", source="marker_expression_statistics", payload=row))
    return result


def _validate_marker_audit(audit: MarkerAudit, task: TaskInput, evidence: list[EvidenceLedgerEntry]) -> None:
    expected = [" ".join(g.strip().casefold().split()) for g in task.ordered_markers]
    seen = [" ".join(item.gene.strip().casefold().split()) for item in audit.marker_assessments]
    # Legacy V5 responses may omit this field; preserve replay compatibility.
    if not seen:
        return
    if set(seen) != set(expected) or len(seen) != len(set(seen)):
        raise ValueError("marker_audit.marker_assessments must cover each input marker exactly once")
    valid = {item.evidence_id for item in evidence}
    if not set(audit.evidence_ids) <= valid:
        raise ValueError("marker_audit references unknown evidence ID")
    for item in audit.marker_assessments:
        if not set(item.evidence_ids) <= valid:
            raise ValueError("marker_audit references unknown evidence ID")


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


def _register(drafts: Any, *, source: str, version: int, parent: CandidateProposal | None = None,
              trusted_labels: list[str] | None = None, reference_max: int = 3,
              subtype_catalog: list[dict[str, Any]] | None = None,
              usable_evidence: set[str] | None = None) -> CandidateRegistry:
    if isinstance(drafts, MajorCandidateResponse):
        raw = []
        for item in drafts.selected_reference_candidates:
            raw.append({"label": item.label, "source": "cellmarker", "reference_rank": item.reference_rank,
                        "evidence_ids": item.evidence_ids, "rationale": item.rationale})
        for item in drafts.marker_based_candidates:
            raw.append({"label": item.label, "source": "marker_based", "reference_rank": None,
                        "evidence_ids": item.evidence_ids, "rationale": item.rationale})
        return register_major(raw, version=version)
    if isinstance(drafts, SubtypeCandidateResponse):
        if parent is None:
            raise ValueError("subtype registration requires a supported parent")
        usable = usable_evidence or set()
        raw = []
        for item in drafts.candidates:
            evidence_ids = list(dict.fromkeys(item.evidence_ids))
            if not set(evidence_ids) <= usable:
                continue
            raw.append({"cl_id": item.cl_id, "evidence_ids": evidence_ids, "rationale": item.rationale})
        return register_subtype(raw, parent=parent, catalog=subtype_catalog or [], version=version,
                                usable_evidence=usable)
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
                    payload: dict[str, Any], model: str, temperature: float, retries: int,
                    response_type: Any, semantic_validator: Any | None = None) -> Any:
    envelope = MessageEnvelope(
        message_id=f"M_{len(state.messages)+1:04d}", task_id=state.task.task_id,
        target_id=state.task.cluster_id, sender="program", recipient=recipient,
        message_type=stage, verification_scope=payload.get("scope"),
        candidate_registry_version=state.candidate_registry.version,
        evidence_snapshot_id=snapshot_id(state.evidence_store), payload=payload)
    state.messages.append(envelope)
    scope = payload.get("scope")
    def validate(value: dict[str, Any]) -> Any:
        try:
            parsed = response_type.model_validate(value)
            if semantic_validator is not None:
                semantic_validator(parsed)
        except (ValueError, TypeError) as error:
            state.verification_history.append({
                "role": recipient, "scope": scope, "stage": stage,
                "round": state.rounds.get(scope, 0), "status": "invalid",
                "response": value, "error": str(error)})
            raise
        state.verification_history.append({
            "role": recipient, "scope": scope, "stage": stage,
            "round": state.rounds.get(scope, 0), "status": "valid", "response": parsed.model_dump()})
        return parsed
    value, _ = await retry_json(
        client, system=system, user=envelope.model_dump_json(), schema=response_type.model_json_schema(),
        model=model, temperature=temperature, retries=retries, validator=validate,
        strict=True, stage=stage, v5_policy=True)
    state.transition_history.append({"stage": stage, "message_id": envelope.message_id,
                                     "evidence_snapshot_id": envelope.evidence_snapshot_id})
    return response_type.model_validate(value)


async def _verify(state: V5State, **kwargs: Any):
    return await verify(state, complete=_complete, **kwargs)


def _query_subtype_catalog(parent: CandidateProposal) -> tuple[list[dict[str, Any]], str | None]:
    if not parent.cl_id:
        return [], "parent_cl_id_missing"
    try:
        from evaluator.solver.CLSolver import CLSolver
        catalog = CLSolver().get_children(parent.cl_id)
    except Exception as error:
        return [], f"catalog_query_failed:{type(error).__name__}:{error}"
    if not catalog:
        return [], "catalog_empty"
    return catalog, None


def _record_subtype_skip(state: V5State, reason: str) -> None:
    state.transition_history.append({
        "stage": "subtype_candidates",
        "status": "skipped",
        "reason": reason,
    })


async def _workflow(state: V5State, *, client: JsonChatClient, model: str, temperature: float, retries: int, context: dict[str, Any]) -> V5State:
    audit = await _complete(client, state, system=MARKER_AUDIT_SYSTEM, recipient="MarkerAudit", stage="marker_audit", model=model, temperature=temperature, retries=retries, response_type=MarkerAudit, payload={"task": state.task.model_dump(), "evidence_ledger": [v.model_dump() for v in state.evidence_store]}, semantic_validator=lambda value: _validate_marker_audit(value, state.task, state.evidence_store))
    state.marker_audit = audit
    trusted_labels = context.get("trusted_major_labels", [])
    reference_candidates = context.get("reference_candidates", [])
    reference_max = int(context.get("reference_max", 3))
    def validate_major(value: MajorCandidateResponse) -> None:
        expected = {str(hit.get("rank")): hit for hit in reference_candidates if isinstance(hit, dict)}
        if len(value.selected_reference_candidates) > reference_max or len(value.marker_based_candidates) > 5 - reference_max:
            raise ValueError("major candidate source budgets exceeded")
        genes = {" ".join(g.strip().casefold().split()) for g in state.task.ordered_markers}
        interpreted = [" ".join(x.gene.strip().casefold().split()) for x in value.marker_interpretations]
        if set(interpreted) != genes or len(interpreted) != len(set(interpreted)):
            raise ValueError("marker_interpretations must cover each input marker exactly once")
        usable = {x.evidence_id for x in state.evidence_store if x.usable}
        for item in [*value.selected_reference_candidates, *value.marker_based_candidates]:
            if not set(item.evidence_ids) <= usable: raise ValueError("candidate cites unknown or unusable evidence")
        for item in value.selected_reference_candidates:
            hit = expected.get(str(item.reference_rank))
            if hit is None or str(hit.get("label", "")).strip().casefold() != item.label.strip().casefold():
                raise ValueError("reference candidate does not match retrieval result")
    major = await _complete(client, state, system=MAJOR_TYPE_SYSTEM, recipient="MajorType", stage="major_candidates", model=model, temperature=temperature, retries=retries, response_type=MajorCandidateResponse, payload={"task": state.task.model_dump(), "marker_audit": audit.model_dump(), "reference_candidates": reference_candidates, "reference_max": reference_max, "marker_based_max": 5 - reference_max, "evidence_ledger": [v.model_dump() for v in state.evidence_store]}, semantic_validator=validate_major)
    state.candidate_registry = _register(major, source="MajorType", version=1, reference_max=reference_max)
    if not state.candidate_registry.candidates:
        raise ValueError("no_major_candidates: no legal biological identity was registered")
    major_report = await _verify(state, scope="major", client=client, model=model, temperature=temperature,
                                 retries=retries, tools=build_tool_registry(context), context=context)
    original_major_registry = state.candidate_registry
    major_supported = [v for v in state.candidate_registry.candidates
                       if v.candidate_id == major_report.selected_candidate_id]
    subtype_report = None
    if major_supported:
        parent = major_supported[0]
        subtype_catalog, catalog_error = _query_subtype_catalog(parent)
        if catalog_error:
            _record_subtype_skip(state, catalog_error)
        else:
            usable = {entry.evidence_id for entry in state.evidence_store if entry.usable}
            def validate_subtype(value: SubtypeCandidateResponse) -> None:
                allowed = {str(item.get("cl_id")) for item in subtype_catalog}
                for item in value.candidates:
                    if item.cl_id not in allowed or item.cl_id == parent.cl_id:
                        raise ValueError("subtype candidate CLID is outside the supplied catalog")
                    if not set(item.evidence_ids) <= usable:
                        raise ValueError("subtype candidate cites unknown or unusable evidence")

            subtype = await _complete(
                client, state, system=SUBTYPE_SYSTEM, recipient="SubType", stage="subtype_candidates",
                model=model, temperature=temperature, retries=retries,
                response_type=SubtypeCandidateResponse,
                payload={**build_llm_context(state), "parent_candidate": parent.model_dump(),
                         "subtype_catalog": subtype_catalog},
                semantic_validator=validate_subtype,
            )
            state.candidate_registry = _register(
                subtype, source="SubType", version=2, parent=parent,
                subtype_catalog=subtype_catalog, usable_evidence=usable,
            )
            child_ids = {v.candidate_id for v in state.candidate_registry.candidates if v.granularity == "subtype"}
            merged = {v.candidate_id: v for v in original_major_registry.candidates}
            merged.update({v.candidate_id: v for v in state.candidate_registry.candidates})
            state.candidate_registry = CandidateRegistry(
                version=2, candidates=list(merged.values()),
                parent_by_id={key: value.parent_id for key, value in merged.items()},
                competing_by_id={key: value.competing_candidate_ids for key, value in merged.items()})
            if not child_ids:
                _record_subtype_skip(state, "zero_candidates")
            else:
                subtype_tools = build_tool_registry(context)
                subtype_report = await _verify(state, scope="subtype", client=client, model=model, temperature=temperature, retries=retries, tools=subtype_tools, context=context)

    authoritative_report = subtype_report or major_report
    chosen = next(item for item in state.candidate_registry.candidates
                  if item.candidate_id == authoritative_report.selected_candidate_id)
    parents = {item.candidate_id: item for item in state.candidate_registry.candidates}
    expected_identity = parents[chosen.parent_id].label if chosen.granularity == "subtype" else chosen.label
    expected_subtype = chosen.label if chosen.granularity == "subtype" else None
    expected_status = "accepted" if chosen.granularity == "subtype" else "major_accepted_subtype_unresolved"
    usable_evidence = {item.evidence_id for item in state.evidence_store if item.usable}
    argument_ids = {item.argument_id for item in state.arguments}

    def validate_adjudication(value: AdjudicationResponse) -> None:
        required = {
            "selected_candidate_id": chosen.candidate_id,
            "annotation_status": expected_status,
            "identity": expected_identity,
            "subtype": expected_subtype,
            "verification_status": authoritative_report.verification_status,
            "verification_report_id": authoritative_report.report_id,
        }
        for field, expected in required.items():
            if getattr(value, field) != expected:
                raise ValueError(f"{field}: must be {expected!r} according to terminal report "
                                 f"{authoritative_report.report_id}; rewrite the complete decision consistently")
        if not value.cluster_summary.strip():
            raise ValueError("cluster_summary: give a non-empty explanation consistent with the selected identity")
        if not value.supporting_evidence_ids or not set(value.supporting_evidence_ids) <= usable_evidence:
            raise ValueError(f"supporting_evidence_ids: cite usable IDs from {sorted(usable_evidence)}")
        if not set(authoritative_report.supporting_evidence_ids) <= set(value.supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids: retain the terminal report evidence")
        if not set(value.accepted_argument_ids) <= argument_ids:
            raise ValueError(f"accepted_argument_ids: legal IDs are {sorted(argument_ids)}")

    decision = await _complete(
        client, state, system=ADJUDICATOR_SYSTEM, recipient="Adjudicator", stage="adjudication",
        model=model, temperature=temperature, retries=retries, response_type=AdjudicationResponse,
        payload={**build_llm_context(state), "authoritative_report_id": authoritative_report.report_id,
                 "arguments": [item.model_dump() for item in state.arguments],
                 },
        semantic_validator=validate_adjudication)
    state.final_result = FinalAnnotation(
        execution_status="success", accepted_level=chosen.granularity,
        resolution_audit=authoritative_report.resolution_audit, **decision.model_dump())
    return state


def _trace(state: V5State) -> dict[str, Any]:
    return {
        "task": state.task.model_dump(),
        "marker_audit": state.marker_audit.model_dump() if state.marker_audit else None,
        "candidate_registry": state.candidate_registry.model_dump(),
        "evidence_ledger": [item.model_dump() for item in state.evidence_store],
        "verification_targets": [item.model_dump() for item in state.verification_targets],
        "questions": [item.model_dump() for item in state.questions],
        "messages": [item.model_dump() for item in state.messages],
        "arguments": [item.model_dump() for item in state.arguments],
        "verification_reports": [item.model_dump() for item in state.verification_reports],
        "reference_comparisons": [item.model_dump() for item in state.reference_comparisons],
        "verification_history": state.verification_history,
        "transition_history": state.transition_history, "rounds": state.rounds,
        "final_record": state.final_result.model_dump() if state.final_result else None,
    }


def build_graph(*, client: JsonChatClient, model: str, temperature: float, retries: int,
                work_dir: Path | None = None):
    async def execute(values: dict[str, Any]) -> dict[str, Any]:
        state = V5State.model_validate(values["state"])
        try:
            completed = await _workflow(state, client=client, model=model, temperature=temperature,
                                        retries=retries, context=values["tool_context"])
        except Exception as error:
            state.final_result = FinalAnnotation(execution_status="failed", annotation_status="failed",
                                                  accepted_level="none", stop_reason=type(error).__name__)
            if work_dir:
                work_dir.mkdir(parents=True, exist_ok=True)
                failure = {"protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION,
                           "workflow_version": WORKFLOW_VERSION, "error_type": type(error).__name__,
                           "error": str(error), "trace": _trace(state)}
                (work_dir / f"{state.task.cluster_id}.v5_failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        return {"state": completed.model_dump()}
    graph = StateGraph(dict)
    graph.add_node("evidence_workflow", execute)
    graph.add_edge(START, "evidence_workflow")
    graph.add_edge("evidence_workflow", END)
    return graph.compile()


def _annotate_cluster(*, cluster_id: str, dataset: str, species: str, tissue: str,
                     markers: list[str], client: JsonChatClient, model: str, temperature: float,
                     retries: int, work_dir: Path | None = None, prompt_version: str = PROMPT_VERSION,
                     protocol_version: str = PROTOCOL_VERSION, expression_file: str | Path | None = None,
                     expression_manifest: str | Path | None = None, verifier_max_rounds: int = MAX_ROUNDS,
                     **kwargs: Any) -> dict[str, Any]:
    marker_path = kwargs.pop("cellmarker_reference_path", "")
    top_k = kwargs.pop("cellmarker_top_k", 5)
    binding = kwargs.pop("comparison_reference_binding", None)
    reference_max = kwargs.pop("reference_max", 3)
    if kwargs:
        raise TypeError(f"unexpected V5 parameters: {', '.join(sorted(kwargs))}")
    if type(verifier_max_rounds) is not int or not 1 <= verifier_max_rounds <= MAX_ROUNDS:
        raise ValueError("verifier_max_rounds must be an integer between 1 and 6")
    if prompt_version != PROMPT_VERSION or protocol_version != PROTOCOL_VERSION:
        raise ValueError(f"CellClick V5 requires protocol_version=v5 and prompt_version={PROMPT_VERSION}")
    if expression_file is None or expression_manifest is None:
        raise ValueError("CellClick V5 requires expression evidence")
    if not marker_path:
        raise ValueError("CellClick V5 requires cellmarker_reference_path")
    task = TaskInput(task_id=cluster_id, cluster_id=cluster_id, dataset=dataset, species=species,
                     tissue=tissue, ordered_markers=markers)
    expression = marker_expression_evidence(expression_file=expression_file, manifest_file=expression_manifest,
                                             source_row_id=cluster_id, genes=markers)
    state = V5State(task=task, evidence_store=_initial_evidence(task, expression))
    reference_packet: dict[str, Any] = {}
    try:
        from cellclick.reference import MarkerReferenceRetriever
        from .runtime import CURRENT
        runtime = CURRENT.get()
        retriever = runtime.resources.retriever(marker_path) if runtime else MarkerReferenceRetriever.from_tsv(Path(marker_path))
        reference_packet = retriever.retrieve(
            markers, tissue, top_k=top_k).prompt_dict()
    except Exception as error:
        state.verification_history.append({"role": "program", "stage": "initial_cellmarker",
                                           "status": "failed", "error_type": type(error).__name__})
    reference_entry = _reference_evidence(task, reference_packet)
    if reference_entry is not None:
        state.evidence_store.append(reference_entry)
    comparison = binding.get("ref_path") if isinstance(binding, dict) else None
    if comparison:
        try:
            from .runtime import read_json
            manifest = read_json(expression_manifest)
            configured_h5ad = Path(str(binding.get("h5ad_path", ""))).resolve()
            source_h5ad = Path(str(manifest.get("source_h5ad_path", ""))).resolve()
            if not configured_h5ad.is_file() or configured_h5ad != source_h5ad:
                raise LookupError("comparison_reference_source_h5ad_mismatch")
            loaded = read_json(comparison)
            if not isinstance(loaded, dict) or not all(
                    isinstance(label, str) and isinstance(genes, list)
                    and all(isinstance(gene, str) for gene in genes) for label, genes in loaded.items()):
                raise ValueError("comparison reference must map labels to gene lists")
        except (OSError, ValueError, LookupError) as error:
            state.verification_history.append({"role": "program", "stage": "reference_binding",
                                               "status": "unavailable", "reason": str(error)})
            comparison = None
    context = {
        "expression_file": str(Path(expression_file).resolve()),
        "expression_manifest": str(Path(expression_manifest).resolve()), "source_row_id": cluster_id,
        "tissue": tissue, "cellmarker_reference_path": marker_path, "cellmarker_top_k": top_k,
        "comparison_reference_path": comparison, "query_markers": markers,
        "work_dir": str(work_dir or Path.cwd()), "reference_candidates": reference_packet.get("hits", []),
        "reference_max": reference_max, "verifier_max_rounds": verifier_max_rounds,
    }
    for key in ("expression_file", "expression_manifest", "comparison_reference_path"):
        if context.get(key):
            context[key + "_sha256"] = _file_sha256(context[key])
    async def run():
        try:
            values = await build_graph(client=client, model=model, temperature=temperature,
                                       retries=retries, work_dir=work_dir).ainvoke(
                                           {"state": state.model_dump(), "tool_context": context})
            return V5State.model_validate(values["state"])
        finally:
            await close_json_client(client)
    completed = asyncio.run(run())
    trace = _trace(completed)
    if work_dir:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / f"{cluster_id}.v5_trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final = trace["final_record"] or {}
    final.update({"protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION,
                  "workflow_version": WORKFLOW_VERSION})
    return {"status": final.get("verification_status", "failed"),
            "round_count": sum(completed.rounds.values()), "rounds": completed.rounds,
            "final_record": final, "trace": trace}


def annotate_cluster(*, runtime=None, **kwargs):
    """Bind non-serializable resources outside graph state and model context."""
    if runtime is None:
        return _annotate_cluster(**kwargs)
    with runtime.activate():
        return _annotate_cluster(**kwargs)
