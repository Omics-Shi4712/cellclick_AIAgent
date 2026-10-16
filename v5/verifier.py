"""One state machine for both V5 verification scopes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .context import build_llm_context
from .ledger import make_target, observation_entry
from .models import (QueryBuilderResponse, ResolutionAudit, ResolutionResponse, ToolObservation,
                     ToolRequest, V5State, VerificationQuestion, VerificationReport, VerifierResponse)
from .prompts import QUERY_BUILDER_SYSTEM, RESOLVER_SYSTEM, VERIFIER_SYSTEM
from .tools import ToolRegistry, parameter_hash
from .runtime import read_json, CURRENT, load_expression, emit

STATE_ACTIONS = {
    "multiple": ("check_expression", "query_expression"),
    "mixture_cell": ("check_coexpression", "gene_cosine"),
    "validation": ("validate_candidate", "compare_reference"),
    "solved": ("finish", None),
    "unresolved": ("final_resolve", None),
    "resolved": ("finish", None),
}


def reference_options(state: V5State, context: dict[str, Any], allowed: set[str]) -> dict[str, dict[str, Any]]:
    path = context.get("comparison_reference_path")
    reference = read_json(path) if path else {}
    options = {}
    for candidate in state.candidate_registry.candidates:
        if candidate.candidate_id not in allowed:
            continue
        matches = [label for label in reference
                   if " ".join(label.casefold().split()) == " ".join(candidate.label.casefold().split())]
        if len(matches) == 1:
            label = matches[0]
            options[candidate.candidate_id] = {"reference_label": label, "genes": reference[label]}
    return options


def _genes(values: list[str], context: dict[str, Any]) -> list[str]:
    """Canonicalize unambiguous gene names; keep missing names for coverage reporting."""
    if "_gene_names" not in context:
        runtime = CURRENT.get()
        if runtime:
            context["_gene_names"] = runtime.resources.casefold_index
    if "_gene_names" not in context:
        import anndata as ad
        adata = load_expression(context["expression_file"]) if CURRENT.get() else ad.read_h5ad(context["expression_file"], backed="r")
        try:
            index: dict[str, list[str]] = {}
            for value in adata.var_names.astype(str):
                index.setdefault(value.casefold(), []).append(value)
            context["_gene_names"] = index
        finally:
            if hasattr(adata, "file"):
                adata.file.close()
    index = context["_gene_names"]
    result = []
    for value in values:
        gene = value.strip()
        if not gene:
            raise ValueError("genes: empty gene; legal range is 1-200 non-empty unique gene names")
        matches = index.get(gene.casefold(), [])
        if len(matches) > 1:
            raise ValueError(f"genes: ambiguous name {gene!r}; use an unambiguous expression gene")
        normalized = matches[0] if matches else gene
        if normalized not in result:
            result.append(normalized)
    if not 1 <= len(result) <= 200:
        raise ValueError("genes: legal range is 1-200 non-empty unique genes")
    return result


def query_arguments(value: QueryBuilderResponse, *, active: str, allowed: set[str],
                    pending: str | None, options: dict[str, dict[str, Any]],
                    context: dict[str, Any]) -> dict[str, Any]:
    if value.no_query_reason is not None:
        if not value.no_query_reason.strip() or value.genes or value.gene_sets or value.compare_pairs or value.marker_sets:
            raise ValueError("no_query_reason: provide a reason and no query parameters")
        return {}
    if active == "multiple":
        if value.gene_sets or value.compare_pairs or value.marker_sets:
            raise ValueError("query_expression: only genes and question are allowed")
        return {"genes": _genes(value.genes, context)}
    if active == "mixture_cell":
        if value.genes or value.marker_sets or value.question:
            raise ValueError("gene_cosine: only gene_sets and compare_pairs are allowed")
        ids = [item.candidate_id for item in value.gene_sets]
        if len(ids) < 2 or len(ids) != len(set(ids)) or not set(ids) <= allowed:
            raise ValueError(f"gene_sets.candidate_id: use at least two distinct IDs from {sorted(allowed)}")
        pairs = [[item.left_candidate_id, item.right_candidate_id] for item in value.compare_pairs]
        if not pairs or any(a == b or a not in ids or b not in ids for a, b in pairs):
            raise ValueError(f"compare_pairs: use distinct IDs from the supplied gene_sets {ids}")
        return {"gene_sets": [{"candidate_id": item.candidate_id, "genes": _genes(item.genes, context)}
                              for item in value.gene_sets], "compare_pairs": pairs}
    if value.genes or value.gene_sets or value.compare_pairs or value.question or len(value.marker_sets) != 1:
        raise ValueError("compare_reference: provide exactly one marker_sets entry and no other query parameters")
    item = value.marker_sets[0]
    if item.candidate_id != pending:
        raise ValueError(f"marker_sets.candidate_id: must be pending candidate {pending}")
    selected = options.get(item.candidate_id)
    if selected:
        if item.source != "reference" or item.reference_label != selected["reference_label"]:
            raise ValueError(f"marker_sets.source/reference_label: use matching reference {selected['reference_label']!r}")
        if not set(item.genes) <= set(selected["genes"]):
            raise ValueError(f"marker_sets.genes: must be selected from {selected['genes']}")
    elif item.source != "llm" or item.reference_label is not None:
        raise ValueError("marker_sets.source: no matching reference; use source=llm and reference_label=null")
    genes = _genes(item.genes, context)
    return {"marker_sets": [{**item.model_dump(), "genes": genes}]}


def dispatch_requests(state: V5State, requests: Iterable[ToolRequest], registry: ToolRegistry,
                      context: dict[str, Any]) -> list[ToolObservation]:
    observations = []
    cache = context.setdefault("_tool_cache", {})
    for request in requests:
        digest = parameter_hash(request, context)
        question = next((value for value in state.questions if value.question_id == request.question_id), None)
        targets = {value.target_id for value in state.verification_targets}
        if question is None or request.target_id not in targets or request.task_id != state.task.task_id:
            observation = ToolObservation(
                observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                task_id=state.task.task_id, target_id=request.target_id, tool_name=request.tool_name,
                status="protocol_error", parameter_hash=digest, limitations=["invalid_question_target_or_task"])
        elif digest in cache:
            emit("tool_cache", cache_hit=True, tool_name=request.tool_name)
            original = cache[digest]
            observation = original.model_copy(update={"observation_id": f"OBS_{request.request_id}",
                                                       "request_id": request.request_id,
                                                       "evidence_ids": [],
                                                       "limitations": [*original.limitations, "cached_result"]})
        else:
            emit("tool_cache", cache_hit=False, tool_name=request.tool_name)
            observation = registry.dispatch(request, context)
            if observation.status == "success":
                cache[digest] = observation
        entry = observation_entry(observation)
        observation.evidence_ids = [entry.evidence_id]
        state.evidence_store.append(entry)
        if question is not None:
            question.evidence_ids.append(entry.evidence_id)
        observations.append(observation)
    return observations


def _check_evidence(ids: list[str], state: V5State) -> None:
    valid = {item.evidence_id for item in state.evidence_store}
    if not set(ids) <= valid:
        raise ValueError(f"evidence_ids: unknown references; legal IDs are {sorted(valid)}")


async def verify(state: V5State, *, scope: str, client: Any, model: str, temperature: float,
                 retries: int, tools: ToolRegistry, context: dict[str, Any],
                 complete: Callable[..., Any]) -> VerificationReport:
    candidates = [item for item in state.candidate_registry.candidates if item.granularity == scope]
    allowed = {item.candidate_id for item in candidates}
    if not allowed:
        raise ValueError(f"{scope}: no registered verification candidates")
    parent_ids = {item.parent_id for item in candidates if item.parent_id}
    final_allowed = allowed | parent_ids
    options = reference_options(state, context, allowed)
    context["allowed_candidate_ids"] = allowed
    max_rounds = context["verifier_max_rounds"]
    target = make_target(target_id=f"T_{scope.upper()}", task_id=state.task.task_id,
                         source_target_id=state.task.cluster_id, cell_barcodes=[],
                         supported_parent_ids=sorted(parent_ids), reason=f"{scope}_verification")
    state.verification_targets.append(target)
    question = VerificationQuestion(question_id=f"Q_{scope.upper()}", description=f"{scope} candidate verification",
                                    hypothesis_ids=sorted(allowed), affected_level=scope, queryable=True)
    state.questions.append(question)
    active, pending, last_active = "multiple", None, "multiple"
    latest = None
    trigger = "max_rounds_reached"

    def payload() -> dict[str, Any]:
        return {**build_llm_context(state), "scope": scope, "state": active,
                "rounds_used": state.rounds[scope], "max_rounds": max_rounds,
                "allowed_candidate_ids": sorted(allowed), "allowed_parent_ids": sorted(parent_ids),
                "pending_candidate_id": pending, "reference_options": options}

    for cycle in range(1, max_rounds + 1):
        state.rounds[scope] = cycle
        last_active = active
        action, tool_name = STATE_ACTIONS[active]
        state.verification_history.append({"role": "program", "scope": scope, "round": cycle,
                                           "state": active, "action": action, "tool_name": tool_name})
        arguments: dict[str, Any] = {}
        try:
            draft = await complete(
                client, state, system=QUERY_BUILDER_SYSTEM, recipient="LLM_B",
                stage=f"{scope}_query", model=model, temperature=temperature, retries=retries,
                response_type=QueryBuilderResponse,
                payload={**payload(), "bound_tool": tool_name},
                semantic_validator=lambda value: query_arguments(
                    value, active=active, allowed=allowed, pending=pending, options=options, context=context))
            if draft.no_query_reason is not None:
                trigger = "no_executable_query"
                state.verification_history.append({"role": "program", "scope": scope, "round": cycle,
                                                   "event": "no_query", "reason": draft.no_query_reason})
                break
            arguments = query_arguments(draft, active=active, allowed=allowed, pending=pending,
                                        options=options, context=context)
            request = ToolRequest(request_id=f"REQ_{scope.upper()}_{cycle:02d}", task_id=state.task.task_id,
                                  question_id=question.question_id, target_id=target.target_id,
                                  tool_name=tool_name, hypothesis_ids=[pending] if pending else sorted(allowed),
                                  purpose=draft.rationale, discriminating_question=draft.question or draft.rationale,
                                  arguments=arguments)
            observation = dispatch_requests(state, [request], tools, context)[0]
        except (ValueError, TypeError) as error:
            observation = ToolObservation(
                observation_id=f"OBS_{scope.upper()}_{cycle:02d}_PARAMETER_ERROR",
                request_id=f"REQ_{scope.upper()}_{cycle:02d}", task_id=state.task.task_id,
                target_id=target.target_id, tool_name=tool_name, status="protocol_error",
                parameter_hash="invalid_parameters", limitations=[str(error)])
            entry = observation_entry(observation)
            observation.evidence_ids = [entry.evidence_id]
            state.evidence_store.append(entry)
        state.verification_history.append({"role": "tool", "scope": scope, "round": cycle,
                                           "state": active, "action": action,
                                           "arguments": arguments, "observation": observation.model_dump()})

        def validate(value: VerifierResponse) -> None:
            _check_evidence(value.evidence_ids, state)
            transitions = {"multiple": {"multiple", "mixture_cell", "validation", "unresolved"},
                           "mixture_cell": {"mixture_cell", "validation", "unresolved"},
                           "validation": {"multiple", "solved", "unresolved"}}
            if value.next_state not in transitions[active]:
                raise ValueError(f"next_state: legal transitions from {active} are {sorted(transitions[active])}")
            if value.next_state in {"validation", "solved"} and value.candidate_id not in allowed:
                raise ValueError(f"candidate_id: choose from {sorted(allowed)}")
            if value.candidate_id is not None and value.candidate_id not in allowed:
                raise ValueError(f"candidate_id: choose from {sorted(allowed)} or null")
            if active != "validation" and value.validation_passed is not None:
                raise ValueError("validation_passed: must be null outside validation")
            if active == "validation" and value.next_state == "multiple" and value.validation_passed is not False:
                raise ValueError("validation_passed: failed validation returning to multiple must be false")
            if value.next_state == "solved":
                result = observation.result or {}
                if (value.validation_passed is not True or value.candidate_id != pending
                        or observation.status != "success"
                        or pending not in result.get("candidate_ids", [])
                        or not any(float(v) > 0 for v in result.get("coverage", {}).values())
                        or not set(observation.evidence_ids) <= set(value.evidence_ids)):
                    raise ValueError("solved: requires the pending candidate, successful comparison with coverage, "
                                     "validation_passed=true, and current observation evidence IDs")

        latest = await complete(
            client, state, system=VERIFIER_SYSTEM, recipient="LLM_A", stage=f"{scope}_assessment",
            model=model, temperature=temperature, retries=retries, response_type=VerifierResponse,
            payload={**payload(), **build_llm_context(state, observation)}, semantic_validator=validate)
        next_state = latest.next_state
        if next_state == "validation":
            pending = latest.candidate_id
        state.transition_history.append({"scope": scope, "round": cycle, "from": active,
                                         "to": next_state, "action": action, "reason": latest.reason,
                                         "candidate_id": latest.candidate_id,
                                         "validation_passed": latest.validation_passed,
                                         "evidence_ids": latest.evidence_ids})
        if next_state == "solved":
            report = VerificationReport(
                report_id=f"VR_{scope.upper()}", target_id=target.target_id, scope=scope,
                verification_status="solved", selected_candidate_id=pending,
                last_active_state=active, supporting_evidence_ids=latest.evidence_ids,
                unresolved_questions=latest.unresolved_questions,
                remaining_limitations=latest.remaining_limitations, stop_reason=latest.reason)
            state.verification_reports.append(report)
            return report
        if next_state == "unresolved":
            trigger = "no_executable_query"
            break
        active = next_state

    state.transition_history.append({"scope": scope, "round": state.rounds[scope],
                                     "from": active, "to": "unresolved", "action": "final_resolve",
                                     "trigger": trigger})
    def validate_resolution(value: ResolutionResponse) -> None:
        if value.selected_candidate_id not in final_allowed:
            raise ValueError(f"selected_candidate_id: choose from {sorted(final_allowed)}")
        _check_evidence(value.evidence_ids, state)
        usable = {entry.evidence_id for entry in state.evidence_store if entry.usable}
        if not value.evidence_ids or not set(value.evidence_ids) <= usable:
            raise ValueError(f"evidence_ids: final identity must cite usable evidence from {sorted(usable)}")

    resolution = await complete(
        client, state, system=RESOLVER_SYSTEM, recipient="LLM_A", stage=f"{scope}_final_resolve",
        model=model, temperature=temperature, retries=retries, response_type=ResolutionResponse,
        payload={**payload(), "state": "unresolved", "trigger": trigger, "last_active_state": last_active,
                 "last_assessment": latest.model_dump() if latest else None},
        semantic_validator=validate_resolution)
    audit = ResolutionAudit(last_active_state=last_active, trigger=trigger,
                            selected_candidate_id=resolution.selected_candidate_id,
                            reason=resolution.reason, evidence_ids=resolution.evidence_ids,
                            remaining_limitations=resolution.remaining_limitations)
    report = VerificationReport(
        report_id=f"VR_{scope.upper()}", target_id=target.target_id, scope=scope,
        verification_status="resolved", selected_candidate_id=resolution.selected_candidate_id,
        last_active_state=last_active, supporting_evidence_ids=resolution.evidence_ids,
        unresolved_questions=latest.unresolved_questions if latest else [],
        remaining_limitations=resolution.remaining_limitations, stop_reason=resolution.reason,
        resolution_audit=audit)
    state.verification_reports.append(report)
    state.transition_history.append({"scope": scope, "round": state.rounds[scope], "from": "unresolved",
                                     "to": "resolved", "action": "finish", "resolution_audit": audit.model_dump()})
    return report
