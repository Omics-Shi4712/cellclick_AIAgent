"""Versioned contracts for the CellClick V5 evidence and argument workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskInput(BaseModel):
    task_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    species: str = Field(min_length=1)
    tissue: str = Field(min_length=1)
    ordered_markers: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_markers(self) -> "TaskInput":
        normalized = [" ".join(x.strip().casefold().split()) for x in self.ordered_markers]
        if any(not x for x in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("ordered_markers must be unique and non-empty")
        return self


class MarkerAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gene: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    suggested_use: Literal["primary", "supporting", "review"]
    reason: str = Field(min_length=1)


class MarkerAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audit_id: str = Field(min_length=1)
    marker_assessments: list[MarkerAssessment] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    summary: str = ""


class CandidateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    granularity: Literal["major", "subtype"]
    parent_id: str | None = None
    provisional: bool = False
    lineage: str | None = None
    competing_candidate_ids: list[str] = Field(default_factory=list)
    argument_ids: list[str] = Field(default_factory=list)
    source: Literal["MajorType", "SubType", "revision"]
    cl_id: str | None = None
    mapping_status: str = "unmapped"
    original_labels: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)


class CandidateRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=0, ge=0)
    candidates: list[CandidateProposal] = Field(default_factory=list)
    parent_by_id: dict[str, str | None] = Field(default_factory=dict)
    competing_by_id: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent(self) -> "CandidateRegistry":
        ids = {c.candidate_id for c in self.candidates}
        if len(ids) != len(self.candidates):
            raise ValueError("candidate IDs must be unique")
        for candidate in self.candidates:
            if candidate.parent_id and candidate.parent_id not in ids:
                raise ValueError("candidate parent is not in registry")
            if any(x not in ids for x in candidate.competing_candidate_ids):
                raise ValueError("candidate competition cites unknown candidate")
        return self


class VerificationTarget(BaseModel):
    target_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source_target_id: str = Field(min_length=1)
    cell_barcodes: list[str] = Field(default_factory=list, repr=False)
    cell_set_fingerprint: str = Field(min_length=1)
    supported_parent_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class VerificationQuestion(BaseModel):
    question_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    affected_level: Literal["major", "subtype", "composition"]
    queryable: bool
    status: Literal["open", "answered", "unanswerable"] = "open"
    evidence_ids: list[str] = Field(default_factory=list)
    blocks_acceptance_of: list[str] = Field(default_factory=list)


class ToulminArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    argument_id: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1)
    target_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    claim_type: Literal["identity", "subtype", "composition", "cell_flag"]
    claim: str = Field(min_length=1)
    grounds_evidence_ids: list[str] = Field(default_factory=list)
    warrants: list[str] = Field(default_factory=list)
    backing_ids: list[str] = Field(default_factory=list)
    qualifier_strength: Literal["high", "medium", "low", "unknown"] = "unknown"
    supported_granularity: Literal["major", "subtype", "composition", "unknown"] = "unknown"
    limitations: list[str] = Field(default_factory=list)
    rebuttals: list[str] = Field(default_factory=list)


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    tool_name: Literal["query_expression", "gene_cosine", "compare_reference"]
    target_id: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    argument_id: str | None = None
    warrant_id: str | None = None
    rebuttal_id: str | None = None
    purpose: str = Field(min_length=1)
    discriminating_question: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    observation_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    tool_name: str
    status: Literal["success", "capability_unavailable", "tool_error", "protocol_error"]
    result: dict[str, Any] | None = None
    parameter_hash: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_result(self) -> "ToolObservation":
        if (self.status == "success") != (self.result is not None):
            raise ValueError("successful observations require a result and failures must not have one")
        return self


class EvidenceLedgerEntry(BaseModel):
    evidence_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    kind: Literal["observed", "gene", "cell_set", "derived", "reference", "tool_error"]
    source: str = Field(min_length=1)
    parent_evidence_ids: list[str] = Field(default_factory=list)
    observation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    usable: bool = True


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    scope: Literal["major", "subtype"]
    verification_status: Literal["solved", "resolved"]
    selected_candidate_id: str = Field(min_length=1)
    last_active_state: Literal["multiple", "mixture_cell", "validation"]
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    remaining_limitations: list[str] = Field(default_factory=list)
    stop_reason: str = Field(min_length=1)
    resolution_audit: ResolutionAudit | None = None


class ReferenceComparisonResult(BaseModel):
    comparison_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    candidate_ids: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    reference_version: str = Field(min_length=1)
    calculation_mode: str = Field(min_length=1)
    score_matrix_handle: str = Field(min_length=1)
    scores: dict[str, float] = Field(default_factory=dict)
    decompositions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    valid_markers: dict[str, list[str]] = Field(default_factory=dict)
    coverage: dict[str, float] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class ResolutionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    last_active_state: Literal["multiple", "mixture_cell", "validation"]
    trigger: Literal["max_rounds_reached", "no_executable_query"]
    selected_candidate_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    remaining_limitations: list[str] = Field(default_factory=list)


class FinalAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_status: Literal["success", "failed"]
    annotation_status: Literal["accepted", "major_accepted_subtype_unresolved", "failed"]
    identity: str | None = None
    subtype: str | None = None
    accepted_level: Literal["major", "subtype", "none"]
    selected_candidate_id: str | None = None
    verification_status: Literal["solved", "resolved"] | None = None
    verification_report_id: str | None = None
    cluster_summary: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    accepted_argument_ids: list[str] = Field(default_factory=list)
    resolution_audit: ResolutionAudit | None = None
    stop_reason: str = Field(min_length=1)


class MessageEnvelope(BaseModel):
    schema_version: Literal["cellclick.v5.message.2"] = "cellclick.v5.message.2"
    message_id: str = Field(min_length=1)
    reply_to: str | None = None
    task_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    message_type: str = Field(min_length=1)
    verification_scope: Literal["major", "subtype", "composition"] | None = None
    candidate_registry_version: int = Field(ge=0)
    evidence_snapshot_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class V5State(BaseModel):
    task: TaskInput
    marker_audit: MarkerAudit | None = None
    candidate_registry: CandidateRegistry = Field(default_factory=CandidateRegistry)
    evidence_store: list[EvidenceLedgerEntry] = Field(default_factory=list)
    verification_targets: list[VerificationTarget] = Field(default_factory=list)
    verification_reports: list[VerificationReport] = Field(default_factory=list)
    reference_comparisons: list[ReferenceComparisonResult] = Field(default_factory=list)
    arguments: list[ToulminArgument] = Field(default_factory=list)
    questions: list[VerificationQuestion] = Field(default_factory=list)
    verification_history: list[dict[str, Any]] = Field(default_factory=list)
    transition_history: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[MessageEnvelope] = Field(default_factory=list)
    rounds: dict[str, int] = Field(default_factory=lambda: {"major": 0, "subtype": 0})
    final_result: FinalAnnotation | None = None


class CandidateDraft(BaseModel):
    """LLM-proposed label; the program assigns the stable candidate ID."""
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1)
    granularity: Literal["major", "subtype"]
    parent_id: str | None = None
    competing_candidate_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class CandidateDraftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[CandidateDraft] = Field(min_length=1, max_length=5)


class SubtypeCandidate(BaseModel):
    """A directory-bound subtype proposal."""
    model_config = ConfigDict(extra="forbid")
    cl_id: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class SubtypeCandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[SubtypeCandidate] = Field(default_factory=list, max_length=5)
    limitations: list[str] = Field(default_factory=list)


class MarkerInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gene: str = Field(min_length=1)
    primary_role: Literal["lineage", "function", "low_information"]
    association: str = Field(min_length=1)


class MajorReferenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_rank: int = Field(ge=1)
    label: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class MajorMarkerCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class MajorCandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    marker_interpretations: list[MarkerInterpretation] = Field(default_factory=list)
    selected_reference_candidates: list[MajorReferenceCandidate] = Field(default_factory=list, max_length=5)
    marker_based_candidates: list[MajorMarkerCandidate] = Field(default_factory=list, max_length=5)


class ArgumentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis_id: str = Field(min_length=1)
    claim_type: Literal["identity", "subtype", "composition", "cell_flag"]
    claim: str = Field(min_length=1)
    # Evidence handles are model suggestions. The graph binds them to the live
    # ledger and drops unsupported arguments instead of failing the task.
    grounds_evidence_ids: list[str] = Field(default_factory=list)
    qualifier_strength: Literal["high", "medium", "low", "unknown"] = "unknown"
    supported_granularity: Literal["major", "subtype", "composition", "unknown"] = "unknown"
    limitations: list[str] = Field(default_factory=list)
    rebuttals: list[str] = Field(default_factory=list)


class ArgumentDraftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arguments: list[ArgumentDraft] = Field(default_factory=list)


class ToolScore(BaseModel):
    """A fixed-shape score entry for strict provider schemas."""
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1)
    score: float


class GeneSetDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1)
    genes: list[str] = Field(default_factory=list)


class ComparePairDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_candidate_id: str = Field(min_length=1)
    right_candidate_id: str = Field(min_length=1)


class MarkerSetDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1)
    source: Literal["reference", "llm"]
    reference_label: str | None = None
    genes: list[str] = Field(default_factory=list)


class QueryBuilderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    genes: list[str] = Field(default_factory=list)
    question: str = ""
    gene_sets: list[GeneSetDraft] = Field(default_factory=list)
    compare_pairs: list[ComparePairDraft] = Field(default_factory=list)
    marker_sets: list[MarkerSetDraft] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    no_query_reason: str | None = None


class VerifierResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    next_state: Literal["multiple", "mixture_cell", "validation", "solved", "unresolved"]
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    candidate_id: str | None = None
    validation_passed: bool | None = None
    remaining_limitations: list[str] = Field(default_factory=list)


class ResolutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_candidate_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    remaining_limitations: list[str] = Field(default_factory=list)


class AdjudicationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotation_status: Literal["accepted", "major_accepted_subtype_unresolved"]
    identity: str | None = None
    subtype: str | None = None
    selected_candidate_id: str = Field(min_length=1)
    verification_status: Literal["solved", "resolved"]
    verification_report_id: str = Field(min_length=1)
    cluster_summary: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    accepted_argument_ids: list[str] = Field(default_factory=list)
    stop_reason: str = Field(min_length=1)
