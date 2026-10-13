"""Contracts for the independent CellClick v4.5 evidence-first workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GeneEvidence(BaseModel):
    gene: str = Field(min_length=1)
    availability: Literal["available", "not_measured", "not_computed", "error"]
    pct_target: float | None = Field(default=None, ge=0, le=1)
    pct_background: float | None = Field(default=None, ge=0, le=1)
    delta_pct: float | None = None
    mean_log1p_difference: float | None = None
    pval_adj: float | None = Field(default=None, ge=0, le=1)
    cosg_score: float | None = Field(default=None, ge=0)
    source_id: str | None = None


class EvidencePacket(BaseModel):
    packet_id: str = Field(min_length=1)
    comparison: Literal["target_vs_rest", "target_vs_peers"]
    genes: list[GeneEvidence]
    capability: Literal["matrix", "full_marker_table", "marker_only"]
    limitations: list[str] = Field(default_factory=list)
    cosg_provenance: dict[str, Any] = Field(default_factory=dict)


class GeneQueryArgs(BaseModel):
    genes: list[str] = Field(min_length=1, max_length=64)
    comparison: Literal["target_vs_rest"] = "target_vs_rest"

    @model_validator(mode="after")
    def unique_genes(self) -> "GeneQueryArgs":
        normalized = [gene.strip().casefold() for gene in self.genes]
        if any(not gene for gene in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("gene query must contain unique non-empty genes")
        return self


class ToolRequest(BaseModel):
    request_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    tool_name: Literal["get_marker_statistics"]
    purpose: str = Field(min_length=1)
    discriminating_question: str = Field(min_length=1)
    arguments: GeneQueryArgs


class ToolObservation(BaseModel):
    observation_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    tool_name: str
    status: Literal["success", "capability_unavailable", "tool_error", "protocol_error"]
    result: EvidencePacket | None = None
    parameter_hash: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def result_matches_status(self) -> "ToolObservation":
        if self.status == "success" and self.result is None:
            raise ValueError("successful tool observation requires result")
        if self.status != "success" and self.result is not None:
            raise ValueError("failed tool observation must not contain result")
        return self


class EvidenceLedgerEntry(BaseModel):
    evidence_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    kind: Literal["observed", "reference", "tool_error"]
    source: str = Field(min_length=1)
    observation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ReferenceGeneEvidence(BaseModel):
    gene: str = Field(min_length=1)
    mgc: float | None = None
    pmids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ReferenceCandidate(BaseModel):
    reference_candidate_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    cl_id: str | None = None
    matched_genes: list[str]
    per_gene_mgc: list[ReferenceGeneEvidence]
    supporting_pmids: list[str] = Field(default_factory=list)
    mgc_score: float
    ai_score: float | None = None
    ai_numerator: float | None = Field(default=None, ge=0)
    ai_denominator: float | None = Field(default=None, gt=0)
    combined_score: float | None = Field(default=None, ge=0)
    coverage: float = Field(ge=0, le=1)


class ReferenceMatchPacket(BaseModel):
    reference_version: str = Field(min_length=1)
    reference_sha256: str = Field(min_length=1)
    query_snapshot_id: str = Field(min_length=1)
    query_genes: list[str]
    scoring_mode: Literal["global_mgc_cosg_ai", "global_mgc_unweighted_fallback"]
    candidates: list[ReferenceCandidate]
    limitations: list[str] = Field(default_factory=list)


class CandidateRegistry(BaseModel):
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    reference_candidates: list[ReferenceCandidate] = Field(default_factory=list)


class MarkerSelection(BaseModel):
    selection_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    primary: list[str]
    supplementary: list[str]
    low_support: list[str]
    unavailable: list[str]

    @model_validator(mode="after")
    def disjoint_groups(self) -> "MarkerSelection":
        groups = [self.primary, self.supplementary, self.low_support, self.unavailable]
        flattened = [gene.casefold() for group in groups for gene in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("marker selection groups must be disjoint")
        return self


class QualitySummary(BaseModel):
    quality_id: str = Field(min_length=1)
    flags: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class TaskInput(BaseModel):
    task_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    species: str = Field(min_length=1)
    tissue: str = Field(min_length=1)
    ordered_markers: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def markers_are_nonempty(self) -> "TaskInput":
        if any(not marker.strip() for marker in self.ordered_markers):
            raise ValueError("ordered_markers cannot contain empty values")
        normalized = [" ".join(marker.strip().casefold().split()) for marker in self.ordered_markers]
        if len(normalized) != len(set(normalized)):
            raise ValueError("ordered_markers cannot contain duplicate genes")
        return self


class A1Result(BaseModel):
    task_id: str
    selection_id: str
    evidence_ids: list[str]
    quality_flags: list[str]
    limitations: list[str]
    summary: str = Field(min_length=1)


class Candidate(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    parent_label: str = Field(min_length=1)
    lineage: str = Field(min_length=1)
    granularity: Literal["broad", "subtype"]
    observed_support: list[str]
    subtype_evidence: list[str] = Field(default_factory=list)
    competing_label: str = Field(min_length=1)
    observed_conflicts: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def subtype_has_evidence(self) -> "Candidate":
        if self.granularity == "subtype" and not self.subtype_evidence:
            raise ValueError("subtype candidate requires observed evidence")
        return self


class A2Result(BaseModel):
    task_id: str
    action: Literal["complete", "insufficient_evidence"]
    candidates: list[Candidate]
    verification_questions: list[str]
    summary: str = Field(min_length=1)


class CandidateAssessment(BaseModel):
    hypothesis_id: str
    status: Literal["supported", "uncertain", "conflicted", "insufficient_evidence"]
    supporting_evidence: list[str]
    conflicting_evidence: list[str]
    support_summary: str
    conflict_summary: str


class DiagnosisResult(BaseModel):
    task_id: str
    candidate_assessments: list[CandidateAssessment]
    unresolved_questions: list[str]
    limitations: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    next_action: Literal["plan_queries"]


class QueryPlanResult(BaseModel):
    task_id: str
    tool_requests: list[ToolRequest] = Field(default_factory=list, max_length=2)
    unresolved_questions: list[str]
    limitations: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    next_action: Literal["execute_tools"]


class EvidenceReviewResult(BaseModel):
    task_id: str
    candidate_assessments: list[CandidateAssessment]
    unresolved_questions: list[str]
    limitations: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    next_action: Literal["continue", "finish"]
    stop_reason: Literal[
        "sufficient_discrimination", "no_further_discriminating_queries", "evidence_insufficient",
    ]


class FinalSummaryResult(BaseModel):
    task_id: str
    candidate_assessments: list[CandidateAssessment]
    recommended_hypothesis_id: str | None = None
    unresolved_questions: list[str]
    limitations: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)


class A4ObservedSupportResult(BaseModel):
    """The frozen, broad support set selected before subtype adjudication."""

    task_id: str
    observed_support_ids: list[str]
    unresolved_conflicts: list[str]
    summary: str = Field(min_length=1)


class A4SubtypeEvidenceResult(BaseModel):
    """Subtype-specific evidence selected only from the frozen support set."""

    task_id: str
    subtype_evidence_ids: list[str]
    summary: str = Field(min_length=1)


class A4DecisionResult(BaseModel):
    """Terminal A4 decision; evidence IDs are deliberately absent and immutable."""

    task_id: str
    action: Literal["accepted", "request_subtype", "request_evidence", "unresolved"]
    decision_status: Literal["validated", "unresolved"]
    label: str
    parent_label: str | None = None
    lineage: str | None = None
    granularity: Literal["broad", "subtype", "unknown"]
    confidence: Literal["high", "medium", "low", "not_assessed"]
    sibling_comparison_ids: list[str]
    unresolved_conflicts: list[str]
    next_questions: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def terminal_consistency(self) -> "A4DecisionResult":
        if not self.label.strip():
            raise ValueError("final result requires a recommended label")
        if self.action == "accepted" and self.decision_status != "validated":
            raise ValueError("accepted result requires validated decision_status")
        if self.action != "accepted" and self.decision_status != "unresolved":
            raise ValueError("non-accepted result requires unresolved decision_status")
        if self.action in {"request_evidence", "request_subtype"} and not self.next_questions:
            raise ValueError("request action requires next_questions")
        return self


class A4Result(BaseModel):
    task_id: str
    action: Literal["accepted", "request_subtype", "request_evidence", "unresolved"]
    decision_status: Literal["validated", "unresolved"]
    label: str
    parent_label: str | None = None
    lineage: str | None = None
    granularity: Literal["broad", "subtype", "unknown"]
    confidence: Literal["high", "medium", "low", "not_assessed"]
    observed_support_ids: list[str]
    subtype_evidence_ids: list[str]
    sibling_comparison_ids: list[str]
    unresolved_conflicts: list[str]
    next_questions: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def terminal_consistency(self) -> "A4Result":
        if not self.label.strip():
            raise ValueError("final result requires a recommended label")
        if self.action == "accepted" and self.decision_status != "validated":
            raise ValueError("accepted result requires validated decision_status")
        if self.action != "accepted" and self.decision_status != "unresolved":
            raise ValueError("non-accepted result requires unresolved decision_status")
        if self.action in {"request_evidence", "request_subtype"} and not self.next_questions:
            raise ValueError("request action requires next_questions")
        if self.action == "accepted" and self.granularity == "subtype" and (
            not self.subtype_evidence_ids or not self.sibling_comparison_ids
        ):
            raise ValueError("accepted subtype requires evidence and sibling comparison")
        return self
