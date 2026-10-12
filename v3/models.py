"""Response contracts for CellClick v3's evidence-comparison workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Audit(BaseModel):
    candidate_lineages: list[str] = Field(min_length=1)
    lineage_evidence: list[str]
    observed_conflicts: list[str]
    generic_markers: list[str]
    missing_expected_markers: list[str]
    confidence: Literal["high", "medium", "low"]
    audit_summary: str = Field(min_length=1)


class Candidate(BaseModel):
    role: Literal["lineage-first", "subtype-conservative", "adversarial-conflict"]
    label: str = Field(min_length=1)
    parent_label: str = Field(min_length=1)
    lineage: str = Field(min_length=1)
    granularity: Literal["broad", "subtype"]
    supporting_markers: list[str]
    subtype_specific_markers: list[str]
    observed_conflicts: list[str]
    generic_markers: list[str]
    competing_label: str = Field(min_length=1)
    competing_markers: list[str]
    confidence: Literal["high", "medium", "low"]
    rationale: str = Field(min_length=1)
    retrieval_ranking: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def subtype_requires_specific_evidence(self) -> "Candidate":
        if self.granularity == "subtype" and not self.subtype_specific_markers:
            raise ValueError("subtype candidate requires observed discriminating evidence")
        return self


class Judgment(BaseModel):
    decision: Literal["validated", "unresolved"]
    label: str | None = None
    lineage: str = ""
    granularity: Literal["broad", "subtype"] | None = None
    winner_label: str = Field(min_length=1)
    runner_up_label: str = Field(min_length=1)
    winner_support: list[str]
    runner_up_support: list[str]
    decisive_markers: list[str]
    unresolved_conflicts: list[str]
    fatal_conflicts: list[str]
    specificity_justification: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
    retrieval_ranking: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> "Judgment":
        if self.decision == "validated" and not self.label:
            raise ValueError("validated judgment requires label")
        if self.decision == "validated" and self.fatal_conflicts:
            raise ValueError("validated judgment cannot retain fatal conflicts")
        if self.confidence == "high" and self.unresolved_conflicts:
            raise ValueError("high confidence cannot retain unresolved conflicts")
        if self.granularity == "subtype" and not self.specificity_justification.strip():
            raise ValueError("subtype judgment requires specificity justification")
        return self
