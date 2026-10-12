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
    label: str = Field(min_length=1)
    lineage: str = Field(min_length=1)
    granularity: Literal["broad", "subtype"]
    supporting_markers: list[str]
    observed_conflicts: list[str]
    generic_markers: list[str]
    alternatives: list[str]
    confidence: Literal["high", "medium", "low"]
    rationale: str = Field(min_length=1)

class Judgment(BaseModel):
    decision: Literal["validated", "unresolved"]
    label: str | None = None
    lineage: str = ""
    granularity: Literal["broad", "subtype"] | None = None
    decisive_markers: list[str]
    fatal_conflicts: list[str]
    selection_reason: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
    @model_validator(mode="after")
    def label_for_validated(self):
        if self.decision == "validated" and not self.label:
            raise ValueError("validated judgment requires label")
        return self
