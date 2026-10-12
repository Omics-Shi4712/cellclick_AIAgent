"""Frozen Pydantic response models for CellClick v1."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Candidate(BaseModel):
    candidate_id: str
    label: str = Field(min_length=1)
    supporting_markers: list[str]
    conflicting_markers: list[str]
    alternatives: list[str]
    rationale: str = Field(min_length=1)


class CandidateReview(BaseModel):
    candidate_id: str
    valid: bool
    issues: list[str]


class ValidationResult(BaseModel):
    decision: Literal["pass", "revise"]
    candidate_reviews: list[CandidateReview] = Field(min_length=5, max_length=5)
    selected_candidate_id: str | None = None
    selection_reason: str = ""
    revision_feedback: str | None = None

    @model_validator(mode="after")
    def decision_fields_are_complete(self) -> "ValidationResult":
        if self.decision == "pass" and (not self.selected_candidate_id or not self.selection_reason.strip()):
            raise ValueError("pass requires selected_candidate_id and selection_reason")
        if self.decision == "revise" and (not self.revision_feedback or not self.revision_feedback.strip()):
            raise ValueError("revise requires revision_feedback")
        return self


class FormattingResult(BaseModel):
    status: Literal["validated", "unresolved"]
    prediction: str | None = None
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def output_matches_status(self) -> "FormattingResult":
        if self.status == "validated" and (not self.prediction or not self.prediction.strip()):
            raise ValueError("validated formatting output requires prediction")
        if self.status == "unresolved" and self.prediction is not None:
            raise ValueError("unresolved formatting output must not contain prediction")
        return self

