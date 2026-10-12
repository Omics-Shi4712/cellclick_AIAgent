"""Machine-readable V4 protocol and contract fingerprints."""

from __future__ import annotations

import hashlib
import json

from .graph import (ANNOTATION_CANDIDATES, MAX_PARALLEL_CLUSTERS, MAX_ROUNDS,
                    MAX_TOOL_REQUESTS, MAX_UNIQUE_QUERY_GENES, PROTOCOL_VERSION,
                    REQUEST_RETRIES, WORKFLOW_VERSION)
from .models import (A1Result, A2Result, A4DecisionResult, A4ObservedSupportResult,
                     A4Result, A4SubtypeEvidenceResult, Candidate, CandidateAssessment,
                     DiagnosisResult, EvidenceReviewResult, FinalSummaryResult,
                     QueryPlanResult, TaskInput)
from .prompts import (A1_SYSTEM, A2_SYSTEM, A3_DIAGNOSIS_SYSTEM, A3_EVIDENCE_REVIEW_SYSTEM,
                      A3_FINAL_SUMMARY_SYSTEM, A3_QUERY_PLAN_SYSTEM, A4_DECISION_SYSTEM,
                      A4_OBSERVED_SUPPORT_SYSTEM, A4_SUBTYPE_EVIDENCE_SYSTEM,
                      PROMPT_VERSION, REQUEST_BUILDER_VERSION)


def payload() -> dict[str, object]:
    return {
        "protocol": PROTOCOL_VERSION, "prompt": PROMPT_VERSION,
        "request_builder": REQUEST_BUILDER_VERSION,
        "systems": [A1_SYSTEM, A2_SYSTEM, A3_DIAGNOSIS_SYSTEM, A3_QUERY_PLAN_SYSTEM,
                    A3_EVIDENCE_REVIEW_SYSTEM, A3_FINAL_SUMMARY_SYSTEM, A4_OBSERVED_SUPPORT_SYSTEM,
                    A4_SUBTYPE_EVIDENCE_SYSTEM, A4_DECISION_SYSTEM],
        "response_schemas": [m.model_json_schema() for m in (
            TaskInput, A1Result, A2Result, DiagnosisResult, QueryPlanResult,
            EvidenceReviewResult, FinalSummaryResult, A4ObservedSupportResult,
            A4SubtypeEvidenceResult, A4DecisionResult, A4Result, Candidate, CandidateAssessment,
        )],
        "workflow": WORKFLOW_VERSION, "candidates": ANNOTATION_CANDIDATES,
        "rounds": MAX_ROUNDS, "retries": REQUEST_RETRIES,
        "max_tool_requests_per_round": MAX_TOOL_REQUESTS,
        "max_unique_query_genes": MAX_UNIQUE_QUERY_GENES,
        "response_validation": "program-controlled-phase-and-evidence-integrity-v4.4-staged-a4",
        "parallel": MAX_PARALLEL_CLUSTERS,
        "unresolved_prediction": "recommended-label-with-unresolved-status-v4.4",
        "evidence_policy": "frozen-observed-support-then-subtype-evidence-v4.4",
        "initial_cosg": "fixed-log1p-all-genes-cached-required-v1",
        "statistics_layer": "normalization+log1p",
        "effect_measure": "mean-log1p-target-minus-rest",
    }


def protocol_fingerprint() -> str:
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def contract_fingerprint() -> str:
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


PROTOCOL_FINGERPRINT = protocol_fingerprint()
CONTRACT_FINGERPRINT = contract_fingerprint()
