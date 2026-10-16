"""V5 protocol and contract fingerprints."""
from __future__ import annotations

import hashlib
import json

from .graph import (ANNOTATION_CANDIDATES, MAX_PARALLEL_CLUSTERS, MAX_ROUNDS, PROTOCOL_VERSION,
                    REQUEST_RETRIES, WORKFLOW_VERSION)
from .models import (CandidateProposal, FinalAnnotation, MessageEnvelope, TaskInput, ToulminArgument,
                     ToolObservation, ToolRequest, V5State, SubtypeCandidateResponse)
from .prompts import PROMPT_VERSION, REQUEST_BUILDER_VERSION
from . import prompts
from .models import (QueryBuilderResponse, VerifierResponse, ResolutionResponse,
                     AdjudicationResponse, MajorCandidateResponse, MarkerAudit)
from .verifier import STATE_ACTIONS
from .expression import FOLD_CHANGE_EPSILON
from .context import (CONTEXT_SCHEMA_VERSION, COSINE_SUMMARY_VERSION,
                      EXPRESSION_REPRESENTATIVE_LIMIT, LLM_CONTEXT_CONTRACT,
                      REPRESENTATIVE_PAIR_LIMIT)


def payload() -> dict[str, object]:
    return {"protocol": PROTOCOL_VERSION, "prompt": PROMPT_VERSION, "request_builder": REQUEST_BUILDER_VERSION,
            "workflow": WORKFLOW_VERSION, "candidates": ANNOTATION_CANDIDATES, "rounds": MAX_ROUNDS,
            "retries": REQUEST_RETRIES, "parallel": MAX_PARALLEL_CLUSTERS,
            "reference_max_default": 3,
            "execution": {"resources": "dataset-memory-v1", "statistics": "export-dtype-lazy-v1",
                          "cosine_reference": "float64-vectorized-v1",
                          "retry_policy": "classified-shared-budget-v1", "sdk_retries": 0,
                          "connect_timeout": 10, "read_timeout": 600, "attempt_timeout": 610},
            "state_actions": STATE_ACTIONS, "fold_change_epsilon": FOLD_CHANGE_EPSILON,
            "prompts": {key: value for key, value in vars(prompts).items() if key.endswith("_SYSTEM")},
            "response_schemas": [m.model_json_schema() for m in (TaskInput, CandidateProposal, ToulminArgument,
                                                                    ToolRequest, ToolObservation, MessageEnvelope,
                                                                    FinalAnnotation, V5State, SubtypeCandidateResponse,
                                                                    QueryBuilderResponse, VerifierResponse, ResolutionResponse,
                                                                    AdjudicationResponse, MajorCandidateResponse, MarkerAudit)],
            "reference_comparison": "query_expression_all_cells_v3",
            "llm_context": {"schema_version": CONTEXT_SCHEMA_VERSION,
                            "cosine_summary_version": COSINE_SUMMARY_VERSION,
                            "cosine_representative_pair_limit": REPRESENTATIVE_PAIR_LIMIT,
                            "expression_representative_limit": EXPRESSION_REPRESENTATIVE_LIMIT,
                            "quantile_method": "linear", "numeric_precision": 6,
                            "gene_cosine_artifact_schema": "cellclick.v5.gene_cosine.1",
                            "contract": LLM_CONTEXT_CONTRACT},
            "barcode_policy": "program-internal-target-handle"}


def protocol_fingerprint() -> str:
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def contract_fingerprint() -> str:
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


PROTOCOL_FINGERPRINT = protocol_fingerprint()
CONTRACT_FINGERPRINT = contract_fingerprint()
