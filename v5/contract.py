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


def payload() -> dict[str, object]:
    return {"protocol": PROTOCOL_VERSION, "prompt": PROMPT_VERSION, "request_builder": REQUEST_BUILDER_VERSION,
            "workflow": WORKFLOW_VERSION, "candidates": ANNOTATION_CANDIDATES, "rounds": MAX_ROUNDS,
            "retries": REQUEST_RETRIES, "parallel": MAX_PARALLEL_CLUSTERS,
            "reference_max_default": 3,
            "state_actions": STATE_ACTIONS, "fold_change_epsilon": FOLD_CHANGE_EPSILON,
            "prompts": {key: value for key, value in vars(prompts).items() if key.endswith("_SYSTEM")},
            "response_schemas": [m.model_json_schema() for m in (TaskInput, CandidateProposal, ToulminArgument,
                                                                    ToolRequest, ToolObservation, MessageEnvelope,
                                                                    FinalAnnotation, V5State, SubtypeCandidateResponse,
                                                                    QueryBuilderResponse, VerifierResponse, ResolutionResponse,
                                                                    AdjudicationResponse, MajorCandidateResponse, MarkerAudit)],
            "reference_comparison": "query_expression_all_cells_v3",
            "barcode_policy": "program-internal-target-handle"}


def protocol_fingerprint() -> str:
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def contract_fingerprint() -> str:
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


PROTOCOL_FINGERPRINT = protocol_fingerprint()
CONTRACT_FINGERPRINT = contract_fingerprint()
