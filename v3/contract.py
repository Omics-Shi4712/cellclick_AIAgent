import hashlib
import json
from .graph import (ANNOTATION_CANDIDATES, CANDIDATE_ROLES, MAX_PARALLEL_CLUSTERS,
                    MAX_ROUNDS, PROTOCOL_VERSION, REQUEST_RETRIES,
                    UNRESOLVED_PREDICTION_POLICY, WORKFLOW_VERSION)
from .prompts import (PROMPT_VERSION, REQUEST_BUILDER_VERSION, AUDIT_SYSTEM,
                      CANDIDATE_SYSTEM, JUDGE_SYSTEM, ROLE_INSTRUCTIONS)
from .rag import DEFAULT_TOP_K, RETRIEVAL_VERSION
from .models import Audit, Candidate, Judgment

def payload():
    return {"protocol": PROTOCOL_VERSION, "prompt": PROMPT_VERSION,
            "request_builder": REQUEST_BUILDER_VERSION,
            "systems": [AUDIT_SYSTEM, CANDIDATE_SYSTEM, JUDGE_SYSTEM],
            "role_instructions": ROLE_INSTRUCTIONS,
            "response_schemas": [model.model_json_schema() for model in (Audit, Candidate, Judgment)],
            "validation_version": "input-markers-consistent-decision-complete-retrieval-ranking-v3",
            "dataset_context": "label-free-target-marker-prevalence-and-overlap",
            "candidates": ANNOTATION_CANDIDATES, "rounds": MAX_ROUNDS,
            "retries": REQUEST_RETRIES, "parallel": MAX_PARALLEL_CLUSTERS,
            "workflow": WORKFLOW_VERSION, "candidate_roles": CANDIDATE_ROLES,
            "unresolved_prediction": UNRESOLVED_PREDICTION_POLICY,
            "retrieval_version": RETRIEVAL_VERSION, "default_rag_top_k": DEFAULT_TOP_K,
            "tissue_context": "exact-match-metadata-only",
            "retrieval_scoring": "sum_sqrt_global_document_mgc"}
def protocol_fingerprint():
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def contract_fingerprint():
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
PROTOCOL_FINGERPRINT = protocol_fingerprint()
CONTRACT_FINGERPRINT = contract_fingerprint()
