from __future__ import annotations
import hashlib,json
from .graph import (ANNOTATION_CANDIDATES, CANDIDATE_ROLES, MAX_PARALLEL_CLUSTERS,
                    MAX_ROUNDS, PROTOCOL_VERSION, REQUEST_RETRIES,
                    UNRESOLVED_PREDICTION_POLICY, WORKFLOW_VERSION)
from .prompts import PROMPT_VERSION,AUDIT_SYSTEM,CANDIDATE_SYSTEM,JUDGE_SYSTEM,REQUEST_BUILDER_VERSION
from .rag import DEFAULT_TOP_K, RETRIEVAL_VERSION
def payload(): return {"protocol":PROTOCOL_VERSION,"prompt":PROMPT_VERSION,"builder":REQUEST_BUILDER_VERSION,"systems":[AUDIT_SYSTEM,CANDIDATE_SYSTEM,JUDGE_SYSTEM],"candidates":ANNOTATION_CANDIDATES,"rounds":MAX_ROUNDS,"retries":REQUEST_RETRIES,"parallel":MAX_PARALLEL_CLUSTERS,"workflow":WORKFLOW_VERSION,"candidate_roles":CANDIDATE_ROLES,"unresolved_prediction":UNRESOLVED_PREDICTION_POLICY,"retrieval_version":RETRIEVAL_VERSION,"default_rag_top_k":DEFAULT_TOP_K}
def protocol_fingerprint(): return hashlib.sha256(json.dumps(payload(),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def contract_fingerprint(): return hashlib.sha256(json.dumps(payload(),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
PROTOCOL_FINGERPRINT=protocol_fingerprint(); CONTRACT_FINGERPRINT=contract_fingerprint()
