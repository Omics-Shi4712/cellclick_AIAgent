"""Stable protocol and full behavioral contract fingerprints for CellClick v1."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .fallback import FALLBACK_POLICY
from .graph import (
    ANNOTATION_CANDIDATES,
    FORMATTING_VERSION,
    GRAPH_VERSION,
    MAX_PARALLEL_CLUSTERS,
    MAX_ROUNDS,
    PROTOCOL_VERSION,
    REQUEST_RETRIES,
    candidate_schema,
    validation_schema,
)
from .prompts import (
    ANNOTATION_SYSTEM,
    FORMATTING_SYSTEM,
    PROMPT_VERSION,
    REQUEST_BUILDER_VERSION,
    VALIDATION_SYSTEM,
)
from .retry import REPAIR_PROMPT, RETRY_BACKOFF_BASE_SECONDS, RETRY_POLICY_VERSION


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def protocol_payload() -> dict[str, Any]:
    """Return the historical coarse payload without changing its old hash."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "annotation_candidates": ANNOTATION_CANDIDATES,
        "max_rounds": MAX_ROUNDS,
        "annotation_system": ANNOTATION_SYSTEM,
        "validation_system": VALIDATION_SYSTEM,
        "formatting": FORMATTING_VERSION,
        "fallback_policy": FALLBACK_POLICY,
    }


def protocol_fingerprint() -> str:
    # Historical code used json.dumps(sort_keys=True) with default separators.
    encoded = json.dumps(protocol_payload(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract_payload() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompts": {
            "annotation_system": ANNOTATION_SYSTEM,
            "validation_system": VALIDATION_SYSTEM,
            "formatting_system": FORMATTING_SYSTEM,
        },
        "request_builders": {
            "annotation": REQUEST_BUILDER_VERSION,
            "validation": REQUEST_BUILDER_VERSION,
        },
        "schemas": {
            "candidate": candidate_schema(),
            "validation": validation_schema(),
        },
        "graph": {
            "version": GRAPH_VERSION,
            "nodes": ["annotation_batch", "validation", "formatting"],
            "edges": [
                ["START", "annotation_batch"],
                ["annotation_batch", "validation"],
                ["validation", "annotation_batch|formatting"],
                ["formatting", "END"],
            ],
            "annotation_candidates": ANNOTATION_CANDIDATES,
            "max_rounds": MAX_ROUNDS,
        },
        "retry": {
            "version": RETRY_POLICY_VERSION,
            "repair_prompt": REPAIR_PROMPT,
            "backoff_base_seconds": RETRY_BACKOFF_BASE_SECONDS,
            "request_retries": REQUEST_RETRIES,
        },
        "cluster_execution": {"max_parallel_clusters": MAX_PARALLEL_CLUSTERS},
        "fallback_policy": FALLBACK_POLICY,
        "formatting": FORMATTING_VERSION,
    }


def contract_fingerprint() -> str:
    return _sha256(contract_payload())


PROTOCOL_FINGERPRINT = protocol_fingerprint()
CONTRACT_FINGERPRINT = contract_fingerprint()
