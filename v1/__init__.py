"""Complete, frozen CellClick v1 protocol package."""

from __future__ import annotations

from ..base import ProtocolSpec
from .contract import (
    CONTRACT_FINGERPRINT,
    PROTOCOL_FINGERPRINT,
    contract_fingerprint,
    contract_payload,
    protocol_fingerprint,
)
from .graph import (
    ANNOTATION_CANDIDATES,
    MAX_PARALLEL_CLUSTERS,
    MAX_ROUNDS,
    PROTOCOL_VERSION,
    REQUEST_RETRIES,
    annotate_cluster,
    build_graph,
)
from .prompts import PROMPT_VERSION


V1_PROTOCOL = ProtocolSpec(
    version=PROTOCOL_VERSION,
    prompt_version=PROMPT_VERSION,
    annotation_candidates=ANNOTATION_CANDIDATES,
    max_rounds=MAX_ROUNDS,
    request_retries=REQUEST_RETRIES,
    max_parallel_clusters=MAX_PARALLEL_CLUSTERS,
    build_graph=build_graph,
    annotate_cluster=annotate_cluster,
    protocol_fingerprint=protocol_fingerprint,
    contract_fingerprint=contract_fingerprint,
)

__all__ = (
    "ANNOTATION_CANDIDATES",
    "CONTRACT_FINGERPRINT",
    "MAX_ROUNDS",
    "MAX_PARALLEL_CLUSTERS",
    "PROMPT_VERSION",
    "PROTOCOL_FINGERPRINT",
    "PROTOCOL_VERSION",
    "REQUEST_RETRIES",
    "V1_PROTOCOL",
    "annotate_cluster",
    "build_graph",
    "contract_fingerprint",
    "contract_payload",
    "protocol_fingerprint",
)
