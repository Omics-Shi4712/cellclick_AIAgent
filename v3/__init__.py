"""CellClick v3 MGC/tissue-embedding annotation protocol."""

from .graph import annotate_cluster, build_graph, ANNOTATION_CANDIDATES, MAX_ROUNDS, REQUEST_RETRIES, MAX_PARALLEL_CLUSTERS, PROTOCOL_VERSION
from .prompts import PROMPT_VERSION
from .contract import contract_fingerprint, protocol_fingerprint, CONTRACT_FINGERPRINT, PROTOCOL_FINGERPRINT
from ..base import ProtocolSpec

V3_PROTOCOL = ProtocolSpec(
    version=PROTOCOL_VERSION, prompt_version=PROMPT_VERSION,
    annotation_candidates=ANNOTATION_CANDIDATES, max_rounds=MAX_ROUNDS,
    request_retries=REQUEST_RETRIES, max_parallel_clusters=MAX_PARALLEL_CLUSTERS,
    build_graph=build_graph, annotate_cluster=annotate_cluster,
    protocol_fingerprint=protocol_fingerprint, contract_fingerprint=contract_fingerprint,
)
