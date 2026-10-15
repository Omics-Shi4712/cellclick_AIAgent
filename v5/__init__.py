"""Independent CellClick V5 protocol."""
from ..base import ProtocolSpec
from .contract import CONTRACT_FINGERPRINT, PROTOCOL_FINGERPRINT, contract_fingerprint, protocol_fingerprint
from .graph import (ANNOTATION_CANDIDATES, MAX_PARALLEL_CLUSTERS, MAX_ROUNDS, PROTOCOL_VERSION,
                    REQUEST_RETRIES, annotate_cluster, build_graph)
from .prompts import PROMPT_VERSION

V5_PROTOCOL = ProtocolSpec(version=PROTOCOL_VERSION, prompt_version=PROMPT_VERSION,
                           annotation_candidates=ANNOTATION_CANDIDATES, max_rounds=MAX_ROUNDS,
                           request_retries=REQUEST_RETRIES, max_parallel_clusters=MAX_PARALLEL_CLUSTERS,
                           build_graph=build_graph, annotate_cluster=annotate_cluster,
                           protocol_fingerprint=protocol_fingerprint, contract_fingerprint=contract_fingerprint)

__all__ = ["V5_PROTOCOL", "annotate_cluster", "build_graph"]
