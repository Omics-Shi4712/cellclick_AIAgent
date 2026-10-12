"""Shared type definitions for explicitly versioned CellClick protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ProtocolSpec:
    """A frozen protocol entry exposed through the public registry."""

    version: str
    prompt_version: str
    annotation_candidates: int
    max_rounds: int
    request_retries: int
    max_parallel_clusters: int
    build_graph: Callable[..., Any]
    annotate_cluster: Callable[..., dict[str, Any]]
    protocol_fingerprint: Callable[[], str]
    contract_fingerprint: Callable[[], str]
