"""CellMarker adapter; kept separate from ReferenceComparison scoring."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..reference import DEFAULT_TOP_K, MarkerReferenceRetriever, load_retriever
from ..v4_5.rag import score_reference


def query_cellmarker(markers: Iterable[str], tissue: str, *, cosg_scores: Mapping[str, float],
                     retriever: MarkerReferenceRetriever | None = None, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    retriever = retriever or load_retriever()
    packet = score_reference(markers, tissue, retriever, top_k=top_k, cosg_scores=cosg_scores)
    return packet.model_dump(mode="json")


__all__ = ["query_cellmarker"]
