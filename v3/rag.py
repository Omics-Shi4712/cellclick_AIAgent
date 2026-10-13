"""Compatibility exports for the shared CellClick marker retriever.

V3 historically owned this module. The implementation now lives in
cellclick.reference so V3, V4, and V4.5 use the same reference-table
loader and scoring semantics.
"""

from ..reference import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    RETRIEVAL_VERSION,
    MarkerReferenceRetriever,
    RetrievalHit,
    RetrievalPacket,
    _REQUIRED,
    _norm,
    _sha,
    default_reference_path,
    load_retriever,
    select_tissues,
)

__all__ = [
    "DEFAULT_TOP_K", "EMBEDDING_MODEL", "RETRIEVAL_VERSION",
    "MarkerReferenceRetriever", "RetrievalHit", "RetrievalPacket",
    "default_reference_path", "load_retriever", "select_tissues",
]
