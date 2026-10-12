"""V4 reference scoring adapter over the pinned CellClick reference table."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping

from ..v3.rag import DEFAULT_TOP_K, MarkerReferenceRetriever, default_reference_path, load_retriever
from .models import ReferenceCandidate, ReferenceGeneEvidence, ReferenceMatchPacket

RETRIEVAL_VERSION = "v4-global-mgc-cosg-ai-v1"


def query_snapshot_id(markers: Iterable[str]) -> str:
    value = "\0".join(dict.fromkeys(str(marker).strip() for marker in markers if str(marker).strip()))
    return "QS_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _reference_sha(retriever: object) -> str:
    value = getattr(retriever, "reference_sha256", None) or getattr(retriever, "source_sha256", None)
    return value if isinstance(value, str) and value else "unknown"


def reference_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def score_reference(markers: Iterable[str], tissue: str, retriever: MarkerReferenceRetriever,
                    *, top_k: int = DEFAULT_TOP_K, cosg_scores: Mapping[str, float]) -> ReferenceMatchPacket:
    query = list(dict.fromkeys(str(marker).strip() for marker in markers if str(marker).strip()))
    if top_k < 1:
        raise ValueError("top_k must be positive")
    normalized_scores: dict[str, float] = {}
    for gene, score in cosg_scores.items():
        normalized_gene = _norm(str(gene))
        if not normalized_gene or normalized_gene in normalized_scores:
            raise ValueError("COSG scores must cover unique non-empty query genes")
        value = float(score)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid COSG score for {gene!r}")
        normalized_scores[normalized_gene] = value
    normalized_query = [_norm(gene) for gene in query]
    if set(normalized_query) != set(normalized_scores):
        raise ValueError("COSG scores must cover every submitted marker exactly once")
    denominator = sum(normalized_scores.values())
    if denominator <= 0:
        raise ValueError("COSG total weight must be positive")
    # V3's retriever computes every global candidate before slicing.  Ask it
    # for that full set here so the AI-weighted score, rather than the old
    # MGC-only score, determines V4's top-k boundary.
    candidate_count = len(getattr(retriever, "global_labels", {})) or top_k
    packet = retriever.retrieve(query, tissue, top_k=candidate_count).prompt_dict()
    ranked: list[tuple[float, float, str, dict[str, object], list[dict[str, object]], list[str], float]] = []
    for hit in packet.get("hits", []):
        evidence = hit.get("marker_evidence", hit.get("evidence", []))
        per_gene = []
        numerator = 0.0
        for item in evidence:
            gene = str(item.get("gene", ""))
            if gene:
                normalized_gene = _norm(gene)
                if normalized_gene in normalized_scores:
                    numerator += normalized_scores[normalized_gene]
                per_gene.append(ReferenceGeneEvidence(
                    gene=gene, mgc=float(item["mgc"]) if item.get("mgc") is not None else None,
                    limitations=[] if item.get("mgc") is not None else ["mgc_unavailable"],
                ))
        matched = [str(gene) for gene in hit.get("matched_input_markers", [])]
        ai = numerator / denominator
        gene_score = float(hit.get("gene_score", hit.get("score", 0.0)))
        combined_score = ai * gene_score
        ranked.append((combined_score, gene_score, str(hit.get("cl_id", "")), hit, per_gene, matched, numerator))
    ranked.sort(key=lambda value: (-value[0], -value[1], value[2]))
    candidates: list[ReferenceCandidate] = []
    for index, (combined_score, gene_score, _, hit, per_gene, matched, numerator) in enumerate(ranked[:top_k], 1):
        candidates.append(ReferenceCandidate(
            reference_candidate_id=f"RC{index:02d}",
            label=str(hit.get("label", "")), cl_id=str(hit.get("cl_id")) if hit.get("cl_id") else None,
            matched_genes=matched, per_gene_mgc=per_gene,
            supporting_pmids=[str(pmid) for pmid in hit.get("supporting_pmids", [])],
            mgc_score=gene_score, ai_score=numerator / denominator,
            ai_numerator=numerator, ai_denominator=denominator, combined_score=combined_score,
            coverage=(len(matched) / len(query)) if query else 0.0,
        ))
    return ReferenceMatchPacket(
        reference_version=RETRIEVAL_VERSION, reference_sha256=_reference_sha(retriever),
        query_snapshot_id=query_snapshot_id(query), query_genes=query,
        scoring_mode="global_mgc_cosg_ai", candidates=candidates,
        limitations=["v3_retriever_adapter"],
    )


__all__ = ["DEFAULT_TOP_K", "MarkerReferenceRetriever", "default_reference_path", "load_retriever",
           "reference_sha256", "RETRIEVAL_VERSION", "ReferenceMatchPacket", "score_reference",
           "query_snapshot_id"]
