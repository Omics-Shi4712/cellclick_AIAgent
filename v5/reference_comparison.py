"""ReferenceComparison GS/RES/EAD/CellScores calculations.

The implementation is deliberately numerical and provider independent. Reference
JSON files provide marker lists; expression vectors are supplied by the trusted
expression layer at runtime.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    den = math.sqrt(sum(float(a) ** 2 for a in left)) * math.sqrt(sum(float(b) ** 2 for b in right))
    return dot / den if den else 0.0


def gene_similarity(query_vectors: Mapping[str, Sequence[float]], reference_gene: str,
                    query_markers: Sequence[str], reference_vectors: Mapping[str, Sequence[float]],
                    query_means: Mapping[str, Sequence[float]], reference_means: Mapping[str, Sequence[float]]) -> float:
    if reference_gene in query_markers:
        return 1.0
    values = []
    for gene in query_markers:
        if gene not in query_vectors or reference_gene not in reference_vectors:
            continue
        if gene not in query_means or reference_gene not in reference_means:
            continue
        values.append(_cosine(query_vectors[gene], reference_vectors[reference_gene]) *
                      _cosine(query_means[gene], reference_means[reference_gene]))
    return sum(values) / len(values) if values else 0.0


def relative_expression_specificity(*, gene: str, query_markers: Sequence[str], cosg_scores: Mapping[str, float],
                                    n: int | None = None) -> float:
    if gene in query_markers:
        return 1.0
    denominator = sum(max(0.0, float(cosg_scores.get(item, 0.0))) for item in query_markers)
    if denominator <= 0 or gene not in cosg_scores:
        return 0.0
    return (n or len(query_markers)) * max(0.0, float(cosg_scores[gene])) / denominator


def expression_abundance_difference(*, gene_expression: float, query_mean_expression: float, k: float = 0.5) -> float:
    if k < 0:
        raise ValueError("k must be non-negative")
    return 1.0 / (1.0 + math.exp(-k * (float(gene_expression) - float(query_mean_expression))))


def cell_scores(*, query_markers: Sequence[str], reference_markers: Sequence[str],
                gs: Mapping[str, float], res: Mapping[str, float],
                ead_by_cell: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    valid = [gene for gene in reference_markers if gene in gs and gene in res and
             all(gene in values for values in ead_by_cell.values())]
    if not valid:
        return {"scores": {}, "valid_markers": [], "coverage": 0.0, "status": "unavailable"}
    scores = {cell: sum(float(gs[g]) * float(res[g]) * float(values[g]) for g in valid) / len(valid)
              for cell, values in ead_by_cell.items()}
    return {"scores": scores, "valid_markers": valid,
            "coverage": len(valid) / len(reference_markers) if reference_markers else 0.0,
            "status": "success"}


def compare_reference(*, query_markers: Sequence[str], reference_markers: Sequence[str],
                      query_vectors: Mapping[str, Sequence[float]], reference_vectors: Mapping[str, Sequence[float]],
                      query_means: Mapping[str, Sequence[float]], reference_means: Mapping[str, Sequence[float]],
                      cosg_scores: Mapping[str, float], cell_expression: Mapping[str, Mapping[str, float]],
                      k: float = 0.5) -> dict[str, Any]:
    valid_reference_markers = [gene for gene in reference_markers if gene in reference_vectors
                               and gene in reference_means and gene in cosg_scores
                               and all(gene in values for values in cell_expression.values())]
    gs = {gene: gene_similarity(query_vectors, gene, query_markers, reference_vectors, query_means, reference_means)
          if gene not in query_markers else 1.0 for gene in valid_reference_markers}
    res = {gene: relative_expression_specificity(gene=gene, query_markers=query_markers,
                                                  cosg_scores=cosg_scores) for gene in valid_reference_markers}
    ead = {cell: {gene: (1.0 if gene in query_markers else expression_abundance_difference(
        gene_expression=values.get(gene, 0.0),
        query_mean_expression=sum(values.get(q, 0.0) for q in query_markers) / len(query_markers), k=k))
            for gene in valid_reference_markers} for cell, values in cell_expression.items()}
    result = cell_scores(query_markers=query_markers, reference_markers=reference_markers, gs=gs, res=res,
                         ead_by_cell=ead)
    missing_reference_vectors = [gene for gene in reference_markers if gene not in reference_vectors]
    limitations = []
    if missing_reference_vectors:
        limitations.append("reference_expression_vectors_unavailable_for:" + ",".join(missing_reference_vectors))
    return {**result, "GS": gs, "RES": res, "EAD": ead, "k": k,
            "calculation_mode": "query_expression_all_cells_v3",
            "limitations": limitations}


__all__ = ["cell_scores", "compare_reference", "expression_abundance_difference",
           "gene_similarity", "relative_expression_specificity"]
