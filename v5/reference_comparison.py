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
    import numpy as np
    cells = list(ead_by_cell)
    weights = np.asarray([float(gs[g]) * float(res[g]) for g in valid], dtype=np.float64)
    matrix = np.asarray([[float(ead_by_cell[cell][g]) for g in valid] for cell in cells], dtype=np.float64)
    scores = dict(zip(cells, (matrix @ weights / len(valid)).tolist())) if cells else {}
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
    import numpy as np
    gs = {}
    query_genes = [g for g in query_markers if g in query_vectors and g in query_means]
    def similarities(vectors, references):
        left = np.asarray([vectors[g] for g in query_genes], dtype=np.float64)
        right = np.asarray([references[g] for g in valid_reference_markers], dtype=np.float64)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError("ragged expression vectors")
        denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
        return np.divide(left @ right.T, denominator, out=np.zeros(denominator.shape), where=denominator != 0)
    if query_genes and valid_reference_markers:
        try:
            scores = (similarities(query_vectors, reference_vectors) *
                      similarities(query_means, reference_means)).mean(axis=0)
            gs = {gene: float(scores[i]) for i, gene in enumerate(valid_reference_markers)}
        except ValueError:
            gs = {gene: gene_similarity(query_vectors, gene, query_markers, reference_vectors, query_means, reference_means)
                  for gene in valid_reference_markers}
    else:
        gs = {gene: 0.0 for gene in valid_reference_markers}
    gs.update({gene: 1.0 for gene in valid_reference_markers if gene in query_markers})
    denominator = sum(max(0., float(cosg_scores.get(g, 0.))) for g in query_markers)
    specificity = np.asarray([max(0., float(cosg_scores[g])) for g in valid_reference_markers], dtype=np.float64)
    specificity = len(query_markers) * specificity / denominator if denominator > 0 else np.zeros_like(specificity)
    res = {gene: 1. if gene in query_markers else float(specificity[i])
           for i, gene in enumerate(valid_reference_markers)}
    cells = list(cell_expression)
    ead = {cell: {} for cell in cells}
    nonoverlap = [gene for gene in valid_reference_markers if gene not in query_markers]
    if cells and nonoverlap:
        if k < 0:
            raise ValueError("k must be non-negative")
        means = np.asarray([sum(cell_expression[cell].get(q, 0.0) for q in query_markers) / len(query_markers)
                            for cell in cells], dtype=np.float64)
        values = np.asarray([[cell_expression[cell][gene] for gene in nonoverlap] for cell in cells], dtype=np.float64)
        exponent = -k * (values - means[:, None])
        if np.any(exponent > math.log(float.fromhex('0x1.fffffffffffffp+1023'))):
            raise OverflowError("math range error")
        abundance = 1.0 / (1.0 + np.exp(exponent))
        for i, cell in enumerate(cells):
            ead[cell] = {gene: float(abundance[i, j]) for j, gene in enumerate(nonoverlap)}
    for cell in cells:
        ead[cell] = {gene: 1.0 if gene in query_markers else ead[cell][gene] for gene in valid_reference_markers}
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
