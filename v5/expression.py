"""Trusted expression helpers for V5 tool handlers."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

FOLD_CHANGE_EPSILON = 1e-9


def context_fingerprint(*, expression_path: str, manifest_path: str, target_id: str,
                         cell_set_fingerprint: str, genes: Iterable[str], parameters: Mapping[str, Any]) -> str:
    value = {"expression_path": expression_path, "manifest_path": manifest_path, "target_id": target_id,
             "cell_set_fingerprint": cell_set_fingerprint, "genes": list(genes), "parameters": dict(parameters)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def query_expression(*, expression: Mapping[str, Mapping[str, float]], cell_barcodes: Iterable[str],
                     genes: Iterable[str]) -> dict[str, Any]:
    cells = list(cell_barcodes)
    requested = list(dict.fromkeys(str(gene) for gene in genes))
    records = {gene: {cell: float(expression.get(cell, {}).get(gene, 0.0)) for cell in cells} for gene in requested}
    return {"genes": requested, "cell_count": len(cells), "values": records,
            "availability": {gene: any(value != 0 for value in values.values()) for gene, values in records.items()}}


def cluster_expression_evidence(*, expression_file: str | Path, manifest_file: str | Path,
                                source_row_id: str, top_n: int = 50) -> dict[str, Any]:
    """Summarize one exported CellClick group without reading any labels.

    The runner export binds a benchmark ``source_row_id`` to an internal group
    ID.  This function uses only that binding, ``cellclick_group_id``, gene
    names, and the exported normalized/log-transformed matrix.  It deliberately
    never reads source annotations or evaluation files.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    expression_path = Path(expression_file).resolve()
    manifest_path = Path(manifest_file).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "cellclick.expression.2":
        raise ValueError("CellClick expression manifest has an unsupported schema")
    if manifest.get("h5ad_path") != str(expression_path):
        raise ValueError("CellClick expression manifest does not bind to the supplied h5ad")
    if manifest.get("groupby") != "cellclick_group_id":
        raise ValueError("CellClick expression manifest has an unexpected group key")
    source_by_group = manifest.get("source_row_id_by_group")
    if not isinstance(source_by_group, dict):
        raise ValueError("CellClick expression manifest lacks source_row_id_by_group")
    groups = [str(group) for group, row_id in source_by_group.items() if row_id == source_row_id]
    if len(groups) != 1:
        raise ValueError(f"Expected exactly one expression group for {source_row_id!r}")

    try:
        import anndata as ad
        import numpy as np
        from scipy import sparse
    except (ImportError, ValueError) as error:
        raise RuntimeError("CellClick V5 expression evidence requires anndata, numpy, and scipy") from error

    adata = ad.read_h5ad(expression_path)
    try:
        layer_name = str(manifest.get("statistics_layer", ""))
        if layer_name != "normalization+log1p" or layer_name not in adata.layers:
            raise ValueError("CellClick expression export lacks normalization+log1p statistics")
        if "cellclick_group_id" not in adata.obs:
            raise ValueError("CellClick expression export lacks cellclick_group_id")
        target = adata.obs["cellclick_group_id"].astype(str).to_numpy() == groups[0]
        target_count = int(target.sum())
        rest_count = int((~target).sum())
        if target_count == 0 or rest_count == 0:
            raise ValueError("CellClick expression group must contain target and background cells")
        matrix = adata.layers[layer_name]
        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
            target_matrix = matrix[target]
            target_mean = np.asarray(target_matrix.mean(axis=0)).ravel()
            rest_mean = np.asarray(matrix[~target].mean(axis=0)).ravel()
            target_pct = np.asarray(target_matrix.getnnz(axis=0)).ravel() / target_count
        else:
            values = np.asarray(matrix)
            target_values = values[target]
            target_mean = target_values.mean(axis=0)
            rest_mean = values[~target].mean(axis=0)
            target_pct = (target_values > 0).mean(axis=0)
        difference = target_mean - rest_mean
        # Stable tie-breaking preserves the export's gene order.
        order = np.argsort(-difference, kind="stable")[:top_n]
        genes = [str(adata.var_names[index]) for index in order]
        return {
            "schema_version": "cellclick.v5.expression-summary.1",
            "source_row_id": source_row_id,
            "statistics_layer": layer_name,
            "cell_count": target_count,
            "background_cell_count": rest_count,
            "top_differential_genes": [
                {"gene": gene, "mean_log1p": round(float(target_mean[index]), 6),
                 "background_mean_log1p": round(float(rest_mean[index]), 6),
                 "mean_difference": round(float(difference[index]), 6),
                 "pct_expressing": round(float(target_pct[index]), 6)}
                for gene, index in zip(genes, order)
            ],
        }
    finally:
        del adata


def marker_expression_evidence(*, expression_file: str | Path, manifest_file: str | Path,
                               source_row_id: str, genes: Iterable[str]) -> dict[str, Any]:
    """Return expression statistics for exactly the submitted genes, in order."""
    requested = list(dict.fromkeys(str(gene).strip() for gene in genes if str(gene).strip()))
    if not requested:
        raise ValueError("genes must contain at least one non-empty marker")
    result = gene_statistics(expression_file=expression_file, manifest_file=manifest_file,
                             source_row_id=source_row_id, genes=requested)
    rows = []
    for row in result["genes"]:
        if not row.get("available"):
            rows.append({"gene": row["gene"], "status": "unavailable", **row})
            continue
        status = "available_positive" if (row.get("pct_expressing", 0) > 0 or
                                           row.get("background_pct_expressing", 0) > 0 or
                                           row.get("mean_log1p", 0) != 0 or
                                           row.get("background_mean_log1p", 0) != 0) else "available_zero"
        rows.append({"gene": row["gene"], "status": status, **row})
    return {**result, "genes": rows}


def gene_statistics(*, expression_file: str | Path, manifest_file: str | Path,
                    source_row_id: str, genes: Iterable[str]) -> dict[str, Any]:
    """Return bounded target/background statistics for requested genes only."""
    requested = list(dict.fromkeys(str(gene).strip() for gene in genes if str(gene).strip()))
    if not requested or len(requested) > 200:
        raise ValueError("query_expression requires between 1 and 200 unique genes")
    expression_path = Path(expression_file).resolve()
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    groups = [str(group) for group, row_id in manifest.get("source_row_id_by_group", {}).items()
              if row_id == source_row_id]
    if manifest.get("h5ad_path") != str(expression_path) or len(groups) != 1:
        raise ValueError("expression evidence binding is invalid")
    try:
        import anndata as ad
        import numpy as np
        from scipy import sparse
    except (ImportError, ValueError) as error:
        raise RuntimeError("CellClick V5 expression evidence requires anndata, numpy, and scipy") from error
    adata = ad.read_h5ad(expression_path)
    try:
        layer = manifest.get("statistics_layer")
        if layer != "normalization+log1p" or layer not in adata.layers or "cellclick_group_id" not in adata.obs:
            raise ValueError("expression export lacks required V5 statistics")
        target = adata.obs["cellclick_group_id"].astype(str).to_numpy() == groups[0]
        if not target.any() or target.all():
            raise ValueError("expression target must have target and background cells")
        index = {str(gene): position for position, gene in enumerate(adata.var_names.astype(str))}
        matrix = adata.layers[layer]
        rows = []
        for gene in requested:
            position = index.get(gene)
            if position is None:
                rows.append({"gene": gene, "available": False})
                continue
            values = matrix[:, position]
            values = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
            target_values, background_values = values[target], values[~target]
            target_pct = float(np.count_nonzero(target_values > 0) / target_values.size)
            background_pct = float(np.count_nonzero(background_values > 0) / background_values.size)
            target_normalized_mean = float(np.expm1(target_values).mean())
            background_normalized_mean = float(np.expm1(background_values).mean())
            fold_change = ((target_normalized_mean + FOLD_CHANGE_EPSILON) /
                           (background_normalized_mean + FOLD_CHANGE_EPSILON))
            rows.append({"gene": gene, "available": True,
                         "mean_log1p": round(float(target_values.mean()), 6),
                         "background_mean_log1p": round(float(background_values.mean()), 6),
                         "mean_difference": round(float(target_values.mean() - background_values.mean()), 6),
                         "pct_expressing": round(target_pct, 6),
                         "background_pct_expressing": round(background_pct, 6),
                         "delta_pct": round(target_pct - background_pct, 6),
                         "fold_change": round(fold_change, 6),
                         "log2_fold_change": round(float(np.log2(fold_change)), 6),
                         "target_normalized_mean": round(target_normalized_mean, 6),
                         "background_normalized_mean": round(background_normalized_mean, 6),
                         "background_zero_expression": background_normalized_mean == 0.0})
        return {"source_row_id": source_row_id, "statistics_layer": layer,
                "cell_count": int(target.sum()), "background_cell_count": int((~target).sum()),
                "fold_change_epsilon": FOLD_CHANGE_EPSILON, "genes": rows}
    finally:
        del adata


def reference_expression_inputs(*, expression_file: str | Path, manifest_file: str | Path,
                                source_row_id: str, genes: Iterable[str]) -> dict[str, Any]:
    """Read real all-cell/cluster vectors from the anonymous V5 export.

    The export contains only normalized expression and anonymous group IDs, so
    this remains label-free while providing the vectors required by GS/RES/EAD.
    """
    requested = list(dict.fromkeys(str(g).strip() for g in genes if str(g).strip()))
    if not requested:
        raise ValueError("reference_expression_inputs requires at least one gene")
    expression_path = Path(expression_file).resolve()
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    if manifest.get("h5ad_path") != str(expression_path):
        raise ValueError("expression manifest does not bind to supplied export")
    groups = sorted(str(group) for group, row in manifest.get("source_row_id_by_group", {}).items()
                    if row == source_row_id)
    if len(groups) != 1:
        raise ValueError("reference expression target binding is invalid")
    try:
        import anndata as ad
        import numpy as np
        from scipy import sparse
    except (ImportError, ValueError) as error:
        raise RuntimeError("reference expression requires anndata, numpy, and scipy") from error
    adata = ad.read_h5ad(expression_path)
    try:
        layer = manifest.get("statistics_layer")
        if layer not in adata.layers or "cellclick_group_id" not in adata.obs:
            raise ValueError("expression export lacks reference comparison layer/group")
        index = {str(g): i for i, g in enumerate(adata.var_names.astype(str))}
        labels = adata.obs["cellclick_group_id"].astype(str).to_numpy()
        matrix = adata.layers[layer]
        all_groups = sorted(set(labels))
        target_mask = labels == groups[0]
        if not target_mask.any() or target_mask.all() or len(all_groups) < 2:
            raise ValueError("reference comparison requires target and background groups")
        vectors: dict[str, list[float]] = {}
        means: dict[str, list[float]] = {}
        cell_expression: dict[str, dict[str, float]] = {}
        missing: list[str] = []
        for gene in requested:
            pos = index.get(gene)
            if pos is None:
                missing.append(gene)
                continue
            values = matrix[:, pos]
            values = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
            vectors[gene] = [float(x) for x in values]
            means[gene] = [float(values[labels == group].mean()) for group in all_groups]
            cell_expression[gene] = {f"cell_{i:09d}": float(value)
                                     for i, value in enumerate(values[target_mask])}
        target_cell_ids = [f"cell_{i:09d}" for i in range(int(target_mask.sum()))]
        by_cell = {cell: {gene: cell_expression[gene][cell] for gene in vectors}
                   for cell in target_cell_ids}
        return {"query_vectors": vectors, "query_means": means,
                "cell_expression": by_cell, "groups": all_groups,
                "target_group": groups[0], "missing_genes": missing,
                "available_genes": sorted(vectors), "cell_count": len(target_cell_ids)}
    finally:
        del adata


def gene_cosine_statistics(*, expression_file: str | Path, manifest_file: str | Path,
                           source_row_id: str, gene_sets: list[dict[str, Any]],
                           compare_pairs: list[list[str]]) -> dict[str, Any]:
    """Calculate target-cell cosine and joint detection without inferring mixture."""
    import math

    candidate_ids = [str(item.get("candidate_id", "")).strip() for item in gene_sets]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)) or any(not value for value in candidate_ids):
        raise ValueError("gene_cosine gene_sets require unique non-empty candidate_id values")
    normalized_sets: dict[str, list[str]] = {}
    for item, candidate_id in zip(gene_sets, candidate_ids):
        genes = list(dict.fromkeys(str(gene).strip() for gene in item.get("genes", []) if str(gene).strip()))
        if not genes:
            raise ValueError("gene_cosine each gene set requires at least one gene")
        normalized_sets[candidate_id] = genes
    normalized_pairs = []
    for pair in compare_pairs:
        if len(pair) != 2 or pair[0] not in normalized_sets or pair[1] not in normalized_sets or pair[0] == pair[1]:
            raise ValueError("gene_cosine compare_pairs must cite two distinct supplied candidate IDs")
        normalized_pairs.append([pair[0], pair[1]])
    if not normalized_pairs:
        raise ValueError("gene_cosine requires at least one compare pair")

    genes = list(dict.fromkeys(gene for values in normalized_sets.values() for gene in values))
    vectors = reference_expression_inputs(expression_file=expression_file, manifest_file=manifest_file,
                                          source_row_id=source_row_id, genes=genes)
    available = {gene: [values[gene] for values in vectors["cell_expression"].values()]
                 for gene in vectors["available_genes"]}
    missing = set(vectors["missing_genes"])
    gene_detection = {
        gene: {"available": gene in available,
               "detected_cells": (sum(value > 0 for value in available[gene]) if gene in available else None),
               "pct_expressing": (round(sum(value > 0 for value in available[gene]) / len(available[gene]), 6)
                                  if gene in available and available[gene] else None)}
        for gene in genes
    }

    def pair_result(left: str, right: str) -> dict[str, Any]:
        if left in missing or right in missing:
            return {"left_gene": left, "right_gene": right, "computable": False,
                    "reason": "missing_gene", "cosine": None}
        a, b = available[left], available[right]
        left_norm = math.sqrt(sum(value * value for value in a))
        right_norm = math.sqrt(sum(value * value for value in b))
        if left_norm == 0 or right_norm == 0:
            return {"left_gene": left, "right_gene": right, "computable": False,
                    "reason": "zero_vector", "cosine": None}
        joint = sum(1 for x, y in zip(a, b) if x > 0 and y > 0)
        return {"left_gene": left, "right_gene": right, "computable": True,
                "cosine": round(sum(x * y for x, y in zip(a, b)) / (left_norm * right_norm), 6),
                "left_detected": sum(value > 0 for value in a),
                "right_detected": sum(value > 0 for value in b),
                "joint_detected": joint,
                "joint_detected_fraction": round(joint / len(a), 6) if a else 0.0}

    within = {candidate_id: [pair_result(genes[i], genes[j])
                             for i in range(len(genes)) for j in range(i + 1, len(genes))]
              for candidate_id, genes in normalized_sets.items()}
    across = []
    for left_id, right_id in normalized_pairs:
        across.append({"left_candidate_id": left_id, "right_candidate_id": right_id,
                       "gene_pairs": [pair_result(left, right)
                                      for left in normalized_sets[left_id]
                                      for right in normalized_sets[right_id]]})
    zero = [gene for gene in genes if gene in available and not any(available[gene])]
    return {"source_row_id": source_row_id, "statistics_layer": "normalization+log1p",
            "cell_count": vectors["cell_count"], "gene_sets": normalized_sets,
            "available_genes": sorted(available), "missing_genes": sorted(missing),
            "zero_vector_genes": sorted(zero), "gene_detection": gene_detection,
            "within_group": within, "cross_group": across,
            "limitations": ["cosine_is_expression_evidence_not_a_mixture_call"]}
