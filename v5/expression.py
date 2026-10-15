"""Trusted expression helpers for V5 tool handlers."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


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
            rows.append({"gene": gene, "available": True,
                         "mean_log1p": round(float(target_values.mean()), 6),
                         "background_mean_log1p": round(float(background_values.mean()), 6),
                         "mean_difference": round(float(target_values.mean() - background_values.mean()), 6),
                         "pct_expressing": round(target_pct, 6),
                         "background_pct_expressing": round(background_pct, 6),
                         "delta_pct": round(target_pct - background_pct, 6)})
        return {"source_row_id": source_row_id, "statistics_layer": layer,
                "cell_count": int(target.sum()), "background_cell_count": int((~target).sum()), "genes": rows}
    finally:
        del adata
