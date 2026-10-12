"""Static, bounded V4 tool registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .expression import NORMALIZED_LOG_LAYER
from .models import EvidencePacket, GeneEvidence, TaskInput, ToolObservation, ToolRequest


def _parameter_hash(request: ToolRequest) -> str:
    return query_fingerprint(request)


def _query_fingerprint_payload(request: ToolRequest) -> dict[str, Any]:
    """Return only the deterministic inputs that affect a tool result."""
    return {
        "task_id": request.task_id,
        "tool_name": request.tool_name,
        "comparison": request.arguments.comparison,
        "genes": sorted(" ".join(gene.strip().casefold().split())
                         for gene in request.arguments.genes),
    }


def query_fingerprint(request: ToolRequest) -> str:
    value = json.dumps(_query_fingerprint_payload(request), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def _query_h5ad(task: TaskInput, request: ToolRequest, expression_file: str | Path,
                expression_manifest: str | Path) -> EvidencePacket:
    import anndata as ad
    expression_path = Path(expression_file).resolve()
    manifest = json.loads(Path(expression_manifest).read_text(encoding="utf-8"))
    if Path(str(manifest.get("h5ad_path", ""))).resolve() != expression_path:
        raise ValueError("tool expression manifest does not match expression_file")
    mapping = manifest.get("source_row_id_by_group", {})
    groups = [str(group) for group, row_id in mapping.items() if str(row_id) == task.task_id]
    if len(groups) != 1:
        raise ValueError("tool task is not bound to exactly one expression group")
    groupby = str(manifest["groupby"])
    statistics_layer = str(manifest["statistics_layer"])
    if statistics_layer != NORMALIZED_LOG_LAYER:
        raise ValueError(f"tool statistics layer must be {NORMALIZED_LOG_LAYER!r}")
    adata = ad.read_h5ad(expression_path)
    if groupby not in adata.obs:
        raise ValueError("tool groupby column is absent")
    if statistics_layer not in adata.layers:
        raise ValueError("tool statistics layer is absent")
    matrix = adata.layers[statistics_layer]
    if matrix.shape != adata.shape:
        raise ValueError("tool matrix shape does not match h5ad")
    target = adata.obs[groupby].astype(str).to_numpy() == groups[0]
    if not target.any() or target.all():
        raise ValueError("tool target must have both target and background cells")
    positions = {str(gene): index for index, gene in enumerate(adata.var_names.astype(str))}
    records = []
    for gene in request.arguments.genes:
        if gene not in positions:
            records.append(GeneEvidence(gene=gene, availability="not_measured"))
            continue
        position = positions[gene]
        target_values = matrix[target, position]
        background_values = matrix[~target, position]
        if hasattr(target_values, "toarray"):
            target_values = target_values.toarray().ravel()
            background_values = background_values.toarray().ravel()
        else:
            target_values = np.asarray(target_values).ravel()
            background_values = np.asarray(background_values).ravel()
        pct_target = float(np.count_nonzero(target_values > 0) / target_values.size)
        pct_background = float(np.count_nonzero(background_values > 0) / background_values.size)
        records.append(GeneEvidence(
            gene=gene, availability="available", pct_target=pct_target,
            pct_background=pct_background, delta_pct=pct_target - pct_background,
            mean_log1p_difference=float(target_values.mean() - background_values.mean()),
            source_id=f"{expression_path}:{statistics_layer}",
        ))
    return EvidencePacket(packet_id=f"EP_{request.request_id}", comparison=request.arguments.comparison,
                           genes=records, capability="matrix")


def execute_tool(task: TaskInput, request: ToolRequest, *, expression_file: str | Path,
                 expression_manifest: str | Path) -> ToolObservation:
    if request.task_id != task.task_id:
        return ToolObservation(observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                               task_id=task.task_id, tool_name=request.tool_name, status="protocol_error",
                               parameter_hash=_parameter_hash(request), limitations=["task_id_mismatch"])
    try:
        result = _query_h5ad(task, request, expression_file, expression_manifest)
    except Exception as error:
        return ToolObservation(observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                               task_id=task.task_id, tool_name=request.tool_name, status="tool_error",
                               parameter_hash=_parameter_hash(request), limitations=[f"{type(error).__name__}:{error}"])
    return ToolObservation(observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                           task_id=task.task_id, tool_name=request.tool_name, status="success", result=result,
                           parameter_hash=_parameter_hash(request))
