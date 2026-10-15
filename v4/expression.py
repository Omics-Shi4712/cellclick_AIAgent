"""Trusted expression binding and initial marker statistics for V4."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .models import EvidencePacket, GeneEvidence, MarkerSelection, QualitySummary, TaskInput


NORMALIZED_LOG_LAYER = "normalization+log1p"
COSG_PARAMETERS = {
    "mu": 1,
    "remove_lowly_expressed": False,
    "expressed_pct": 0.1,
    "reference": "rest",
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _cache_path(work_dir: Path | None, cache_key: str) -> Path | None:
    if work_dir is None:
        return None
    return work_dir / "cosg_cache" / f"{cache_key}.json"


def _cosg_scores(
    *, adata: Any, manifest: dict[str, Any], expression_path: Path, manifest_path: Path,
    work_dir: Path | None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Return submitted-marker COSG scores keyed by source row id.

    COSG is deliberately run over the complete filtered gene universe.  Only
    the frozen submitted-marker snapshot is persisted, so cache files cannot
    expose group labels or an unnecessary full expression-derived ranking.
    """
    if NORMALIZED_LOG_LAYER not in adata.layers:
        raise ValueError(f"CellClick V4 expression h5ad lacks COSG layer {NORMALIZED_LOG_LAYER!r}")
    if not adata.var_names.is_unique:
        raise ValueError("CellClick V4 COSG requires unique var_names")
    try:
        cosg_version = importlib.metadata.version("cosg")
        cache_root = (work_dir / "cosg_cache" if work_dir is not None else Path(tempfile.gettempdir()) / "cellclick_v4_cosg")
        cache_root.mkdir(parents=True, exist_ok=True)
        # The pinned Scanpy/Numba stack needs a writable cache location when
        # imported from a Conda site-packages directory.
        os.environ.setdefault("NUMBA_CACHE_DIR", str((cache_root / "numba").resolve()))
        import cosg
    except Exception as error:
        raise RuntimeError("CellClick V4 requires COSG for initial evidence") from error
    mapping = manifest["source_row_id_by_group"]
    initial_markers = manifest["initial_markers_by_source_row_id"]
    if not isinstance(initial_markers, dict):
        raise ValueError("Expression manifest lacks initial marker snapshot for COSG")
    source_rows = {str(value) for value in mapping.values()}
    if set(initial_markers) != source_rows:
        raise ValueError("Expression manifest COSG marker snapshot does not match source-row binding")
    gene_universe_hash = hashlib.sha256("\n".join(adata.var_names.astype(str)).encode()).hexdigest()
    cache_identity = {
        "schema": "cellclick.v4.cosg.2", "h5ad_sha256": _digest(expression_path),
        "manifest_sha256": _digest(manifest_path), "groupby": manifest["groupby"],
        "layer": NORMALIZED_LOG_LAYER, "gene_universe_sha256": gene_universe_hash,
        "cosg_version": cosg_version, "parameters": COSG_PARAMETERS,
        "binding_sha256": _hash_json(mapping), "marker_snapshot_sha256": _hash_json(initial_markers),
    }
    cache_key = _hash_json(cache_identity)
    path = _cache_path(work_dir, cache_key)
    if path is not None and path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("identity") == cache_identity and isinstance(cached.get("scores_by_source_row_id"), dict):
                scores = {
                    str(row_id): {str(gene): float(score) for gene, score in values.items()}
                    for row_id, values in cached["scores_by_source_row_id"].items()
                    if isinstance(values, dict)
                }
                valid = set(scores) == source_rows
                for source_row_id, markers in initial_markers.items():
                    values = scores.get(str(source_row_id), {})
                    if set(values) != set(markers) or any(
                        not math.isfinite(score) or score < 0 for score in values.values()
                    ) or sum(values.values()) <= 0:
                        valid = False
                        break
                if valid:
                    return scores, {"cache_key": cache_key, "cache_status": "hit", **cache_identity}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    key_added = "__cellclick_v4_cosg__"
    cosg_groupby = "__cellclick_v4_cosg_group__"
    try:
        # The trusted binding and manifest normalize groups to strings; make
        # COSG use the same ephemeral representation so categorical/integer
        # obs values cannot change the result field names.
        adata.obs[cosg_groupby] = adata.obs[str(manifest["groupby"])].astype(str).to_numpy()
        cosg.cosg(
            adata, groupby=cosg_groupby, n_genes_user=adata.n_vars,
            key_added=key_added, use_raw=False, layer=NORMALIZED_LOG_LAYER,
            copy=False, **COSG_PARAMETERS,
        )
        rankings = adata.uns[key_added]
        names, values = rankings["names"], rankings["scores"]
    except Exception as error:
        raise RuntimeError("CellClick V4 COSG calculation failed") from error
    if not getattr(names.dtype, "names", None) or not getattr(values.dtype, "names", None):
        raise ValueError("CellClick V4 COSG result lacks structured names/scores")
    scores_by_source: dict[str, dict[str, float]] = {}
    for group, source_row_id in mapping.items():
        field = str(group)
        if field not in names.dtype.names or field not in values.dtype.names:
            raise ValueError("CellClick V4 COSG result lacks a bound group")
        gene_scores = {str(gene): float(score) for gene, score in zip(names[field], values[field])}
        selected: dict[str, float] = {}
        for gene in initial_markers[str(source_row_id)]:
            if gene not in gene_scores:
                raise ValueError(f"CellClick V4 COSG score is missing for marker {gene!r}")
            score = gene_scores[gene]
            if not math.isfinite(score) or score < 0:
                raise ValueError(f"CellClick V4 COSG score is invalid for marker {gene!r}")
            selected[gene] = score
        if sum(selected.values()) <= 0:
            raise ValueError(f"CellClick V4 COSG total weight is zero for {source_row_id!r}")
        scores_by_source[str(source_row_id)] = selected
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": cache_identity, "scores_by_source_row_id": scores_by_source},
                                        ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    return scores_by_source, {"cache_key": cache_key, "cache_status": "miss", **cache_identity}


def collect_initial_evidence(task: TaskInput, *, expression_file: str | Path, manifest_file: str | Path,
                             work_dir: Path | None = None) -> tuple[EvidencePacket, MarkerSelection, QualitySummary]:
    """Load the runner-bound h5ad and calculate statistics for submitted markers.

    The manifest is the only permitted source of the group binding. Missing or
    inconsistent files are infrastructure errors, never marker-only evidence.
    """
    try:
        import anndata as ad
    except ImportError as error:
        raise RuntimeError("CellClick V4 requires anndata to read expression_file") from error
    expression_path = Path(expression_file).resolve()
    manifest_path = Path(manifest_file).resolve()
    if not expression_path.is_file():
        raise FileNotFoundError(f"CellClick V4 expression h5ad is missing: {expression_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CellClick V4 expression manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid CellClick V4 expression manifest: {manifest_path}") from error
    if manifest.get("schema_version") not in {
        "cellclick.expression.2",
        "cellclick.v4.expression.2",
    }:
        raise ValueError("Unsupported CellClick V4 expression manifest schema")
    if Path(str(manifest.get("h5ad_path", ""))).resolve() != expression_path:
        raise ValueError("Expression manifest h5ad_path does not match task expression_file")
    mapping = manifest.get("source_row_id_by_group")
    marker_map = manifest.get("initial_markers_by_source_row_id")
    groupby = manifest.get("groupby")
    statistics_layer = manifest.get("statistics_layer")
    if not isinstance(mapping, dict) or not isinstance(marker_map, dict) or not isinstance(groupby, str) or not isinstance(statistics_layer, str):
        raise ValueError("Expression manifest lacks trusted binding fields")
    if statistics_layer != NORMALIZED_LOG_LAYER:
        raise ValueError(f"CellClick V4 statistics_layer must be {NORMALIZED_LOG_LAYER!r}")
    groups = [str(group) for group, row_id in mapping.items() if str(row_id) == task.task_id]
    if len(groups) != 1:
        raise ValueError("Task source_row_id is not bound to exactly one expression group")
    expected_markers = marker_map.get(task.task_id)
    if expected_markers != task.ordered_markers:
        raise ValueError("Task markers do not match the frozen expression marker snapshot")
    adata = ad.read_h5ad(expression_path)
    if groupby not in adata.obs:
        raise ValueError(f"Bound expression h5ad lacks groupby column {groupby!r}")
    if statistics_layer not in adata.layers:
        raise ValueError(f"Bound expression h5ad lacks statistics layer {statistics_layer!r}")
    matrix = adata.layers[statistics_layer]
    if matrix.shape != adata.shape:
        raise ValueError("Bound statistics layer shape does not match h5ad")
    scores_by_source, cosg_provenance = _cosg_scores(
        adata=adata, manifest=manifest, expression_path=expression_path,
        manifest_path=manifest_path, work_dir=work_dir,
    )
    cosg_scores = scores_by_source[task.task_id]
    target = adata.obs[groupby].astype(str).to_numpy() == groups[0]
    if not target.any() or target.all():
        raise ValueError("Bound target group must have both target and background cells")
    positions = {str(gene): index for index, gene in enumerate(adata.var_names.astype(str))}
    records = []
    for gene in task.ordered_markers:
        if gene not in positions:
            raise ValueError(f"CellClick V4 cannot calculate COSG for unmeasured marker {gene!r}")
        if gene not in cosg_scores:
            raise ValueError(f"CellClick V4 COSG score is missing for submitted marker {gene!r}")
        position = positions[gene]
        target_values = np.asarray(matrix[target, position].toarray()).ravel() if hasattr(matrix[target, position], "toarray") else np.asarray(matrix[target, position]).ravel()
        background_values = np.asarray(matrix[~target, position].toarray()).ravel() if hasattr(matrix[~target, position], "toarray") else np.asarray(matrix[~target, position]).ravel()
        pct_target = float(np.count_nonzero(target_values > 0) / target_values.size)
        pct_background = float(np.count_nonzero(background_values > 0) / background_values.size)
        records.append(GeneEvidence(gene=gene, availability="available", pct_target=pct_target,
                                    pct_background=pct_background, delta_pct=pct_target - pct_background,
                                    mean_log1p_difference=float(target_values.mean() - background_values.mean()),
                                    cosg_score=cosg_scores[gene],
                                    source_id=f"{expression_path}:{statistics_layer}"))
    packet = EvidencePacket(packet_id=f"EP_{task.task_id}_initial", comparison="target_vs_rest", genes=records,
                             capability="matrix", limitations=[], cosg_provenance=cosg_provenance)
    selection = MarkerSelection(selection_id=f"SEL_{task.task_id}_matrix", policy_id="matrix_observed_v1",
                                primary=[gene for gene, record in zip(task.ordered_markers, records) if record.availability == "available" and (record.delta_pct or 0) > 0],
                                supplementary=[], low_support=[], unavailable=[record.gene for record in records if record.availability != "available"])
    quality = QualitySummary(quality_id=f"Q_{task.task_id}", flags=[], limitations=[])
    return packet, selection, quality
