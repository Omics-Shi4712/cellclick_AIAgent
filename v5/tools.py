"""Bounded V5 tool registry. Tools return observations; they never adjudicate."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .expression import gene_cosine_statistics, gene_statistics, reference_expression_inputs
from .models import ToolObservation, ToolRequest
from .reference_comparison import compare_reference

ToolHandler = Callable[[ToolRequest, dict[str, Any]], dict[str, Any]]


def parameter_hash(request: ToolRequest, context: dict[str, Any] | None = None) -> str:
    binding = context or {}
    value = {"tool_name": request.tool_name, "target_id": request.target_id, "arguments": request.arguments,
             "binding": {key: binding.get(key) for key in (
                 "source_row_id", "expression_file", "expression_manifest", "expression_file_sha256",
                 "expression_manifest_sha256", "comparison_reference_path_sha256")},
             "statistics_layer": "normalization+log1p"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        if name in self._handlers:
            raise ValueError(f"tool already registered: {name}")
        self._handlers[name] = handler

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def dispatch(self, request: ToolRequest, context: dict[str, Any]) -> ToolObservation:
        import time
        from .runtime import emit
        started = time.perf_counter()
        try:
            return self._dispatch(request, context)
        finally:
            emit("tool", tool_name=request.tool_name, duration_seconds=time.perf_counter() - started)

    def _dispatch(self, request: ToolRequest, context: dict[str, Any]) -> ToolObservation:
        digest = parameter_hash(request, context)
        handler = self._handlers.get(request.tool_name)
        if handler is None:
            return ToolObservation(observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                                   task_id=request.task_id, target_id=request.target_id, tool_name=request.tool_name,
                                   status="capability_unavailable", parameter_hash=digest,
                                   limitations=["tool_not_registered"])
        try:
            result = handler(request, context)
        except LookupError as error:
            return ToolObservation(observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                                   task_id=request.task_id, target_id=request.target_id, tool_name=request.tool_name,
                                   status="capability_unavailable", parameter_hash=digest,
                                   limitations=[str(error)])
        except Exception as error:
            return ToolObservation(observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                                   task_id=request.task_id, target_id=request.target_id, tool_name=request.tool_name,
                                   status="tool_error", parameter_hash=digest,
                                   limitations=[f"{type(error).__name__}:{error}"])
        return ToolObservation(observation_id=f"OBS_{request.request_id}", request_id=request.request_id,
                               task_id=request.task_id, target_id=request.target_id, tool_name=request.tool_name,
                               status="success", result=result, parameter_hash=digest)


def _not_implemented(request: ToolRequest, context: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("V5 tool requires an environment-specific handler")


DEFAULT_TOOLS = ToolRegistry()
for _name in ("query_expression", "gene_cosine", "compare_reference"):
    DEFAULT_TOOLS.register(_name, _not_implemented)


__all__ = ["DEFAULT_TOOLS", "ToolRegistry", "parameter_hash"]


def build_tool_registry(context: dict[str, Any]) -> ToolRegistry:
    """Build V5's bounded, task-local tool set from pinned method evidence."""
    registry = ToolRegistry()

    def expression(request: ToolRequest, _: dict[str, Any]) -> dict[str, Any]:
        genes = request.arguments.get("genes", [])
        if not isinstance(genes, list):
            raise ValueError("query_expression.arguments.genes must be a list")
        return gene_statistics(expression_file=context["expression_file"], manifest_file=context["expression_manifest"],
                               source_row_id=context["source_row_id"], genes=genes)

    def cosine(request: ToolRequest, _: dict[str, Any]) -> dict[str, Any]:
        gene_sets = request.arguments.get("gene_sets")
        compare_pairs = request.arguments.get("compare_pairs")
        if not isinstance(gene_sets, list) or not isinstance(compare_pairs, list):
            raise ValueError("gene_cosine requires gene_sets and compare_pairs lists")
        result = gene_cosine_statistics(
            expression_file=context["expression_file"], manifest_file=context["expression_manifest"],
            source_row_id=context["source_row_id"], gene_sets=gene_sets, compare_pairs=compare_pairs,
        )
        digest = parameter_hash(request, context)
        artifact_id = "GC_" + digest[:20]
        document = {
            "schema_version": "cellclick.v5.gene_cosine.1",
            "artifact_id": artifact_id,
            "parameter_hash": digest,
            "task_id": request.task_id,
            "target_id": request.target_id,
            "source_row_id": context["source_row_id"],
            "binding": {key: context.get(key) for key in (
                "expression_file", "expression_manifest", "expression_file_sha256",
                "expression_manifest_sha256",
            )},
            "arguments": request.arguments,
            "result": result,
        }
        serialized = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        artifact_sha256 = hashlib.sha256(serialized.encode()).hexdigest()
        artifact_path = None
        work_dir = context.get("work_dir")
        if work_dir:
            path = Path(work_dir) / f"{artifact_id}.json"
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
            artifact_path = str(path)
        result["artifact"] = {
            "artifact_id": artifact_id,
            "artifact_path": artifact_path,
            "sha256": artifact_sha256,
            "schema_version": document["schema_version"],
        }
        return result

    def comparison(request: ToolRequest, _: dict[str, Any]) -> dict[str, Any]:
        reference_path = context.get("comparison_reference_path")
        from .runtime import read_json
        reference = read_json(reference_path) if reference_path else {}
        if not isinstance(reference, dict) or not all(isinstance(k, str) and isinstance(v, list) for k, v in reference.items()):
            raise ValueError("ReferenceComparison JSON must map cell-type strings to marker lists")
        marker_sets = request.arguments.get("marker_sets")
        if not isinstance(marker_sets, list) or len(marker_sets) != 1:
            raise ValueError("compare_reference requires exactly one marker set")
        marker_set = marker_sets[0]
        candidate_id = str(marker_set.get("candidate_id", "")).strip()
        allowed = set(context.get("allowed_candidate_ids", ()))
        if candidate_id not in allowed or request.hypothesis_ids != [candidate_id]:
            raise ValueError("compare_reference candidate must be the current validation candidate")
        source = marker_set.get("source")
        reference_label = marker_set.get("reference_label")
        requested_genes = list(dict.fromkeys(str(g).strip() for g in marker_set.get("genes", []) if str(g).strip()))
        if not requested_genes:
            raise ValueError("compare_reference marker set requires genes")
        if source == "reference":
            if reference_label not in reference:
                raise ValueError("reference_label must exactly match an available reference")
            allowed_genes = list(dict.fromkeys(str(g).strip() for g in reference[reference_label] if str(g).strip()))
            if not set(requested_genes) <= set(allowed_genes):
                raise ValueError("reference marker genes must come from the selected reference")
            reference_genes = requested_genes
            reference_version_source = {"path": str(Path(reference_path).resolve()), "label": reference_label,
                                        "genes": reference[reference_label]}
        elif source == "llm":
            if reference_label is not None:
                raise ValueError("source=llm must not claim a reference label")
            reference_genes = requested_genes
            reference_version_source = {"source": "llm", "candidate_id": candidate_id}
        else:
            raise ValueError("marker set source must be reference or llm")
        query_markers = context.get("query_markers", [])
        if not query_markers:
            raise ValueError("query markers are unavailable for reference comparison")
        genes = list(dict.fromkeys([*query_markers, *reference_genes]))
        vectors = reference_expression_inputs(expression_file=context["expression_file"], manifest_file=context["expression_manifest"],
                                              source_row_id=context["source_row_id"], genes=genes)
        specificity = {}
        target_index = vectors["groups"].index(vectors["target_group"])
        for gene in genes:
            values = vectors["query_means"].get(gene)
            if values:
                rest = [value for i, value in enumerate(values) if i != target_index]
                specificity[gene] = max(0.0, float(values[target_index] - (sum(rest) / len(rest) if rest else 0.0)))
        result = compare_reference(query_markers=query_markers, reference_markers=reference_genes,
                query_vectors=vectors["query_vectors"], reference_vectors=vectors["query_vectors"],
                query_means=vectors["query_means"], reference_means=vectors["query_means"],
                cosg_scores=specificity, cell_expression=vectors["cell_expression"])
        label = reference_label or candidate_id
        scores = {label: float(sum(result["scores"].values()) / len(result["scores"])) if result["scores"] else 0.0}
        coverage = {label: float(result["coverage"])}
        valid_markers = {label: result["valid_markers"]}
        decompositions = {label: {key: result[key] for key in ("GS", "RES", "EAD")}}
        reference_version = hashlib.sha256(json.dumps(reference_version_source, sort_keys=True).encode()).hexdigest()
        digest = hashlib.sha256(json.dumps({"reference": reference_version_source,
                                             "source_row_id": context["source_row_id"],
                                             "expression_file_sha256": context.get("expression_file_sha256"),
                                             "genes": reference_genes, "query_markers": query_markers},
                                            sort_keys=True).encode()).hexdigest()
        comparison_id = "RC_" + digest[:20]
        cell_scores = {label: result.get("scores", {})}
        artifact = {"comparison_id": comparison_id, "labels": [label], "scores": scores,
                    "cell_scores": cell_scores}
        work_dir = context.get("work_dir")
        if work_dir:
            path = Path(work_dir) / f"{comparison_id}.json"
            path.write_text(json.dumps({"comparison_id": comparison_id, "scores": scores,
                                        "cell_scores": cell_scores,
                                        "coverage": coverage, "valid_markers": valid_markers,
                                        "decompositions": decompositions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            artifact["artifact_path"] = str(path)
        ead_values = result["EAD"]
        ead_summary = {gene: {"mean": sum(values[gene] for values in ead_values.values()) / len(ead_values),
                              "min": min(values[gene] for values in ead_values.values()),
                              "max": max(values[gene] for values in ead_values.values())}
                       for gene in result["valid_markers"]} if ead_values else {}
        summarized = {label: {"GS": result["GS"], "RES": result["RES"], "EAD": ead_summary}}
        cell_values = list(result["scores"].values())
        return {"comparison_id": comparison_id, "candidate_ids": [candidate_id], "labels": [label],
                "gene_source": source, "reference_label": reference_label,
                "reference_path": str(Path(reference_path).resolve()) if reference_path else None,
                "reference_version": reference_version,
                "artifact_path": artifact.get("artifact_path"),
                "score_matrix_handle": comparison_id, "scores": scores, "coverage": coverage,
                "valid_markers": valid_markers, "missing_genes": vectors["missing_genes"],
                "query_marker_overlap": sorted(set(query_markers) & set(reference_genes)),
                "decompositions": summarized,
                "cell_score_summary": {"count": len(cell_values),
                                       "mean": scores[label],
                                       "min": min(cell_values) if cell_values else None,
                                       "max": max(cell_values) if cell_values else None},
                "cell_count": vectors["cell_count"], "statistics_layer": "normalization+log1p",
                "calculation_mode": "query_expression_all_cells_v3",
                "limitations": (["llm_marker_set_is_not_independent_reference"] if source == "llm" else [])
                                + (["missing_genes:" + ",".join(vectors["missing_genes"])] if vectors["missing_genes"] else [])}

    registry.register("query_expression", expression)
    registry.register("gene_cosine", cosine)
    registry.register("compare_reference", comparison)
    return registry
