"""Bounded V5 tool registry. Tools return observations; they never adjudicate."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .expression import gene_statistics
from .models import ToolObservation, ToolRequest
from .reference_comparison import compare_reference

ToolHandler = Callable[[ToolRequest, dict[str, Any]], dict[str, Any]]


def parameter_hash(request: ToolRequest) -> str:
    value = {"tool_name": request.tool_name, "target_id": request.target_id, "arguments": request.arguments}
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
        digest = parameter_hash(request)
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
for _name in ("query_cellmarker", "query_expression", "compare_reference", "inspect_score_patterns"):
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

    def cellmarker(request: ToolRequest, _: dict[str, Any]) -> dict[str, Any]:
        from cellclick.reference import MarkerReferenceRetriever
        genes = request.arguments.get("genes", [])
        if not isinstance(genes, list) or not genes:
            raise ValueError("query_cellmarker.arguments.genes must be a non-empty list")
        retriever = MarkerReferenceRetriever.from_tsv(Path(context["cellmarker_reference_path"]))
        return retriever.retrieve(genes, context["tissue"], top_k=int(context["cellmarker_top_k"])).prompt_dict()

    def comparison(request: ToolRequest, _: dict[str, Any]) -> dict[str, Any]:
        reference_path = context.get("comparison_reference_path")
        if not reference_path:
            raise LookupError("reference_comparison_unavailable_for_tissue")
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        if not isinstance(reference, dict) or not all(isinstance(k, str) and isinstance(v, list) for k, v in reference.items()):
            raise ValueError("ReferenceComparison JSON must map cell-type strings to marker lists")
        labels = request.arguments.get("labels")
        chosen = {label: reference[label] for label in labels if label in reference} if isinstance(labels, list) else reference
        query_markers = request.arguments.get("query_markers", [])
        if not isinstance(query_markers, list) or not query_markers:
            raise ValueError("compare_reference.arguments.query_markers must be a non-empty list")
        genes = list(dict.fromkeys([*query_markers, *(gene for markers in chosen.values() for gene in markers)]))
        stats = gene_statistics(expression_file=context["expression_file"], manifest_file=context["expression_manifest"],
                                source_row_id=context["source_row_id"], genes=genes)
        by_gene = {row["gene"]: row for row in stats["genes"] if row.get("available")}
        # The export exposes target/background group statistics, not a labeled
        # reference expression matrix.  Use the trusted target vector as the
        # projection cell and surface absent reference vectors as limitations.
        expression = {gene: float(row["mean_log1p"]) for gene, row in by_gene.items()}
        cosg = {gene: max(0.0, float(row["mean_difference"])) for gene, row in by_gene.items()}
        scores, coverage, decompositions, valid_markers = {}, {}, {}, {}
        for label, markers in chosen.items():
            result = compare_reference(query_markers=query_markers, reference_markers=markers, query_vectors={}, reference_vectors={},
                query_means={}, reference_means={}, cosg_scores=cosg, cell_expression={"target": expression})
            scores[label] = float(result["scores"].get("target", 0.0))
            coverage[label] = float(result["coverage"])
            valid_markers[label] = result["valid_markers"]
            decompositions[label] = {key: result[key] for key in ("GS", "RES", "EAD")}
        digest = hashlib.sha256(json.dumps({"path": str(Path(reference_path).resolve()), "labels": sorted(chosen)}, sort_keys=True).encode()).hexdigest()
        return {"reference_path": str(Path(reference_path).resolve()), "reference_version": digest,
                "score_matrix_handle": "RC_" + digest[:20], "scores": scores, "coverage": coverage,
                "valid_markers": valid_markers, "decompositions": decompositions,
                "calculation_mode": "query_expression_projection_v1",
                "limitations": ["reference_expression_vectors_unavailable; target_expression_projection_used"]}

    def patterns(request: ToolRequest, _: dict[str, Any]) -> dict[str, Any]:
        scores = request.arguments.get("scores")
        if isinstance(scores, list):
            scores = {
                item.get("label"): item.get("score")
                for item in scores if isinstance(item, dict) and item.get("label")
            }
        if not isinstance(scores, dict) or not scores:
            raise ValueError("inspect_score_patterns.arguments.scores must be a non-empty mapping")
        ranked = sorted(((str(label), float(score)) for label, score in scores.items()), key=lambda item: (-item[1], item[0]))
        return {"ranked_labels": [{"label": label, "score": score} for label, score in ranked],
                "top_score_gap": round(ranked[0][1] - ranked[1][1], 6) if len(ranked) > 1 else None,
                "mixture_suspected": len(ranked) > 1 and ranked[0][1] > 0 and ranked[1][1] / ranked[0][1] >= 0.8}

    registry.register("query_expression", expression)
    registry.register("query_cellmarker", cellmarker)
    registry.register("compare_reference", comparison)
    registry.register("inspect_score_patterns", patterns)
    return registry
