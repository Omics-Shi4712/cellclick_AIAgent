"""Bounded V5 tool registry. Tools return observations; they never adjudicate."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .expression import gene_statistics, reference_expression_inputs
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
        if not isinstance(labels, list) or not labels:
            raise ValueError("compare_reference.arguments.labels must contain at least one label")
        if len(labels) != len(set(labels)) or any(label not in reference for label in labels):
            raise ValueError("compare_reference labels must exactly match available reference labels")
        allowed = set(context.get("allowed_candidate_ids", ()))
        if not set(request.hypothesis_ids) <= allowed or not request.hypothesis_ids:
            raise ValueError("compare_reference hypothesis_ids must belong to the current candidates")
        chosen = {label: reference[label] for label in labels}
        query_markers = context.get("query_markers", [])
        if not query_markers:
            raise ValueError("query markers are unavailable for reference comparison")
        genes = list(dict.fromkeys([*query_markers, *(gene for markers in chosen.values() for gene in markers)]))
        vectors = reference_expression_inputs(expression_file=context["expression_file"], manifest_file=context["expression_manifest"],
                                              source_row_id=context["source_row_id"], genes=genes)
        cosg = {}
        target_index = vectors["groups"].index(vectors["target_group"])
        for gene in query_markers:
            values = vectors["query_means"].get(gene)
            if values:
                rest = [value for i, value in enumerate(values) if i != target_index]
                cosg[gene] = max(0.0, float(values[target_index] - (sum(rest) / len(rest) if rest else 0.0)))
        scores, coverage, decompositions, valid_markers, cell_scores = {}, {}, {}, {}, {}
        for label, markers in chosen.items():
            result = compare_reference(query_markers=query_markers, reference_markers=markers,
                query_vectors=vectors["query_vectors"], reference_vectors=vectors["query_vectors"],
                query_means=vectors["query_means"], reference_means=vectors["query_means"],
                cosg_scores=cosg, cell_expression=vectors["cell_expression"])
            scores[label] = float(sum(result["scores"].values()) / len(result["scores"])) if result["scores"] else 0.0
            cell_scores[label] = result.get("scores", {})
            coverage[label] = float(result["coverage"])
            valid_markers[label] = result["valid_markers"]
            decompositions[label] = {key: result[key] for key in ("GS", "RES", "EAD")}
        digest = hashlib.sha256(json.dumps({"path": str(Path(reference_path).resolve()), "labels": labels,
                                             "query_markers": query_markers}, sort_keys=True).encode()).hexdigest()
        comparison_id = "RC_" + digest[:20]
        artifact = {"comparison_id": comparison_id, "labels": labels, "scores": scores,
                    "cell_scores": cell_scores}
        work_dir = context.get("work_dir")
        if work_dir:
            path = Path(work_dir) / f"{comparison_id}.json"
            path.write_text(json.dumps({"comparison_id": comparison_id, "scores": scores,
                                        "cell_scores": cell_scores,
                                        "coverage": coverage, "valid_markers": valid_markers,
                                        "decompositions": decompositions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            artifact["artifact_path"] = str(path)
        return {"comparison_id": comparison_id, "candidate_ids": list(request.hypothesis_ids), "labels": labels,
                "reference_path": str(Path(reference_path).resolve()), "reference_version": digest,
                "score_matrix_handle": comparison_id, "scores": scores, "coverage": coverage,
                "valid_markers": valid_markers, "decompositions": decompositions,
                "calculation_mode": "query_expression_all_cells_v2",
                "limitations": ["missing_genes:" + ",".join(vectors["missing_genes"])] if vectors["missing_genes"] else []}

    def patterns(request: ToolRequest, _: dict[str, Any]) -> dict[str, Any]:
        comparison_id = request.arguments.get("comparison_id")
        if not isinstance(comparison_id, str) or not comparison_id.strip():
            raise ValueError("inspect_score_patterns.arguments.comparison_id is required")
        allowed_comparisons = set(context.get("comparison_ids", ()))
        if comparison_id not in allowed_comparisons:
            raise LookupError("comparison_id_not_available_for_target")
        path = Path(context.get("work_dir", ".")) / f"{comparison_id}.json"
        if not path.is_file():
            raise LookupError("comparison_id_not_found")
        scores = json.loads(path.read_text(encoding="utf-8")).get("scores", {})
        if not isinstance(scores, dict) or not scores:
            raise LookupError("comparison_has_no_scores")
        ranked = sorted(((str(label), float(score)) for label, score in scores.items()), key=lambda item: (-item[1], item[0]))
        return {"ranked_labels": [{"label": label, "score": score} for label, score in ranked],
                "top_score_gap": round(ranked[0][1] - ranked[1][1], 6) if len(ranked) > 1 else None,
                "mixture_suspected": len(ranked) > 1 and ranked[0][1] > 0 and ranked[1][1] / ranked[0][1] >= 0.8}

    registry.register("query_expression", expression)
    registry.register("query_cellmarker", cellmarker)
    if context.get("comparison_reference_path"):
        registry.register("compare_reference", comparison)
        if context.get("allow_score_patterns", False):
            registry.register("inspect_score_patterns", patterns)
    return registry
