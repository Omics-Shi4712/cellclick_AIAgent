"""Bounded, deduplicated projections of complete V5 evidence for LLM calls."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from .models import EvidenceLedgerEntry, ToolObservation, V5State

CONTEXT_SCHEMA_VERSION = "cellclick.v5.llm_context.1"
COSINE_SUMMARY_VERSION = "cellclick.v5.gene_cosine_summary.1"
REPRESENTATIVE_PAIR_LIMIT = 3
EXPRESSION_REPRESENTATIVE_LIMIT = 5
LLM_CONTEXT_CONTRACT = {
    "context_fields": [
        "task", "candidate_registry", "marker_audit", "rounds", "verification_history",
        "transition_history", "verification_reports", "evidence_ledger", "current_observation",
    ],
    "history_tool_payload": "observation_handles_only",
    "current_observation": "single_summarized_copy",
    "historical_evidence": "one_summary_then_duplicate_reference",
    "cosine_distribution_fields": ["count", "min", "p25", "median", "p75", "max"],
    "cosine_representative_rules": [
        "highest_cosine", "lowest_cosine", "largest_cosine_joint_difference",
    ],
    "expression_representative_metrics": ["mean_difference", "log2_fold_change", "delta_pct"],
}


def _rounded(value: float | int | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return _rounded(ordered[lower])
    weight = position - lower
    return _rounded(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": _quantile(values, 0),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "p75": _quantile(values, 0.75),
        "max": _quantile(values, 1),
    }


def _pair_key(pair: dict[str, Any]) -> tuple[str, str]:
    return str(pair.get("left_gene", "")), str(pair.get("right_gene", ""))


def _representative_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    computable = [pair for pair in pairs if pair.get("computable") is True
                  and isinstance(pair.get("cosine"), (int, float))
                  and isinstance(pair.get("joint_detected_fraction"), (int, float))]
    lexical = sorted(computable, key=_pair_key)
    selections: dict[tuple[str, str], set[str]] = {}

    rankings = (
        ("highest_cosine", sorted(lexical, key=lambda pair: -float(pair["cosine"]))),
        ("lowest_cosine", sorted(lexical, key=lambda pair: float(pair["cosine"]))),
        ("largest_cosine_joint_difference", sorted(
            lexical,
            key=lambda pair: -abs(float(pair["cosine"]) - float(pair["joint_detected_fraction"])),
        )),
    )
    for reason, ranked in rankings:
        for pair in ranked[:REPRESENTATIVE_PAIR_LIMIT]:
            selections.setdefault(_pair_key(pair), set()).add(reason)

    by_key = {_pair_key(pair): pair for pair in computable}
    result = []
    for key in sorted(selections):
        pair = by_key[key]
        result.append({
            "left_gene": key[0],
            "right_gene": key[1],
            "cosine": _rounded(pair["cosine"]),
            "joint_detected_fraction": _rounded(pair["joint_detected_fraction"]),
            "left_detected": pair.get("left_detected"),
            "right_detected": pair.get("right_detected"),
            "joint_detected": pair.get("joint_detected"),
            "selection_reasons": sorted(selections[key]),
        })
    return result


def _pair_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    computable = [pair for pair in pairs if pair.get("computable") is True
                  and isinstance(pair.get("cosine"), (int, float))]
    reason_counts: dict[str, int] = {}
    for pair in pairs:
        if pair.get("computable") is not True:
            reason = str(pair.get("reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    cosine = [float(pair["cosine"]) for pair in computable]
    joint = [float(pair["joint_detected_fraction"]) for pair in computable
             if isinstance(pair.get("joint_detected_fraction"), (int, float))]
    return {
        "pair_count": len(pairs),
        "computable_pair_count": len(computable),
        "uncomputable_pair_count": len(pairs) - len(computable),
        "uncomputable_reason_counts": dict(sorted(reason_counts.items())),
        "cosine": _distribution(cosine),
        "joint_detected_fraction": _distribution(joint),
        "representative_pairs": _representative_pairs(pairs),
    }


def summarize_gene_cosine(result: dict[str, Any]) -> dict[str, Any]:
    gene_sets = result.get("gene_sets") if isinstance(result.get("gene_sets"), dict) else {}
    detection = result.get("gene_detection") if isinstance(result.get("gene_detection"), dict) else {}
    within = result.get("within_group") if isinstance(result.get("within_group"), dict) else {}
    across = result.get("cross_group") if isinstance(result.get("cross_group"), list) else []
    available = result.get("available_genes") if isinstance(result.get("available_genes"), list) else []
    missing = result.get("missing_genes") if isinstance(result.get("missing_genes"), list) else []
    zero = result.get("zero_vector_genes") if isinstance(result.get("zero_vector_genes"), list) else []
    return {
        "summary_version": COSINE_SUMMARY_VERSION,
        "source_row_id": result.get("source_row_id"),
        "statistics_layer": result.get("statistics_layer"),
        "cell_count": result.get("cell_count"),
        "gene_sets": copy.deepcopy(gene_sets),
        "gene_counts": {candidate_id: len(genes) for candidate_id, genes in sorted(gene_sets.items())},
        "available_gene_count": len(available),
        "available_genes": copy.deepcopy(available),
        "missing_genes": copy.deepcopy(missing),
        "zero_vector_genes": copy.deepcopy(zero),
        "gene_detection": copy.deepcopy(detection),
        "within_group_summary": [
            {"candidate_id": candidate_id, **_pair_summary(pairs if isinstance(pairs, list) else [])}
            for candidate_id, pairs in sorted(within.items())
        ],
        "cross_group_summary": [
            {
                "left_candidate_id": item.get("left_candidate_id"),
                "right_candidate_id": item.get("right_candidate_id"),
                **_pair_summary(item.get("gene_pairs") if isinstance(item.get("gene_pairs"), list) else []),
            }
            for item in across if isinstance(item, dict)
        ],
        "artifact": copy.deepcopy(result.get("artifact")),
        "limitations": copy.deepcopy(result.get("limitations", [])),
    }


def _expression_summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("genes") if isinstance(result.get("genes"), list) else []
    missing = [row.get("gene") for row in rows if isinstance(row, dict) and not row.get("available")]
    zero = [row.get("gene") for row in rows if isinstance(row, dict) and row.get("available")
            and float(row.get("target_normalized_mean", 0)) == 0
            and float(row.get("background_normalized_mean", 0)) == 0]
    available = [row for row in rows if isinstance(row, dict) and row.get("available")]
    selections: dict[str, set[str]] = {}
    by_gene = {str(row.get("gene", "")): row for row in available}
    for metric in ("mean_difference", "log2_fold_change", "delta_pct"):
        eligible = [row for row in available if isinstance(row.get(metric), (int, float))]
        for direction, ranked in (
            ("highest", sorted(eligible, key=lambda row: (-float(row[metric]), str(row.get("gene", ""))))),
            ("lowest", sorted(eligible, key=lambda row: (float(row[metric]), str(row.get("gene", ""))))),
        ):
            for row in ranked[:EXPRESSION_REPRESENTATIVE_LIMIT]:
                selections.setdefault(str(row.get("gene", "")), set()).add(f"{direction}_{metric}")
    representatives = []
    for gene in sorted(selections):
        representatives.append({**copy.deepcopy(by_gene[gene]),
                                "selection_reasons": sorted(selections[gene])})
    return {
        "source_row_id": result.get("source_row_id"),
        "statistics_layer": result.get("statistics_layer"),
        "cell_count": result.get("cell_count"),
        "background_cell_count": result.get("background_cell_count"),
        "fold_change_epsilon": result.get("fold_change_epsilon"),
        "requested_gene_count": len(rows),
        "available_gene_count": len(available),
        "missing_genes": missing,
        "zero_expression_genes": zero,
        "representative_genes": representatives,
    }


def summarize_tool_result(tool_name: str, result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    if tool_name == "gene_cosine":
        return summarize_gene_cosine(result)
    if tool_name == "query_expression":
        return _expression_summary(result)
    if tool_name == "compare_reference":
        keys = (
            "comparison_id", "candidate_ids", "labels", "gene_source", "reference_label",
            "reference_version", "artifact_path", "score_matrix_handle", "scores", "coverage",
            "valid_markers", "missing_genes", "query_marker_overlap", "decompositions",
            "cell_score_summary", "cell_count", "statistics_layer", "calculation_mode", "limitations",
        )
        return {key: copy.deepcopy(result.get(key)) for key in keys if key in result}
    return copy.deepcopy(result)


def _observation_projection(observation: dict[str, Any], *, include_summary: bool) -> dict[str, Any]:
    projected = {key: copy.deepcopy(observation.get(key)) for key in (
        "observation_id", "request_id", "task_id", "target_id", "tool_name", "status",
        "parameter_hash", "evidence_ids", "limitations",
    ) if key in observation}
    result = observation.get("result")
    if include_summary and isinstance(result, dict):
        projected["result_summary"] = summarize_tool_result(str(observation.get("tool_name", "")), result)
    return projected


def _history_projection(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for entry in history:
        value = copy.deepcopy(entry)
        observation = value.get("observation")
        if isinstance(observation, dict):
            value["observation"] = _observation_projection(observation, include_summary=False)
            # LLM_B's immediately preceding response is the canonical record of
            # the requested parameters and rationale.
            value.pop("arguments", None)
        projected.append(value)
    return projected


def _evidence_projection(entries: list[EvidenceLedgerEntry], current_ids: set[str]) -> list[dict[str, Any]]:
    projected = []
    seen_results: dict[str, str] = {}
    for entry in entries:
        base = {key: copy.deepcopy(value) for key, value in entry.model_dump().items() if key != "payload"}
        if entry.evidence_id in current_ids:
            base["payload"] = {"current_observation": True}
        elif entry.observation_id is None:
            base["payload"] = copy.deepcopy(entry.payload)
        else:
            digest = hashlib.sha256(json.dumps(
                {"source": entry.source, "payload": entry.payload}, sort_keys=True,
                ensure_ascii=False, separators=(",", ":"),
            ).encode()).hexdigest()
            if digest in seen_results:
                base["payload"] = {"duplicate_of_evidence_id": seen_results[digest]}
            else:
                seen_results[digest] = entry.evidence_id
                base["payload"] = summarize_tool_result(entry.source, entry.payload)
        projected.append(base)
    return projected


def build_llm_context(state: V5State, current_observation: ToolObservation | None = None) -> dict[str, Any]:
    """Return a complete semantic history without repeating full tool payloads."""
    current_ids = set(current_observation.evidence_ids) if current_observation else set()
    result = {
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "task": state.task.model_dump(),
        "candidate_registry": state.candidate_registry.model_dump(),
        "marker_audit": state.marker_audit.model_dump() if state.marker_audit else None,
        "rounds": copy.deepcopy(state.rounds),
        "verification_history": _history_projection(state.verification_history),
        "transition_history": copy.deepcopy(state.transition_history),
        "verification_reports": [item.model_dump() for item in state.verification_reports],
        "evidence_ledger": _evidence_projection(state.evidence_store, current_ids),
    }
    if current_observation is not None:
        result["current_observation"] = _observation_projection(
            current_observation.model_dump(), include_summary=True)
    return result


__all__ = [
    "CONTEXT_SCHEMA_VERSION", "COSINE_SUMMARY_VERSION", "REPRESENTATIVE_PAIR_LIMIT",
    "EXPRESSION_REPRESENTATIVE_LIMIT", "LLM_CONTEXT_CONTRACT", "build_llm_context", "summarize_gene_cosine",
    "summarize_tool_result",
]
