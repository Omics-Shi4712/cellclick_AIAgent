"""Frozen deterministic fallback selection for CellClick v1."""

from __future__ import annotations

from collections import Counter
from typing import Any


FALLBACK_POLICY = "label_consensus_marker_evidence_v1"


def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def select_fallback_candidate(
    candidates: list[dict[str, Any]], ordered_markers: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select by label consensus, ranked submitted-marker evidence, then ID."""
    if not candidates:
        raise ValueError("Fallback selection requires at least one candidate")
    marker_weights: dict[str, int] = {}
    for index, marker in enumerate(ordered_markers):
        normalized = _normalize(marker)
        if normalized and normalized not in marker_weights:
            marker_weights[normalized] = len(ordered_markers) - index
    label_votes = Counter(_normalize(str(candidate["label"])) for candidate in candidates)
    scored: list[tuple[tuple[int, int, str], dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates:
        supporting = {_normalize(str(marker)) for marker in candidate.get("supporting_markers", [])}
        conflicting = {_normalize(str(marker)) for marker in candidate.get("conflicting_markers", [])}
        supporting_score = sum(marker_weights.get(marker, 0) for marker in supporting)
        conflicting_score = sum(marker_weights.get(marker, 0) for marker in conflicting)
        evidence_score = supporting_score - conflicting_score
        vote_count = label_votes[_normalize(str(candidate["label"]))]
        detail = {
            "candidate_id": candidate["candidate_id"],
            "label": candidate["label"],
            "label_vote_count": vote_count,
            "evidence_score": evidence_score,
            "supporting_score": supporting_score,
            "conflicting_score": conflicting_score,
        }
        scored.append(((-vote_count, -evidence_score, str(candidate["candidate_id"])), candidate, detail))
    scored.sort(key=lambda item: item[0])
    selected = scored[0][1]
    audit = {
        "policy": FALLBACK_POLICY,
        "selected_candidate_id": selected["candidate_id"],
        "candidate_scores": [item[2] for item in scored],
    }
    return selected, audit

