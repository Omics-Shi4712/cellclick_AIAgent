"""Build fail-closed replay fixtures from completed CellClick v1 traces."""

from __future__ import annotations

import json
from typing import Any

from cellclick.llm import FrozenJsonResponse, ReplayJsonClient, request_fingerprint

from .graph import candidate_schema, validation_schema
from .prompts import annotation_user, systems, validation_user


def _parsed_object(raw: str, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Frozen {context} response is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"Frozen {context} response must be a JSON object")
    return value


def replay_responses_from_trace(
    trace: dict[str, Any], *, temperature: float,
) -> dict[str, FrozenJsonResponse]:
    """Reconstruct successful v1 requests recorded by one completed trace.

    Historical traces retain only the successful raw response for each logical
    request. Retry failure sequences are therefore covered by synthetic unit
    fixtures rather than fabricated from trace data.
    """
    annotation_system, validation_system, _ = systems()
    model = trace.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Trace lacks a non-empty model")
    required = ("species", "tissue", "markers", "round_history")
    if any(field not in trace for field in required):
        raise ValueError("Trace lacks required v1 replay fields")
    responses: dict[str, FrozenJsonResponse] = {}
    prior_feedback: str | None = None

    def add_response(
        *, system: str, user: str, schema: dict[str, Any], raw: str, context: str,
    ) -> None:
        fingerprint = request_fingerprint(
            system=system,
            user=user,
            schema=schema,
            model=model,
            temperature=temperature,
        )
        response = FrozenJsonResponse(parsed=_parsed_object(raw, context=context), raw=raw)
        existing = responses.get(fingerprint)
        if existing is not None and existing != response:
            raise ValueError(f"Conflicting frozen responses for request {fingerprint}")
        responses[fingerprint] = response

    for round_record in trace["round_history"]:
        candidates = round_record.get("candidates")
        raw_records = round_record.get("annotation_raw_responses")
        validation_value = round_record.get("validation")
        if not isinstance(candidates, list) or not isinstance(raw_records, list) or not isinstance(validation_value, dict):
            raise ValueError("Trace round lacks candidates, raw annotation responses, or validation")
        raw_by_id = {
            record.get("candidate_id"): record.get("raw_response")
            for record in raw_records if isinstance(record, dict)
        }
        for candidate in candidates:
            candidate_id = candidate.get("candidate_id") if isinstance(candidate, dict) else None
            raw = raw_by_id.get(candidate_id)
            if not isinstance(candidate_id, str) or not isinstance(raw, str):
                raise ValueError("Trace annotation response is missing or misaligned")
            request = {
                "candidate_id": candidate_id,
                "species": trace["species"],
                "tissue": trace["tissue"],
                "ordered_markers": trace["markers"],
                "revision_feedback": prior_feedback,
            }
            add_response(
                system=annotation_system,
                user=annotation_user(request),
                schema=candidate_schema(),
                raw=raw,
                context=f"annotation {candidate_id}",
            )
        validation_request = {
            "species": trace["species"],
            "tissue": trace["tissue"],
            "ordered_markers": trace["markers"],
            "candidates": candidates,
            "prior_revision_feedback": prior_feedback,
        }
        validation_raw = round_record.get("validation_raw_response")
        if not isinstance(validation_raw, str):
            raise ValueError("Trace validation response is missing")
        add_response(
            system=validation_system,
            user=validation_user(validation_request),
            schema=validation_schema(),
            raw=validation_raw,
            context=f"validation round {round_record.get('round')}",
        )
        prior_feedback = validation_value.get("revision_feedback")
    return responses


def replay_client_from_trace(trace: dict[str, Any], *, temperature: float) -> ReplayJsonClient:
    return ReplayJsonClient(replay_responses_from_trace(trace, temperature=temperature))

