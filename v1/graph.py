"""Frozen bounded three-node LangGraph workflow for CellClick v1."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from cellclick.llm import JsonChatClient, close_json_client

from .fallback import select_fallback_candidate
from .models import Candidate, ValidationResult
from .prompts import PROMPT_VERSION, annotation_user, systems, validation_user
from .retry import retry_json


PROTOCOL_VERSION = "v1"
ANNOTATION_CANDIDATES = 5
MAX_ROUNDS = 3
REQUEST_RETRIES = 2
MAX_PARALLEL_CLUSTERS = 1
GRAPH_VERSION = "annotation-validation-formatting-v1"
FORMATTING_VERSION = "deterministic-python-v1"


class ClusterState(TypedDict):
    cluster_id: str
    dataset: str
    species: str
    model: str
    prompt_version: str
    protocol_version: str
    tissue: str
    markers: list[str]
    round_count: int
    current_candidates: list[dict[str, Any]]
    validation: dict[str, Any] | None
    round_history: list[dict[str, Any]]
    final_record: dict[str, Any] | None
    status: str
    revision_feedback: str | None
    annotation_raw: list[dict[str, Any]]


def candidate_schema() -> dict[str, Any]:
    """Return the exact JSON schema supplied to v1 annotation requests."""
    schema = Candidate.model_json_schema()
    schema["properties"].pop("candidate_id", None)
    schema["required"] = [field for field in schema.get("required", []) if field != "candidate_id"]
    return schema


def validation_schema() -> dict[str, Any]:
    """Return the exact JSON schema supplied to v1 validation requests."""
    return ValidationResult.model_json_schema()


def _audit_path(work_dir: Path, cluster_id: str) -> Path:
    return work_dir / "traces" / f"{cluster_id}.json"


def _write_trace(work_dir: Path | None, state: ClusterState) -> None:
    if work_dir is None:
        return
    path = _audit_path(work_dir, state["cluster_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_graph(
    *, client: JsonChatClient, model: str, temperature: float, retries: int,
    work_dir: Path | None = None,
):
    annotation_system, validation_system, _ = systems()

    async def annotation_batch(state: ClusterState) -> dict[str, Any]:
        round_number = state["round_count"] + 1

        async def one_candidate(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
            candidate_id = f"r{round_number}_a{index}"
            request = {
                "candidate_id": candidate_id,
                "species": state["species"],
                "tissue": state["tissue"],
                "ordered_markers": state["markers"],
                "revision_feedback": state["revision_feedback"],
            }
            parsed, raw = await retry_json(
                client,
                system=annotation_system,
                user=annotation_user(request),
                schema=candidate_schema(),
                model=model,
                temperature=temperature,
                retries=retries,
                validator=lambda value: Candidate.model_validate({**value, "candidate_id": candidate_id}),
            )
            candidate = Candidate.model_validate({**parsed, "candidate_id": candidate_id})
            return candidate.model_dump(), {"candidate_id": candidate_id, "raw_response": raw}

        results = await asyncio.gather(
            *(one_candidate(index) for index in range(1, ANNOTATION_CANDIDATES + 1))
        )
        candidates, raw_responses = zip(*results)
        return {"current_candidates": list(candidates), "annotation_raw": list(raw_responses)}

    async def validation(state: ClusterState) -> dict[str, Any]:
        candidates = state["current_candidates"]
        if (
            len(candidates) != ANNOTATION_CANDIDATES
            or len({candidate["candidate_id"] for candidate in candidates}) != ANNOTATION_CANDIDATES
        ):
            raise ValueError(
                f"Annotation batch must contain exactly {ANNOTATION_CANDIDATES} uniquely identified candidates"
            )
        request = {
            "species": state["species"],
            "tissue": state["tissue"],
            "ordered_markers": state["markers"],
            "candidates": candidates,
            "prior_revision_feedback": state["revision_feedback"],
        }
        parsed, raw = await retry_json(
            client,
            system=validation_system,
            user=validation_user(request),
            schema=validation_schema(),
            model=model,
            temperature=temperature,
            retries=retries,
            validator=ValidationResult.model_validate,
        )
        result = ValidationResult.model_validate(parsed)
        ids = {candidate["candidate_id"] for candidate in candidates}
        review_ids = {review.candidate_id for review in result.candidate_reviews}
        if review_ids != ids:
            raise ValueError("Validation must review exactly the five submitted candidates")
        if result.decision == "pass" and result.selected_candidate_id not in ids:
            raise ValueError("Validation selected an unknown candidate")
        validation_value = result.model_dump()
        history = list(state["round_history"])
        history.append({
            "round": state["round_count"] + 1,
            "candidates": candidates,
            "validation": validation_value,
            "annotation_raw_responses": state["annotation_raw"],
            "validation_raw_response": raw,
        })
        return {
            "validation": validation_value,
            "round_count": state["round_count"] + 1,
            "round_history": history,
            "revision_feedback": result.revision_feedback,
        }

    def route_after_validation(state: ClusterState) -> Literal["formatting", "annotation_batch"]:
        assert state["validation"] is not None
        if state["validation"]["decision"] == "pass" or state["round_count"] >= MAX_ROUNDS:
            return "formatting"
        return "annotation_batch"

    def formatting(state: ClusterState) -> dict[str, Any]:
        validation_value = state["validation"]
        assert validation_value is not None
        if validation_value["decision"] == "pass":
            selected = next(
                candidate for candidate in state["current_candidates"]
                if candidate["candidate_id"] == validation_value["selected_candidate_id"]
            )
            record = {
                "status": "validated",
                "prediction": selected["label"],
                "selected_candidate_id": selected["candidate_id"],
                "validation_summary": validation_value["selection_reason"],
                "candidate": selected,
            }
            update = {"status": "validated", "final_record": record}
        else:
            selected, fallback = select_fallback_candidate(state["current_candidates"], state["markers"])
            record = {
                "status": "fallback",
                "prediction": selected["label"],
                "selected_candidate_id": selected["candidate_id"],
                "revision_feedback": validation_value["revision_feedback"],
                "candidate": selected,
                "fallback": fallback,
                "validation": validation_value,
            }
            update = {"status": "fallback", "final_record": record}
        final_state = {**state, **update}
        _write_trace(work_dir, final_state)
        return update

    graph = StateGraph(ClusterState)
    graph.add_node("annotation_batch", annotation_batch)
    graph.add_node("validation", validation)
    graph.add_node("formatting", formatting)
    graph.add_edge(START, "annotation_batch")
    graph.add_edge("annotation_batch", "validation")
    graph.add_conditional_edges("validation", route_after_validation)
    graph.add_edge("formatting", END)
    return graph.compile()


def annotate_cluster(
    *, cluster_id: str, dataset: str, species: str, tissue: str, markers: list[str],
    client: JsonChatClient, model: str, temperature: float, retries: int,
    work_dir: Path | None = None, prompt_version: str = PROMPT_VERSION,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    if not cluster_id or not dataset or not species or not tissue or not markers or not model or not prompt_version:
        raise ValueError("cluster_id, dataset, species, tissue, markers, model, and prompt_version must be non-empty")
    if protocol_version != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported CellClick protocol_version {protocol_version!r}; supported: {PROTOCOL_VERSION}"
        )
    if prompt_version != PROMPT_VERSION:
        raise ValueError(f"CellClick {protocol_version} requires prompt_version={PROMPT_VERSION!r}")
    state: ClusterState = {
        "cluster_id": cluster_id,
        "dataset": dataset,
        "species": species,
        "model": model,
        "prompt_version": prompt_version,
        "protocol_version": protocol_version,
        "tissue": tissue,
        "markers": markers,
        "round_count": 0,
        "current_candidates": [],
        "validation": None,
        "round_history": [],
        "final_record": None,
        "status": "running",
        "revision_feedback": None,
        "annotation_raw": [],
    }
    async def run_graph() -> dict[str, Any]:
        try:
            return await build_graph(
                client=client,
                model=model,
                temperature=temperature,
                retries=retries,
                work_dir=work_dir,
            ).ainvoke(state)
        finally:
            # AsyncOpenAI's transport is attached to this loop.  It must be
            # closed before asyncio.run() tears the loop down.
            await close_json_client(client)

    result = asyncio.run(run_graph())
    return dict(result)
