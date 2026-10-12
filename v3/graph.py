"""CellClick v3 marker-first, contrastive annotation workflow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from cellclick.llm import JsonChatClient, close_json_client
from cellclick.retry import retry_json

from .models import Audit, Candidate, Judgment
from .prompts import (AUDIT_SYSTEM, JUDGE_SYSTEM, PROMPT_VERSION,
                      candidate_payload, candidate_system, user)
from .rag import DEFAULT_TOP_K, default_reference_path, load_retriever


PROTOCOL_VERSION = "v3"
ANNOTATION_CANDIDATES = 3
MAX_ROUNDS = 1
REQUEST_RETRIES = 2
MAX_PARALLEL_CLUSTERS = 1
WORKFLOW_VERSION = "marker-first-dataset-relative-comparison-v3"
UNRESOLVED_PREDICTION_POLICY = "broad-lineage-before-unresolved-v3"
CANDIDATE_ROLES = ("lineage-first", "subtype-conservative", "adversarial-conflict")


def build_graph(**kwargs: Any) -> None:
    del kwargs
    raise NotImplementedError("v3 uses annotate_cluster")


def _schema(response_model: Any) -> dict[str, Any]:
    return response_model.model_json_schema()


def _validate_markers(value: dict[str, Any], markers: list[str], response_model: Any) -> Any:
    canonical = {marker.strip().casefold(): marker.strip() for marker in markers}
    marker_fields = (
        "lineage_evidence", "observed_conflicts", "generic_markers", "supporting_markers",
        "subtype_specific_markers", "competing_markers", "winner_support", "runner_up_support",
        "decisive_markers", "unresolved_conflicts", "fatal_conflicts",
    )
    for field in marker_fields:
        normalized = []
        for marker in value.get(field, []):
            marker_text = str(marker).strip()
            marker_key = marker_text.casefold()
            matched = canonical.get(marker_key)
            if matched is None:
                matches = [
                    original for key, original in canonical.items()
                    if marker_key.startswith(key)
                    and marker_key[len(key):len(key) + 1] in {" ", ":", "(", "["}
                ]
                matched = matches[0] if len(matches) == 1 else None
            if matched is None:
                raise ValueError(f"{field} cites marker not present in input: {marker}")
            if matched not in normalized:
                normalized.append(matched)
        if field in value:
            value[field] = normalized
    return response_model.model_validate(value)


def _validate_candidate(value: dict[str, Any], markers: list[str], role: str) -> Candidate:
    candidate = _validate_markers(value, markers, Candidate)
    if candidate.role != role:
        raise ValueError(f"candidate returned role {candidate.role!r}; expected {role!r}")
    return candidate


def _validate_retrieval_ranking(value: dict[str, Any], retrieval: dict[str, Any]) -> None:
    expected = set(range(1, len(retrieval.get("hits", [])) + 1))
    ranking = [int(rank) for rank in value.get("retrieval_ranking", [])]
    if len(ranking) != len(set(ranking)):
        raise ValueError("retrieval_ranking contains duplicate ranks")
    if set(ranking) != expected:
        raise ValueError("retrieval_ranking must contain every retrieval rank exactly once")


def _relative_context(
    cluster_id: str, markers: list[str], dataset_context: list[dict[str, Any]],
) -> dict[str, Any]:
    marker_sets = {
        str(item["cluster_id"]): {str(gene).strip().casefold() for gene in item["ordered_markers"]}
        for item in dataset_context
    }
    target = {marker.casefold(): marker for marker in markers}
    peers = []
    for peer_id, peer_markers in marker_sets.items():
        if peer_id == cluster_id:
            continue
        shared = [original for key, original in target.items() if key in peer_markers]
        if shared:
            peers.append({"cluster_id": peer_id, "shared_target_markers": shared})
    return {
        "cluster_count": len(marker_sets),
        "target_marker_prevalence": [
            {
                "marker": original,
                "clusters_containing_marker": sum(key in genes for genes in marker_sets.values()),
            }
            for key, original in target.items()
        ],
        "peer_overlaps": peers,
    }


async def _complete(
    client: JsonChatClient, *, system: str, payload: dict[str, Any], schema: dict[str, Any],
    model: str, temperature: float, retries: int,
    validator: Callable[[dict[str, Any]], Any],
) -> tuple[dict[str, Any], str]:
    return await retry_json(
        client, system=system, user=user(payload), schema=schema, model=model,
        temperature=temperature, retries=retries, validator=validator,
    )


def _write_trace(work_dir: Path | None, cluster_id: str, trace: dict[str, Any]) -> None:
    if work_dir is None:
        return
    path = work_dir / "traces" / f"{cluster_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _run_workflow(
    *, cluster_id: str, dataset: str, tissue: str, markers: list[str], species: str,
    client: JsonChatClient, model: str, temperature: float, retries: int,
    work_dir: Path | None, reference_path: Path, rag_top_k: int,
    embedding_cache_path: str | Path | None, dataset_context: list[dict[str, Any]],
) -> dict[str, Any]:
    base = {
        "species": species, "tissue": tissue, "target_cluster_id": cluster_id,
        "ordered_markers": markers,
        "unlabeled_dataset_context": _relative_context(cluster_id, markers, dataset_context),
    }
    audit_raw, audit_response = await _complete(
        client, system=AUDIT_SYSTEM, payload=base, schema=_schema(Audit), model=model,
        temperature=temperature, retries=retries,
        validator=lambda value: _validate_markers(value, markers, Audit),
    )
    audit = Audit.model_validate(audit_raw).model_dump()

    retriever = load_retriever(str(reference_path))
    del embedding_cache_path
    tissue_candidates = retriever.exact_tissue_context(tissue)
    retrieval = retriever.retrieve(
        markers, tissue, top_k=rag_top_k, tissue_candidates=tissue_candidates,
    ).prompt_dict()

    async def generate_candidate(role: str) -> tuple[dict[str, Any], str]:
        def validate_candidate(value: dict[str, Any]) -> Candidate:
            _validate_retrieval_ranking(value, retrieval)
            return _validate_candidate(value, markers, role)

        raw, response = await _complete(
            client, system=candidate_system(role),
            payload=candidate_payload(base, role=role, audit=audit, retrieval=retrieval),
            schema=_schema(Candidate), model=model, temperature=temperature, retries=retries,
            validator=validate_candidate,
        )
        return Candidate.model_validate(raw).model_dump(), response

    candidate_pairs = await asyncio.gather(*(generate_candidate(role) for role in CANDIDATE_ROLES))
    candidates = [candidate for candidate, _ in candidate_pairs]
    def validate_judgment(value: dict[str, Any]) -> Judgment:
        _validate_retrieval_ranking(value, retrieval)
        return _validate_markers(value, markers, Judgment)

    judgment_raw, judgment_response = await _complete(
        client, system=JUDGE_SYSTEM,
        payload={**base, "evidence_audit": audit, "reference_retrieval": retrieval, "candidates": candidates},
        schema=_schema(Judgment), model=model, temperature=temperature, retries=retries,
        validator=validate_judgment,
    )
    judgment = Judgment.model_validate(judgment_raw).model_dump()
    status = "validated" if judgment["decision"] == "validated" and judgment.get("label") else "unresolved"
    record = {
        "status": status, "prediction": judgment.get("label") if status == "validated" else None,
        "judgment": judgment, "audit": audit, "reference_retrieval": retrieval,
        "candidates": candidates,
        "call_counts": {"audit": 1, "candidates": len(candidates), "judgment": 1, "total": 5},
    }
    _write_trace(work_dir, cluster_id, {
        "cluster_id": cluster_id, "dataset": dataset, "tissue": tissue, "markers": markers,
        "species": species, "model": model, "prompt_version": PROMPT_VERSION,
        "protocol_version": PROTOCOL_VERSION, "workflow_version": WORKFLOW_VERSION, **record,
        "raw_responses": {"audit": audit_response,
                          "candidates": [response for _, response in candidate_pairs],
                          "judgment": judgment_response},
    })
    return {"status": status, "round_count": 1, "final_record": record}


def annotate_cluster(
    *, cluster_id: str, dataset: str, species: str, tissue: str, markers: list[str],
    client: JsonChatClient, model: str, temperature: float, retries: int,
    work_dir: Path | None = None, prompt_version: str = PROMPT_VERSION,
    protocol_version: str = PROTOCOL_VERSION, rag_reference_path: str | Path | None = None,
    rag_top_k: int = DEFAULT_TOP_K, embedding_cache_path: str | Path | None = None,
    dataset_context: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    values = (cluster_id, dataset, species, tissue, model, prompt_version, protocol_version)
    if not markers or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("cluster_id, dataset, species, tissue, markers, model, prompt_version, and protocol_version must be non-empty")
    if prompt_version != PROMPT_VERSION or protocol_version != PROTOCOL_VERSION:
        raise ValueError("CellClick v3 requires prompt_version/protocol_version=v3")
    reference_path = Path(rag_reference_path) if rag_reference_path else default_reference_path()
    context = dataset_context or [{"cluster_id": cluster_id, "ordered_markers": markers}]
    if not any(item.get("cluster_id") == cluster_id for item in context):
        raise ValueError("dataset_context must contain the target cluster")

    async def run() -> dict[str, Any]:
        try:
            return await _run_workflow(
                cluster_id=cluster_id, dataset=dataset, tissue=tissue, markers=markers,
                species=species, client=client, model=model, temperature=temperature,
                retries=retries, work_dir=work_dir, reference_path=reference_path,
                rag_top_k=rag_top_k, embedding_cache_path=embedding_cache_path,
                dataset_context=context,
            )
        finally:
            await close_json_client(client)

    return asyncio.run(run())
