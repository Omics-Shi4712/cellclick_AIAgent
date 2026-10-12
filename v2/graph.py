"""Evidence-first, bounded CellClick v2 annotation workflow."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cellclick.llm import JsonChatClient, close_json_client
from cellclick.retry import retry_json

from .models import Audit, Candidate, Judgment
from .prompts import AUDIT_SYSTEM, CANDIDATE_SYSTEM, JUDGE_SYSTEM, PROMPT_VERSION, user
from .rag import DEFAULT_TOP_K, default_reference_path, load_retriever


PROTOCOL_VERSION = "v2"
ANNOTATION_CANDIDATES = 3
MAX_ROUNDS = 1
REQUEST_RETRIES = 2
MAX_PARALLEL_CLUSTERS = 1
WORKFLOW_VERSION = "evidence-audit-role-candidates-contrastive-judge-v2"
UNRESOLVED_PREDICTION_POLICY = "null-v2"
CANDIDATE_ROLES = ("lineage-first", "subtype-conservative", "adversarial-conflict")


def heuristic_prediction(markers: list[str], tissue: str) -> tuple[str, str]:
    """Conservative marker-only rescue rule used by the caller, never by V2 itself."""
    del tissue
    genes = {str(marker).strip().upper() for marker in markers}

    def hit(values: set[str]) -> int:
        return len(genes & values)

    if hit({"CD3D", "CD3E", "CD3G", "TRAC", "BCL11B", "IL7R", "CCR7", "CD8A"}) >= 2:
        return ("Naive T Cell" if hit({"CCR7", "IL7R"}) >= 2 else "T cell", "immune:T-cell")
    if hit({"GNLY", "NKG7", "KLRD1", "NCR1", "PRF1", "GZMB"}) >= 2: return "Natural Killer (NK) Cell", "immune:NK"
    if hit({"MZB1", "JCHAIN", "XBP1", "SEC11C"}) >= 2: return "Plasma Cell", "immune:plasma"
    if hit({"MS4A1", "CD79A", "CD79B", "CD74", "HLA-DRA"}) >= 2: return "B Cell", "immune:B-cell"
    if hit({"FCGR3B", "CSF3R", "CD177", "S100A12", "MMP9"}) >= 2: return "Neutrophil", "myeloid:neutrophil"
    if hit({"TYROBP", "LCP1", "AIF1", "CTSS", "CD163", "CST3", "LYZ", "PSAP"}) >= 2:
        return ("Macrophage" if hit({"TYROBP", "LCP1", "CD163", "CTSB"}) >= 2 else "Monocyte", "myeloid")
    if hit({"PTPRC"}) >= 1 or hit({"IL7R", "CD69", "CYSTM1"}) >= 2: return "T cell", "immune:T-cell"
    if hit({"HBB", "HBA1", "HBA2", "ALAS2", "GYPA"}) >= 2: return "Erythrocyte", "erythroid"
    if hit({"PPBP", "PF4", "NRGN", "GNG11"}) >= 2: return "Platelet", "megakaryocyte"
    if hit({"PRSS57", "SOX4", "CDK6", "RACK1"}) >= 2: return "Hematopoietic Stem Cell", "hematopoietic"
    if hit({"VWF", "FLT1", "KDR", "EMCN", "PECAM1"}) >= 2: return "Endothelial Cell", "endothelial"
    if hit({"TTN", "ACTC1", "MYH6", "TNNT2", "TNNI1", "MYL7"}) >= 2: return "Cardiomyocyte", "muscle"
    if hit({"DCN", "PRELP", "COL1A1", "COL1A2", "C1S", "C1R"}) >= 2: return "Fibroblast", "mesenchymal"
    if hit({"RGS5", "PDGFRB", "CPE", "NDUFA4L2"}) >= 2: return "Smooth Muscle Cell", "mesenchymal"
    if hit({"ALDOB", "PDZK1IP1", "FXYD2", "PCK1", "ALDH1A1", "S100A1", "GPX3"}) >= 2: return "Kidney Epithelial Cell", "epithelial"
    if hit({"EPCAM", "KRT8", "KRT18", "CD24"}) >= 2: return "Epithelial Cell", "epithelial"
    return "Unresolved", "unresolved"


def _schema(response_model: type[Audit] | type[Candidate] | type[Judgment]) -> dict[str, Any]:
    return response_model.model_json_schema()


def _validate_markers(value: dict[str, Any], markers: list[str], response_model: Any) -> Any:
    allowed = {marker.strip().casefold() for marker in markers}
    for field in ("lineage_evidence", "observed_conflicts", "generic_markers", "supporting_markers", "decisive_markers", "fatal_conflicts"):
        for marker in value.get(field, []):
            if str(marker).strip().casefold() not in allowed:
                raise ValueError(f"{field} cites marker not present in input: {marker}")
    return response_model.model_validate(value)


async def _complete(client: JsonChatClient, *, system: str, payload: dict[str, Any], schema: dict[str, Any], model: str, temperature: float, retries: int, validator: Callable[[dict[str, Any]], Any]) -> tuple[dict[str, Any], str]:
    return await retry_json(client, system=system, user=user(payload), schema=schema, model=model, temperature=temperature, retries=retries, validator=validator)


async def _audit_evidence(client: JsonChatClient, *, base: dict[str, Any], markers: list[str], model: str, temperature: float, retries: int) -> tuple[dict[str, Any], str]:
    raw, response = await _complete(client, system=AUDIT_SYSTEM, payload=base, schema=_schema(Audit), model=model, temperature=temperature, retries=retries, validator=lambda value: _validate_markers(value, markers, Audit))
    return Audit.model_validate(raw).model_dump(), response


async def _generate_candidates(client: JsonChatClient, *, base: dict[str, Any], audit: dict[str, Any], retrieval: dict[str, Any], markers: list[str], model: str, temperature: float, retries: int) -> tuple[list[dict[str, Any]], list[str]]:
    async def generate(role: str) -> tuple[dict[str, Any], str]:
        raw, response = await _complete(client, system=CANDIDATE_SYSTEM, payload={**base, "evidence_audit": audit, "reference_retrieval": retrieval, "role": role}, schema=_schema(Candidate), model=model, temperature=temperature, retries=retries, validator=lambda value: _validate_markers(value, markers, Candidate))
        return Candidate.model_validate(raw).model_dump(), response

    results = await asyncio.gather(*(generate(role) for role in CANDIDATE_ROLES))
    return [candidate for candidate, _ in results], [response for _, response in results]


async def _judge_candidates(client: JsonChatClient, *, base: dict[str, Any], audit: dict[str, Any], retrieval: dict[str, Any], candidates: list[dict[str, Any]], markers: list[str], model: str, temperature: float, retries: int) -> tuple[dict[str, Any], str]:
    raw, response = await _complete(client, system=JUDGE_SYSTEM, payload={**base, "evidence_audit": audit, "reference_retrieval": retrieval, "candidates": candidates}, schema=_schema(Judgment), model=model, temperature=temperature, retries=retries, validator=lambda value: _validate_markers(value, markers, Judgment))
    return Judgment.model_validate(raw).model_dump(), response


def _write_trace(work_dir: Path | None, cluster_id: str, state: dict[str, Any]) -> None:
    if work_dir is not None:
        path = work_dir / "traces" / f"{cluster_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_input(*, cluster_id: str, dataset: str, species: str, tissue: str, markers: list[str], model: str, prompt_version: str, protocol_version: str) -> None:
    values = (cluster_id, dataset, species, tissue, model, prompt_version, protocol_version)
    if not markers or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("cluster_id, dataset, species, tissue, markers, model, prompt_version, and protocol_version must be non-empty")
    if protocol_version != PROTOCOL_VERSION or prompt_version != PROMPT_VERSION:
        raise ValueError("CellClick v2 requires protocol_version/prompt_version=v2")


async def _run_workflow(*, cluster_id: str, dataset: str, species: str, tissue: str, markers: list[str], client: JsonChatClient, model: str, temperature: float, retries: int, work_dir: Path | None, rag_reference_path: Path, rag_top_k: int) -> dict[str, Any]:
    base = {"species": species, "tissue": tissue, "ordered_markers": markers}
    audit, audit_response = await _audit_evidence(client, base=base, markers=markers, model=model, temperature=temperature, retries=retries)
    retrieval_packet = load_retriever(str(rag_reference_path.resolve())).retrieve(markers, tissue, top_k=rag_top_k)
    retrieval = retrieval_packet.prompt_dict()
    candidates, candidate_responses = await _generate_candidates(client, base=base, audit=audit, retrieval=retrieval, markers=markers, model=model, temperature=temperature, retries=retries)
    judgment, judgment_response = await _judge_candidates(client, base=base, audit=audit, retrieval=retrieval, candidates=candidates, markers=markers, model=model, temperature=temperature, retries=retries)
    status = "validated" if judgment["decision"] == "validated" and judgment.get("label") else "unresolved"
    call_counts = {"audit": 1, "candidates": len(candidates), "judgment": 1, "total": 2 + len(candidates)}
    record = {"status": status, "prediction": judgment["label"] if status == "validated" else None, "judgment": judgment, "audit": audit, "reference_retrieval": retrieval, "candidates": candidates, "call_counts": call_counts}
    _write_trace(work_dir, cluster_id, {"cluster_id": cluster_id, "dataset": dataset, "species": species, "model": model, "prompt_version": PROMPT_VERSION, "protocol_version": PROTOCOL_VERSION, "workflow_version": WORKFLOW_VERSION, "tissue": tissue, "markers": markers, **record, "raw_responses": {"audit": audit_response, "candidates": candidate_responses, "judgment": judgment_response}})
    return {"status": status, "round_count": 1, "final_record": record}


def annotate_cluster(*, cluster_id: str, dataset: str, species: str, tissue: str, markers: list[str], client: JsonChatClient, model: str, temperature: float, retries: int, work_dir: Path | None = None, prompt_version: str = PROMPT_VERSION, protocol_version: str = PROTOCOL_VERSION, rag_reference_path: str | Path | None = None, rag_top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Run V2's fixed audit, role-specific candidate, and judgment sequence."""
    _validate_input(cluster_id=cluster_id, dataset=dataset, species=species, tissue=tissue, markers=markers, model=model, prompt_version=prompt_version, protocol_version=protocol_version)
    reference_path = Path(rag_reference_path) if rag_reference_path is not None else default_reference_path()

    async def run_workflow() -> dict[str, Any]:
        try:
            return await _run_workflow(cluster_id=cluster_id, dataset=dataset, species=species, tissue=tissue, markers=markers, client=client, model=model, temperature=temperature, retries=retries, work_dir=work_dir, rag_reference_path=reference_path, rag_top_k=rag_top_k)
        finally:
            # AsyncOpenAI's transport is attached to this loop.  It must be
            # closed before asyncio.run() tears the loop down.
            await close_json_client(client)

    return asyncio.run(run_workflow())


def build_graph(**kwargs: Any) -> None:
    """Compatibility placeholder; V2's public workflow is ``annotate_cluster``."""
    del kwargs
    raise NotImplementedError("v2 uses the evidence-first bounded workflow; annotate_cluster is the public entry point")
