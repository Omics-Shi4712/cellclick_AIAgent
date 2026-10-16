"""Candidate normalization and registration for CellClick V5."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from .models import CandidateProposal, CandidateRegistry

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
EMBEDDING_CACHE = CACHE_DIR / "cell_name_emebdding.json"
MAPPING_CACHE = CACHE_DIR / "label_to_clid.tsv"
CL_ID = re.compile(r"^CL:\d+$")
_INDEX_CACHE: dict[tuple[str, str], "CLEmbeddingIndex"] = {}
_QUERY_CACHE: dict[tuple[str, str], list[float]] = {}
_CL_NAMES_CACHE: dict[str, dict[str, str]] = {}

def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

def _valid_cl(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and CL_ID.fullmatch(value.strip()) else None

def _load_map(path: Path = MAPPING_CACHE) -> dict[str, dict[str, str]]:
    if not path.is_file(): return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {normalize_label(row.get("label_normalized") or row.get("label_original") or ""): row
                for row in csv.DictReader(handle, delimiter="\t") if normalize_label(row.get("label_normalized") or row.get("label_original") or "")}

def _save_map(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    os.close(fd)
    try:
        with open(name, "w", encoding="utf-8", newline="") as handle:
            fields = ("label_normalized", "label_original", "cl_id", "mapping_status", "canonical_label", "score", "margin", "model", "policy_version")
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t"); writer.writeheader()
            for key in sorted(rows): writer.writerow({field: rows[key].get(field, "") for field in fields})
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def _load_cl_names(obo_path: Path) -> dict[str, str]:
    cache_key = str(obo_path.resolve())
    if cache_key in _CL_NAMES_CACHE:
        return _CL_NAMES_CACHE[cache_key]
    names: dict[str, str] = {}; current = None
    if not obo_path.is_file():
        _CL_NAMES_CACHE[cache_key] = names
        return names
    for line in obo_path.read_text(encoding="utf-8").splitlines() + [""]:
        if line == "[Term]": current = {}
        elif current is not None and line.startswith("id: "): current["id"] = line[4:].strip()
        elif current is not None and line.startswith("name: "): current["name"] = line[6:].strip()
        elif not line and current and CL_ID.fullmatch(str(current.get("id", ""))):
            names[current["id"]] = str(current.get("name", current["id"])); current = None
    _CL_NAMES_CACHE[cache_key] = names
    return names

def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / (na*nb) if na and nb else 0.0

class CLEmbeddingIndex:
    """One-process CL canonical-label embedding index."""
    def __init__(self, embedding_path: Path, obo_path: Path) -> None:
        started = time.monotonic()
        data = json.loads(embedding_path.read_text(encoding="utf-8")) if embedding_path.is_file() else {}
        names = _load_cl_names(obo_path)
        self.labels: list[str] = []
        self.cl_ids: list[str] = []
        self.vectors: Any = None
        rows = [(cid, name, data.get(name)) for cid, name in names.items() if isinstance(data.get(name), list)]
        if not rows:
            raise ValueError("No canonical CL embeddings are available")
        self.cl_ids = [row[0] for row in rows]
        self.labels = [row[1] for row in rows]
        try:
            import numpy as np
            matrix = np.asarray([row[2] for row in rows], dtype=float)
            norms = np.linalg.norm(matrix, axis=1)
            self.vectors = matrix / np.maximum(norms[:, None], 1e-12)
        except (ImportError, ValueError):
            self.vectors = [[float(v) for v in row[2]] for row in rows]
        self.load_seconds = time.monotonic() - started

    def best_matches(self, query: list[float], limit: int = 2) -> list[tuple[float, str, str]]:
        try:
            import numpy as np
            vector = np.asarray(query, dtype=float)
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
            scores = self.vectors @ vector
            order = np.argsort(-scores, kind="stable")[:limit]
            return [(float(scores[i]), self.cl_ids[int(i)], self.labels[int(i)]) for i in order]
        except (ImportError, TypeError, ValueError):
            scored = sorted(
                ((_cosine(query, vector), cid, label)
                 for cid, label, vector in zip(self.cl_ids, self.labels, self.vectors)),
                key=lambda item: item[0], reverse=True,
            )
            return scored[:limit]


def _get_index(embedding_path: Path, obo_path: Path) -> CLEmbeddingIndex:
    key = (str(embedding_path.resolve()), str(obo_path.resolve()))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = CLEmbeddingIndex(embedding_path, obo_path)
    return _INDEX_CACHE[key]


def _embedding(label: str, *, cache: Path, model: str, api_key_env: str, timeout: float = 30.0) -> list[float]:
    cache_key = (model, label)
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]
    api_key = os.environ.get(api_key_env)
    if not api_key: raise RuntimeError(f"{api_key_env} is required for embedding cache miss")
    import urllib.request
    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openlux.ai/v1").rstrip("/")
    request = urllib.request.Request(endpoint + "/embeddings", data=json.dumps({"model": model, "input": [label]}).encode(), headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response: vector = json.load(response)["data"][0]["embedding"]
    _QUERY_CACHE[cache_key] = vector
    return vector

def map_label(label: str, *, mapping_path: Path = MAPPING_CACHE, embedding_path: Path = EMBEDDING_CACHE, obo_path: Path | None = None, threshold: float = .55, margin: float = 0.0, rows: dict[str, dict[str, str]] | None = None, index: CLEmbeddingIndex | None = None) -> dict[str, Any]:
    key = normalize_label(label); rows = rows if rows is not None else _load_map(mapping_path); row = rows.get(key)
    if row and _valid_cl(row.get("cl_id")):
        canonical = row.get("canonical_label", "")
        if not canonical:
            if obo_path is None:
                obo_path = Path(__file__).resolve().parents[4] / "evaluator" / "ref_data" / "cl.obo"
            canonical = _load_cl_names(obo_path).get(row["cl_id"], "")
        return {"cl_id": row["cl_id"], "mapping_status": "mapped", "canonical_label": canonical, "score": row.get("score"), "margin": row.get("margin")}
    # Mapping cache is authoritative when present but invalid IDs are surfaced.
    if row: return {"cl_id": None, "mapping_status": row.get("mapping_status") or "unmapped"}
    # Runtime embedding supplementation requires a local CL label index. If it is
    # unavailable, leave the candidate auditable rather than guessing.
    if obo_path is None: obo_path = Path(__file__).resolve().parents[4] / "evaluator" / "ref_data" / "cl.obo"
    names = _load_cl_names(obo_path)
    if not names: return {"cl_id": None, "mapping_status": "unmapped", "canonical_label": ""}
    try:
        index = index or _get_index(embedding_path, obo_path)
        query = _embedding(label, cache=embedding_path, model="text-embedding-3-small", api_key_env="OPENAI_API_KEY_LLM")
        scored = index.best_matches(query)
    except Exception:
        return {"cl_id": None, "mapping_status": "mapping_error", "canonical_label": ""}
    top = scored[0][0] if scored else 0.0; second = scored[1][0] if len(scored) > 1 else top; gap = top-second
    if top < threshold: status, clid = "unmapped", None
    elif gap < margin: status, clid = "ambiguous", None
    else: status, clid = "mapped", scored[0][1]
    canonical = next((item[2] for item in scored if item[1] == clid), "") if clid else ""
    rows[key] = {"label_normalized": key, "label_original": label, "cl_id": clid or "", "mapping_status": status, "canonical_label": canonical, "score": f"{top:.6f}", "margin": f"{gap:.6f}", "model": "text-embedding-3-small", "policy_version": "cellclick-v5.8-embedding"}
    _save_map(mapping_path, rows)
    return {"cl_id": clid, "mapping_status": status, "canonical_label": canonical, "score": top, "margin": gap}

def register_major(raw_candidates: Iterable[dict[str, Any]], *, version: int = 1, max_candidates: int = 5) -> CandidateRegistry:
    raw_candidates = list(raw_candidates)
    mapping_path = MAPPING_CACHE
    rows = _load_map(mapping_path)
    obo_path = Path(__file__).resolve().parents[4] / "evaluator" / "ref_data" / "cl.obo"
    pending = [str(raw.get("label", "")).strip() for raw in raw_candidates
               if str(raw.get("label", "")).strip()
               and normalize_label(str(raw.get("label", "")).strip()) not in rows]
    index = _get_index(EMBEDDING_CACHE, obo_path) if pending else None
    normalized: list[dict[str, Any]] = []
    for raw in raw_candidates:
        label = str(raw.get("label", "")).strip()
        if not label: continue
        mapped = map_label(label, rows=rows, index=index)
        normalized.append({**raw, **mapped, "original_label": label,
                           "canonical_label": mapped.get("canonical_label") or label})
    merged: list[dict[str, Any]] = []
    by_cl: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for item in normalized:
        key = item.get("cl_id") or ("name:" + normalize_label(item["original_label"]))
        target = by_cl.get(key) if item.get("cl_id") else by_name.get(key)
        if target is None:
            target = {**item, "original_labels": [item["original_label"]], "provenance": [{k: item.get(k) for k in ("source", "reference_rank", "rationale", "evidence_ids")} ]}
            merged.append(target)
            (by_cl if item.get("cl_id") else by_name)[key] = target
        else:
            target["original_labels"] = list(dict.fromkeys(target["original_labels"] + [item["original_label"]]))
            target["provenance"].append({k: item.get(k) for k in ("source", "reference_rank", "rationale", "evidence_ids")})
    merged = merged[:max_candidates]
    proposals = []
    for i, item in enumerate(merged, 1):
        proposals.append(CandidateProposal(candidate_id=f"H{i:02d}", label=item.get("canonical_label") or item["original_label"], granularity="major", provisional=True, source="MajorType", cl_id=item.get("cl_id"), mapping_status=item.get("mapping_status", "unmapped"), original_labels=item["original_labels"], provenance=item["provenance"]))
    ids = [x.candidate_id for x in proposals]
    return CandidateRegistry(version=version, candidates=proposals, parent_by_id={x: None for x in ids}, competing_by_id={x: [y for y in ids if y != x] for x in ids})


def register_subtype(raw_candidates: Iterable[dict[str, Any]], *, parent: CandidateProposal,
                     catalog: Iterable[dict[str, Any]], version: int = 2,
                     max_candidates: int = 5, usable_evidence: set[str] | None = None) -> CandidateRegistry:
    """Register directory-bound subtype proposals beneath ``parent``.

    Structural validation belongs here so invalid model output cannot enter
    the registry. Biological sufficiency remains the verifier's decision.
    """
    catalog_by_id = {
        str(item.get("cl_id")): item for item in catalog
        if isinstance(item, dict) and isinstance(item.get("cl_id"), str)
    }
    parent_cl_id = parent.cl_id
    merged: dict[str, dict[str, Any]] = {}
    for raw in raw_candidates:
        cl_id = _valid_cl(raw.get("cl_id"))
        if not cl_id or cl_id == parent_cl_id or cl_id not in catalog_by_id:
            continue
        evidence_ids = list(dict.fromkeys(
            item for item in raw.get("evidence_ids", [])
            if isinstance(item, str) and item.strip()
        ))
        rationale = str(raw.get("rationale", "")).strip()
        if not evidence_ids or not rationale:
            continue
        if usable_evidence is not None and not set(evidence_ids) <= usable_evidence:
            continue
        item = merged.setdefault(cl_id, {
            "cl_id": cl_id,
            "label": str(catalog_by_id[cl_id].get("label", cl_id)).strip() or cl_id,
            "evidence_ids": [],
            "rationale": [],
        })
        item["evidence_ids"] = list(dict.fromkeys(item["evidence_ids"] + evidence_ids))
        if rationale not in item["rationale"]:
            item["rationale"].append(rationale)

    children = list(merged.values())[:max_candidates]
    values = [parent]
    for offset, item in enumerate(children, 20):
        values.append(CandidateProposal(
            candidate_id=f"H{offset:02d}",
            label=item["label"],
            granularity="subtype",
            parent_id=parent.candidate_id,
            provisional=True,
            source="SubType",
            cl_id=item["cl_id"],
            mapping_status="mapped",
            original_labels=[item["label"]],
            provenance=[{
                "source": "SubType",
                "evidence_ids": item["evidence_ids"],
                "rationale": " ".join(item["rationale"]),
            }],
        ))
    ids = [item.candidate_id for item in values]
    return CandidateRegistry(
        version=version,
        candidates=values,
        parent_by_id={item.candidate_id: item.parent_id for item in values},
        competing_by_id={item.candidate_id: [other for other in ids
                                             if other not in {item.candidate_id, item.parent_id}]
                         for item in values},
    )
