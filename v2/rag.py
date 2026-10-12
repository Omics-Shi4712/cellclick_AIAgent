"""Replaceable marker-reference retrieval and scoring for CellClick v2.

The retriever is deliberately independent of prompts and workflow orchestration:
callers can replace its scoring strategy without changing the V2 agent flow.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


RETRIEVAL_VERSION = "ranked-marker-idf-pmid-tissue-v1"
DEFAULT_TOP_K = 5
_REQUIRED_COLUMNS = {
    "tissue", "pmid", "marker", "normalised_uberon_id", "tissue_ontology_name",
    "normalised_cl_id", "ontology_name",
}


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    label: str
    cl_id: str
    tissue: str
    tissue_id: str
    score: float
    tissue_match: str
    matched_input_markers: tuple[str, ...]
    supporting_pmids: tuple[str, ...]

    def prompt_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalPacket:
    retrieval_version: str
    reference_path: str
    reference_sha256: str
    top_k: int
    query_tissue: str
    hits: tuple[RetrievalHit, ...]

    def prompt_dict(self) -> dict[str, object]:
        return {
            "retrieval_version": self.retrieval_version,
            "reference_sha256": self.reference_sha256,
            "query_tissue": self.query_tissue,
            "hits": [hit.prompt_dict() for hit in self.hits],
        }


@dataclass
class _ReferenceDocument:
    label: str
    cl_id: str
    tissue: str
    tissue_id: str
    markers: dict[str, set[str]]


def default_reference_path() -> Path:
    return Path(__file__).resolve().parents[1] / "ref" / "marker" / "human_cell_marker_filtered_v1_main.tsv"


def reference_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(value: str) -> str:
    return " ".join(value.strip().casefold().split())


class MarkerReferenceRetriever:
    """In-memory inverted index over tissue/cell-type marker documents."""

    def __init__(self, *, path: Path, source_sha256: str, documents: dict[tuple[str, str], _ReferenceDocument], marker_document_frequency: dict[str, int]) -> None:
        self.path = path
        self.source_sha256 = source_sha256
        self._documents = documents
        self._marker_document_frequency = marker_document_frequency
        self._document_count = len(documents)

    @classmethod
    def from_tsv(cls, path: Path) -> "MarkerReferenceRetriever":
        path = path.resolve()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or not _REQUIRED_COLUMNS.issubset(reader.fieldnames):
                raise ValueError(f"RAG reference TSV lacks required columns: {sorted(_REQUIRED_COLUMNS)}")
            documents: dict[tuple[str, str], _ReferenceDocument] = {}
            for row in reader:
                marker = _normalise(row["marker"])
                cl_id = row["normalised_cl_id"].strip()
                label = row["ontology_name"].strip()
                tissue = row["tissue_ontology_name"].strip() or row["tissue"].strip()
                tissue_id = row["normalised_uberon_id"].strip()
                if not marker or not cl_id or not label or not tissue:
                    continue
                key = (tissue_id or _normalise(tissue), cl_id)
                document = documents.setdefault(key, _ReferenceDocument(label, cl_id, tissue, tissue_id, {}))
                document.markers.setdefault(marker, set()).add(row["pmid"].strip())
        marker_document_frequency: dict[str, int] = defaultdict(int)
        for document in documents.values():
            for marker in document.markers:
                marker_document_frequency[marker] += 1
        return cls(path=path, source_sha256=reference_sha256(path), documents=documents, marker_document_frequency=dict(marker_document_frequency))

    def retrieve(self, markers: Iterable[str], tissue: str, *, top_k: int = DEFAULT_TOP_K) -> RetrievalPacket:
        if top_k < 1:
            raise ValueError("RAG top_k must be positive")
        query = [_normalise(marker) for marker in markers]
        query = [marker for marker in query if marker]
        query_tissue = _normalise(tissue)
        ranked: list[tuple[float, _ReferenceDocument, list[str], str]] = []
        for document in self._documents.values():
            matched = [marker for marker in query if marker in document.markers]
            if not matched:
                continue
            score = 0.0
            for position, marker in enumerate(query):
                if marker not in document.markers:
                    continue
                rank_weight = len(query) - position
                idf = math.log((1 + self._document_count) / (1 + self._marker_document_frequency[marker])) + 1.0
                pmid_weight = 1.0 + math.log1p(len(document.markers[marker]))
                score += rank_weight * idf * pmid_weight
            tissue_match = "exact" if _normalise(document.tissue) == query_tissue else "other"
            if tissue_match == "exact":
                score *= 1.05
            ranked.append((score, document, matched, tissue_match))
        ranked.sort(key=lambda item: (-item[0], item[1].label.casefold(), item[1].cl_id))
        hits: list[RetrievalHit] = []
        for rank, (score, document, matched, tissue_match) in enumerate(ranked[:top_k], start=1):
            pmids = sorted({pmid for marker in matched for pmid in document.markers[marker] if pmid})
            hits.append(RetrievalHit(rank, document.label, document.cl_id, document.tissue, document.tissue_id, round(score, 6), tissue_match, tuple(matched), tuple(pmids)))
        return RetrievalPacket(RETRIEVAL_VERSION, str(self.path), self.source_sha256, top_k, tissue, tuple(hits))


@lru_cache(maxsize=4)
def load_retriever(path: str) -> MarkerReferenceRetriever:
    """Cache a reference index per Python process; callers may supply another path."""
    return MarkerReferenceRetriever.from_tsv(Path(path))
