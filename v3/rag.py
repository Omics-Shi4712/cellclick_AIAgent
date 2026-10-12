"""MGC and tissue-embedding retrieval for CellClick v3."""
from __future__ import annotations
import csv, hashlib, json, math, os
import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

RETRIEVAL_VERSION = "global-document-deduplicated-sqrt-mgc-v3"
DEFAULT_TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-small"
_REQUIRED = {"tissue", "pmid", "marker", "normalised_uberon_id", "tissue_ontology_name", "normalised_cl_id", "ontology_name"}

def _norm(v: str) -> str: return " ".join(str(v).strip().casefold().split())
def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

@dataclass(frozen=True)
class RetrievalHit:
    rank: int; label: str; cl_id: str; tissue: str; tissue_id: str
    score: float; gene_score: float; coverage_factor: float; support_factor: float
    tissue_cosine: float; tissue_weight: float
    matched_input_markers: tuple[str, ...]; supporting_pmids: tuple[str, ...]
    marker_evidence: tuple[dict[str, object], ...]
    matched_gene_count: int; matched_query_fraction: float
    def prompt_dict(self): return asdict(self)

@dataclass(frozen=True)
class RetrievalPacket:
    retrieval_version: str; reference_sha256: str; query_tissue: str
    selected_tissues: tuple[dict[str, object], ...]; hits: tuple[RetrievalHit, ...]
    cache_status: str
    def prompt_dict(self):
        return {"retrieval_version": self.retrieval_version, "reference_sha256": self.reference_sha256,
                "query_tissue": self.query_tissue, "selected_tissues": list(self.selected_tissues),
                "cache_status": self.cache_status, "hits": [h.prompt_dict() for h in self.hits]}

def default_reference_path() -> Path:
    return Path(__file__).resolve().parents[1] / "ref" / "marker" / "human_cell_marker_filtered_v1_main.tsv"

class MarkerReferenceRetriever:
    def __init__(self, path: Path, records: dict[tuple[str, str, str], dict[str, set[str]]], labels: dict[tuple[str, str], tuple[str, str]], counts: dict[str, int], tissue_ids: dict[str, str]):
        self.path, self.records, self.labels, self.n_gene, self.tissue_ids = path, records, labels, counts, tissue_ids
        self.reference_sha256 = _sha(path)
        self.n_docs = sum(len(v) for v in records.values())
        self.n_celltype: dict[tuple[str, str], int] = {}
        self.n_gene_ct: dict[tuple[str, str, str], int] = {}
        for (tid, cl, pmid), genes in records.items():
            self.n_celltype[(tid, cl)] = self.n_celltype.get((tid, cl), 0) + 1
            for gene in genes: self.n_gene_ct[(tid, cl, gene)] = self.n_gene_ct.get((tid, cl, gene), 0) + 1
        self.global_celltype: dict[str, int] = {}
        self.global_gene_celltype: dict[tuple[str, str], int] = {}
        self.global_labels: dict[str, str] = {}
        global_documents = defaultdict(set)
        for (tid, cl, pmid), genes in records.items():
            global_documents[(cl, pmid)].update(genes)
            self.global_labels.setdefault(cl, labels[(tid, cl)][0])
        for (cl, pmid), genes in global_documents.items():
            self.global_celltype[cl] = self.global_celltype.get(cl, 0) + 1
            for gene in genes:
                key = (cl, gene)
                self.global_gene_celltype[key] = self.global_gene_celltype.get(key, 0) + 1

    @classmethod
    def from_tsv(cls, path: Path):
        path = path.resolve(); records = {}; labels = {}; tissues = {}; gene_pmids = defaultdict(set)
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames or not _REQUIRED.issubset(reader.fieldnames): raise ValueError("RAG reference TSV lacks required columns")
            seen = set()
            for row in reader:
                gene, pmid, tid, cl = _norm(row["marker"]), row["pmid"].strip(), row["normalised_uberon_id"].strip(), row["normalised_cl_id"].strip()
                tissue = row["tissue_ontology_name"].strip() or row["tissue"].strip(); label = row["ontology_name"].strip()
                if not gene or not pmid or not tid or not cl or not label: continue
                key = (tid, cl, pmid, gene)
                if key in seen: continue
                seen.add(key); records.setdefault((tid, cl, pmid), {}).setdefault(gene, set()).add(pmid)
                labels[(tid, cl)] = (label, tissue); tissues[_norm(tissue)] = tid; gene_pmids[gene].add(pmid)
        return cls(path, records, labels, {g: len(p) for g, p in gene_pmids.items()}, tissues)

    def retrieve(self, markers: Iterable[str], tissue: str, *, top_k=DEFAULT_TOP_K, tissue_candidates: list[dict[str, object]] | None = None):
        query = list(dict.fromkeys(_norm(x) for x in markers if _norm(x)))
        selected = tissue_candidates if tissue_candidates is not None else self.exact_tissue_context(tissue)
        selected = selected[:3]
        scores = {}
        for cl, label in self.global_labels.items():
            evidence = []
            for gene in query:
                nct = self.global_gene_celltype.get((cl, gene), 0)
                ng = self.n_gene.get(gene, 0)
                nc = self.global_celltype.get(cl, 0)
                if nct and ng and nc:
                    evidence.append({"gene": gene, "mgc": (nct*nct)/(ng*nc), "n_gene_celltype": nct, "n_gene": ng, "n_celltype": nc})
            if not evidence:
                continue
            gs = sum(math.sqrt(float(x["mgc"])) for x in evidence)
            matched_fraction = len(evidence) / len(query) if query else 0.0
            coverage_factor = 0.25 + 0.75 * math.sqrt(matched_fraction)
            support_factor = 1.0
            score = gs
            scores[(cl,)] = {"cl": cl, "label": label, "tissue": "all tissues",
                             "tissue_id": "global", "cosine": 0.0, "weight": 1.0,
                             "gene_score": gs, "coverage_factor": coverage_factor,
                             "support_factor": support_factor, "score": score,
                             "evidence": evidence}
        ranked = sorted(scores.values(), key=lambda x: (-x["score"], -x["gene_score"], x["cl"]))[:top_k]
        hits = []
        for i, x in enumerate(ranked, 1):
            evidence_genes = {str(e["gene"]) for e in x["evidence"]}
            pmids = sorted({pmid for (tid, cl, pmid), genes in self.records.items()
                            if cl == x["cl"] and genes.keys() & evidence_genes})
            matched = tuple(e["gene"] for e in x["evidence"])
            hits.append(RetrievalHit(i, x["label"], x["cl"], x["tissue"], x["tissue_id"], round(x["score"], 8), round(x["gene_score"], 8), round(x["coverage_factor"], 8), round(x["support_factor"], 8), round(x["cosine"], 8), round(x["weight"], 8), matched, tuple(pmids), tuple(x["evidence"]), len(matched), len(matched)/len(query) if query else 0.0))
        return RetrievalPacket(RETRIEVAL_VERSION, self.reference_sha256, tissue, tuple(selected), tuple(hits), "provided")

    def tissue_names(self):
        return tuple(sorted(self.tissue_ids))

    def exact_tissue_context(self, tissue: str) -> list[dict[str, object]]:
        tissue_id = self.tissue_ids.get(_norm(tissue))
        if tissue_id is None:
            return []
        return [{"tissue": tissue, "tissue_id": tissue_id, "cosine": 1.0}]

async def select_tissues(retriever: MarkerReferenceRetriever, tissue: str, cache_path: Path | None = None):
    """Select exact tissue plus embedding nearest tissues; cache vectors by tissue."""
    exact = retriever.tissue_ids.get(_norm(tissue))
    names = retriever.tissue_names()
    if exact:
        selected = [{"tissue": tissue, "tissue_id": exact, "cosine": 1.0}]
        if len(names) == 1: return selected
    else: selected = []
    cache = {}
    if cache_path and cache_path.exists():
        try: cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception: cache = {}
    missing = [x for x in ([tissue] + list(names)) if x not in cache]
    if missing:
        from openai import AsyncOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: raise EnvironmentError("OPENAI_API_KEY is required for tissue embeddings")
        base = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").strip() or None
        client = AsyncOpenAI(api_key=api_key, base_url=base)
        try:
            response = await client.embeddings.create(model=EMBEDDING_MODEL, input=missing)
            for name, item in zip(missing, response.data): cache[name] = item.embedding
        finally: await client.close()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True); tmp=cache_path.with_suffix(cache_path.suffix+'.tmp'); tmp.write_text(json.dumps(cache,separators=(',',':')),encoding='utf-8'); tmp.replace(cache_path)
    q = cache[tissue]; qnorm = math.sqrt(sum(x*x for x in q))
    ranked=[]
    for name in names:
        if exact and name == _norm(tissue): continue
        v=cache[name]; den=qnorm*math.sqrt(sum(x*x for x in v)); cos=sum(a*b for a,b in zip(q,v))/den if den else 0.0
        ranked.append((cos,name,retriever.tissue_ids[name]))
    ranked.sort(key=lambda x:(-x[0],x[2]))
    selected.extend({"tissue":n,"tissue_id":tid,"cosine":c} for c,n,tid in ranked[:max(0,3-len(selected))])
    return selected

def load_retriever(path: str): return MarkerReferenceRetriever.from_tsv(Path(path))
