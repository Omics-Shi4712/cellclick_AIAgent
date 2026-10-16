"""Dataset-local resources and metadata-only execution telemetry.

Resources never enter graph state or model messages. Context variables isolate
worker bindings while allowing the immutable dataset and bounded cache to be shared.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

CURRENT = contextvars.ContextVar("cellclick_v5_runtime", default=None)


class BoundedCache:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.size = 0
        self.values = OrderedDict()
        self.lock = threading.RLock()
        self.pending = {}
        self.hits = self.misses = 0

    @staticmethod
    def weight(value):
        if hasattr(value, "nbytes"):
            return sys.getsizeof(value) + value.nbytes
        if isinstance(value, dict):
            return sys.getsizeof(value) + sum(BoundedCache.weight(k) + BoundedCache.weight(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return sys.getsizeof(value) + sum(map(BoundedCache.weight, value))
        return sys.getsizeof(value)

    def get(self, key, compute):
        while True:
            with self.lock:
                if key in self.values:
                    self.hits += 1
                    value, size = self.values.pop(key)
                    self.values[key] = (value, size)
                    emit("cache", cache_hit=True)
                    return value
                event = self.pending.get(key)
                if event is None:
                    self.misses += 1
                    event = self.pending[key] = threading.Event()
                    break
            event.wait()
        try:
            value = compute()
            size = self.weight(value) + self.weight(key)
            with self.lock:
                if size <= self.max_bytes:
                    while self.values and self.size + size > self.max_bytes:
                        _, (_, removed) = self.values.popitem(last=False)
                        self.size -= removed
                    self.values[key] = (value, size)
                    self.size += size
            emit("cache", cache_hit=False)
            return value
        finally:
            with self.lock:
                self.pending.pop(key).set()


class DatasetResources:
    def __init__(self, expression_file, manifest_file, *, cache_max_mb=512):
        import anndata as ad
        import numpy as np
        from scipy import sparse

        started = time.perf_counter()
        self.path = Path(expression_file).resolve()
        self.manifest_path = Path(manifest_file).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (self.manifest.get("schema_version") != "cellclick.expression.2"
                or self.manifest.get("h5ad_path") != str(self.path)
                or self.manifest.get("groupby") != "cellclick_group_id"):
            raise ValueError("invalid expression resource binding")
        data = ad.read_h5ad(self.path)
        layer = self.manifest.get("statistics_layer")
        if layer != "normalization+log1p" or layer not in data.layers:
            raise ValueError("expression export lacks required V5 statistics")
        matrix = data.layers[layer]
        self.matrix = matrix.tocsc() if sparse.issparse(matrix) else np.asarray(matrix)
        self.labels = data.obs["cellclick_group_id"].astype(str).to_numpy()
        self.names = data.var_names.copy()
        self.names.values.flags.writeable = False
        self.index = {str(g): i for i, g in enumerate(self.names)}
        self.casefold_index = {}
        for gene in self.names:
            self.casefold_index.setdefault(str(gene).casefold(), []).append(str(gene))
        self.masks = {g: self.labels == g for g in sorted(set(self.labels))}
        self.cell_counts = {g: int(mask.sum()) for g, mask in self.masks.items()}
        for mask in self.masks.values():
            mask.flags.writeable = False
        arrays = (self.matrix.data, self.matrix.indices, self.matrix.indptr) if sparse.issparse(self.matrix) else (self.matrix,)
        for array in arrays:
            array.flags.writeable = False
        self.labels.flags.writeable = False
        from types import SimpleNamespace
        import pandas as pd
        self.view = SimpleNamespace(layers={layer: self.matrix},
                                    obs=pd.DataFrame({"cellclick_group_id": self.labels}), var_names=self.names)
        self.cache = BoundedCache(int(cache_max_mb * 1024 * 1024))
        self.jsons = {str(self.manifest_path): self.manifest}
        self.retrievers = {}
        self.lock = threading.RLock()
        self.load_seconds = time.perf_counter() - started
        # Content fingerprints are computed once per dataset, never per cache lookup.
        digest = hashlib.sha256()
        self.hashes = {}
        for path in (self.path, self.manifest_path):
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    file_digest.update(chunk)
            self.hashes[str(path)] = file_digest.hexdigest()
        self.fingerprint = digest.hexdigest()
        self.load_seconds = time.perf_counter() - started
        del data

    def json(self, path):
        key = str(Path(path).resolve())
        with self.lock:
            if key not in self.jsons:
                self.jsons[key] = json.loads(Path(key).read_text(encoding="utf-8"))
            return self.jsons[key]

    def retriever(self, path):
        from cellclick.reference import MarkerReferenceRetriever
        key = str(Path(path).resolve())
        with self.lock:
            if key not in self.retrievers:
                self.retrievers[key] = MarkerReferenceRetriever.from_tsv(Path(path))
            return self.retrievers[key]


class Runtime:
    def __init__(self, resources, sink, cluster_id, mapping_path=None, total_timeout=610,
                 connect_timeout=10, read_timeout=600):
        self.resources, self.sink, self.cluster_id = resources, sink, cluster_id
        self.mapping_path, self.total_timeout = mapping_path, total_timeout
        self.connect_timeout, self.read_timeout = connect_timeout, read_timeout

    @contextmanager
    def activate(self):
        token = CURRENT.set(self)
        try:
            yield
        finally:
            CURRENT.reset(token)


class Telemetry:
    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()

    def write(self, record):
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def emit(event, **fields):
    runtime = CURRENT.get()
    if runtime:
        runtime.sink.write({"event": event, "source_row_id": runtime.cluster_id, **fields})


def read_json(path):
    runtime = CURRENT.get()
    started = time.perf_counter()
    try:
        return runtime.resources.json(path) if runtime else json.loads(Path(path).read_text(encoding="utf-8"))
    finally:
        emit("data_read", kind="json", duration_seconds=time.perf_counter()-started)


def read_expression_manifest(path):
    runtime = CURRENT.get()
    if runtime and Path(path).resolve() != runtime.resources.manifest_path:
        raise ValueError("expression manifest resource binding mismatch")
    return read_json(path)


def load_expression(path):
    runtime = CURRENT.get()
    if runtime:
        if Path(path).resolve() != runtime.resources.path:
            raise ValueError("expression resource binding mismatch")
        emit("data_read", kind="expression", cache_hit=True, duration_seconds=0)
        return runtime.resources.view
    import anndata as ad
    return ad.read_h5ad(path)


def gene_vector(matrix, position, gene):
    import numpy as np
    from scipy import sparse
    def compute():
        values = matrix[:, position]
        values = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
        # Preserve the export dtype for legacy six-decimal expression summaries;
        # cosine and reference scoring explicitly accumulate in float64.
        values = np.asarray(values)
        values.flags.writeable = False
        return values
    runtime = CURRENT.get()
    return runtime.resources.cache.get((runtime.resources.fingerprint, "vector-v1", gene), compute) if runtime else compute()


def cached_statistics(function):
    """Reuse numerical per-gene results, never observations or model decisions."""
    from functools import wraps
    import copy

    @wraps(function)
    def wrapped(*, expression_file, manifest_file, source_row_id, genes):
        runtime = CURRENT.get()
        if not runtime:
            return function(expression_file=expression_file, manifest_file=manifest_file,
                            source_row_id=source_row_id, genes=genes)
        requested = list(dict.fromkeys(str(g).strip() for g in genes if str(g).strip()))
        if not requested or len(requested) > 200:
            raise ValueError("query_expression requires between 1 and 200 unique genes")
        r = runtime.resources
        if Path(expression_file).resolve() != r.path or Path(manifest_file).resolve() != r.manifest_path:
            raise ValueError("expression cache binding mismatch")
        results = []
        for gene in requested:
            key = (r.fingerprint, "normalization+log1p", source_row_id, gene, "statistics-v1")
            results.append(r.cache.get(key, lambda: function(expression_file=expression_file,
                manifest_file=manifest_file, source_row_id=source_row_id, genes=[gene])))
        result = copy.deepcopy(results[0])
        result["genes"] = [copy.deepcopy(item["genes"][0]) for item in results]
        return result
    return wrapped
