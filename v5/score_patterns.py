"""Small, deterministic summaries of cell-by-reference score matrices."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def inspect_score_patterns(scores: Mapping[str, Sequence[float]], *, high_threshold: float = 0.5) -> dict[str, Any]:
    if high_threshold < 0:
        raise ValueError("high_threshold must be non-negative")
    high_cells = {cell: [i for i, value in enumerate(values) if float(value) >= high_threshold]
                  for cell, values in scores.items()}
    mixed = [cell for cell, labels in high_cells.items() if len(labels) > 1]
    single = [cell for cell, labels in high_cells.items() if len(labels) == 1]
    return {"cell_count": len(scores), "high_threshold": high_threshold,
            "high_score_labels_by_cell": high_cells, "mixed_cells": mixed, "single_label_cells": single}
