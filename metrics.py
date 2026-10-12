"""Write compact, label-free CellClick run metrics for offline comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def append_metrics(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append one task's metrics without raw model responses or truth labels."""
    rows = list(records)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
