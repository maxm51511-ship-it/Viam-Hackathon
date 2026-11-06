"""Append-only JSON-lines event log (one record per state change / command)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "value"):
        return o.value
    return str(o)


class EventLog:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, **data) -> dict:
        rec = {"t": time.time(), "kind": kind, **data}
        self.records.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=_default) + "\n")
        return rec
