"""Run-scoped structured logging without global mutable logging configuration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from logicytics.contracts import EventLogger


class FileEventLogger(EventLogger):
    """Append JSONL events to a single run- or collector-owned log file."""

    def __init__(self, path: Path, *, run_id: str, collector_id: str | None = None) -> None:
        self.path = path
        self.run_id = run_id
        self.collector_id = collector_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, level: str, message: str, **fields: int | float | str) -> None:
        """Write a timestamped, structured event without relying on global handlers."""
        payload: dict[str, object] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "level": level.lower(),
            "message": message,
            "run_id": self.run_id,
        }
        if self.collector_id is not None:
            payload["collector_id"] = self.collector_id
        if fields:
            payload["fields"] = fields
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
