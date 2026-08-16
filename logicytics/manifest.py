"""Atomic run-manifest persistence and normalized execution records."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from logicytics.contracts import Artifact, CollectorResult, RunStatus


def utc_now() -> str:
    """Return a stable UTC timestamp for manifests and records."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CollectorRecord:
    """Manifest-visible state for one planned collector."""

    id: str
    source: str
    status: str = "planned"
    started_at: str | None = None
    finished_at: str | None = None
    summary: str | None = None
    errors: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: Mapping[str, int | float | str] = field(default_factory=dict)
    duration_seconds: float | None = None
    event_count: int = 0
    last_progress_at: str | None = None

    def apply_result(self, result: CollectorResult, duration_seconds: float | None = None) -> None:
        """Copy a worker result into this serializable record."""
        self.status = result.status.value
        self.summary = result.summary
        self.errors = list(result.errors)
        self.artifacts = [artifact.to_dict() for artifact in result.artifacts]
        self.metrics = dict(result.metrics)
        self.duration_seconds = duration_seconds
        self.finished_at = utc_now()


@dataclass(slots=True)
class RunManifest:
    """The durable description of one run, including partial failures."""

    run_id: str
    requested_at: str
    status: RunStatus
    request: Mapping[str, Any]
    configuration: Mapping[str, Any]
    collectors: list[CollectorRecord]
    host: Mapping[str, str]
    finished_at: str | None = None
    package: Mapping[str, str] | None = None
    total_artifact_bytes: int = 0

    @classmethod
    def create(
        cls,
        run_id: str,
        request: Mapping[str, Any],
        configuration: Mapping[str, Any],
        collector_sources: list[tuple[str, Path]],
    ) -> "RunManifest":
        """Create the initial planned manifest before collection begins."""
        return cls(
            run_id=run_id,
            requested_at=utc_now(),
            status=RunStatus.PLANNED,
            request=request,
            configuration=configuration,
            collectors=[CollectorRecord(id=collector_id, source=str(source)) for collector_id, source in collector_sources],
            host={"platform": sys_platform(), "hostname": platform.node(), "python": platform.python_version()},
        )

    def finalize_status(self) -> None:
        """Derive a truthful run status from all collector outcomes."""
        statuses = {record.status for record in self.collectors}
        if "cancelled" in statuses:
            self.status = RunStatus.CANCELLED
        elif "failed" in statuses and len(statuses) == 1:
            self.status = RunStatus.FAILED
        elif "failed" in statuses or "partial" in statuses or "skipped" in statuses:
            self.status = RunStatus.PARTIAL
        else:
            self.status = RunStatus.SUCCEEDED
        self.finished_at = utc_now()
        self.total_artifact_bytes = sum(artifact.size_bytes for artifact in self.artifact_list())

    def artifact_list(self) -> tuple[Artifact, ...]:
        """Return all registered artifacts reconstructed from the manifest."""
        return tuple(
            Artifact(**artifact)
            for record in self.collectors
            for artifact in record.artifacts
        )

    def to_dict(self) -> dict[str, Any]:
        """Produce JSON-safe manifest data."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


def sys_platform() -> str:
    """Expose the runtime platform without importing platform in consumers."""
    return platform.system().lower()


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Write a manifest atomically so partial writes never resemble completed runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
