"""Atomic run-manifest persistence and normalized execution records."""

from __future__ import annotations

import getpass
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from logicytics.contracts import CONTRACT_VERSION, Artifact, CollectorResult, RunStatus
from logicytics.module.redaction import redact_mapping, redact_text
from logicytics.platform_adapters import windows_api_adapter

MANIFEST_SCHEMA_VERSION = 1


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
    failure: dict[str, str | bool] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: Mapping[str, int | float | str] = field(default_factory=dict)
    progress: dict[str, int | float] = field(
        default_factory=lambda: {
            "files_scanned": 0,
            "files_copied": 0,
            "bytes_written": 0,
            "packets_observed": 0,
            "events_processed": 0,
            "elapsed_seconds": 0.0,
        }
    )
    duration_seconds: float | None = None
    event_count: int = 0
    last_progress_at: str | None = None
    heartbeat_at: str | None = None
    isolation_mode: str = "process"
    worker_pid: int | None = None
    worker_exit_code: int | None = None
    termination_reason: str | None = None
    peak_memory_bytes: int = 0
    attempt_count: int = 0
    retry_history: list[dict[str, Any]] = field(default_factory=list)

    def apply_result(self, result: CollectorResult, duration_seconds: float | None = None) -> None:
        """Copy a worker result into this serializable record."""
        self.status = result.status.value
        self.summary = redact_text(result.summary)
        self.errors = [redact_text(error) for error in result.errors]
        self.artifacts = [artifact.to_dict() for artifact in result.artifacts]
        self.metrics = redact_mapping(result.metrics)
        self.failure = self._failure_details() if self.status == "failed" else None
        self.duration_seconds = duration_seconds
        self.finished_at = utc_now()

    def _failure_details(self) -> dict[str, str | bool]:
        """Convert collector-specific failures into actionable, redacted guidance."""
        platform_error = self.errors[0] if self.errors else self.summary or "unknown collector failure"
        diagnostic = f"{self.summary or ''} {platform_error}".casefold()
        operation = "collect"
        remediation = "Inspect the collector traceback and correct the reported platform failure."
        retry_safe = not self.artifacts
        if "timeout" in diagnostic:
            operation = "collect"
            remediation = "Reduce the collection scope or increase the collector timeout before retrying."
        elif "memory" in diagnostic or "working set" in diagnostic:
            operation = "collect"
            remediation = "Reduce the collection scope or increase its declared memory limit."
        elif "cleanup" in diagnostic or "finaliz" in diagnostic:
            operation = "cleanup"
            remediation = "Inspect the private collector workspace and repair its cleanup prerequisites."
            retry_safe = False
        elif "permission" in diagnostic or "capability" in diagnostic or "access is denied" in diagnostic:
            operation = "access"
            remediation = "Declare the capability in CollectorMetadata.capabilities and remove any active block before retrying."
            retry_safe = False
        elif "artifact" in diagnostic or "output limit" in diagnostic:
            operation = "artifact_registration"
            remediation = "Reduce evidence size/count or adjust the collector's declared artifact limits."
            retry_safe = False
        elif "validat" in diagnostic:
            operation = "validate"
            remediation = "Correct the collector configuration and platform prerequisites before retrying."
            retry_safe = False
        return {
            "collector_id": self.id,
            "operation": operation,
            "platform_error": platform_error,
            "remediation": remediation,
            "retry_safe": retry_safe,
        }


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
    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION
    resolved_plan: tuple[str, ...] = ()
    plan_fingerprint: str | None = None
    engine_version: str = CONTRACT_VERSION
    action: str = "run"
    parent_run_id: str | None = None
    finished_at: str | None = None
    cancellation_requested: bool = False
    errors: list[dict[str, str]] = field(default_factory=list)
    skipped_collectors: list[str] = field(default_factory=list)
    package: Mapping[str, str] | None = None
    package_layout_version: str = "1.0"
    total_artifact_bytes: int = 0

    @classmethod
    def create(
            cls,
            run_id: str,
            request: Mapping[str, Any],
            configuration: Mapping[str, Any],
            collector_sources: list[tuple[str, Path]],
            *,
            parent_run_id: str | None = None,
            plan_fingerprint: str | None = None,
    ) -> "RunManifest":
        """Create the initial planned manifest before collection begins."""
        return cls(
            run_id=run_id,
            requested_at=utc_now(),
            status=RunStatus.PLANNED,
            request=request,
            configuration=configuration,
            collectors=[CollectorRecord(id=collector_id, source=str(source)) for collector_id, source in
                        collector_sources],
            resolved_plan=tuple(collector_id for collector_id, _ in collector_sources),
            plan_fingerprint=plan_fingerprint,
            action="rerun" if parent_run_id is not None else "run",
            parent_run_id=parent_run_id,
            host={
                "platform": sys_platform(),
                "hostname": platform.node(),
                "python": platform.python_version(),
                "user": getpass.getuser(),
                "is_administrator": _privilege_label(),
            },
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
        self.cancellation_requested = "cancelled" in statuses
        self.skipped_collectors = [record.id for record in self.collectors if record.status == "skipped"]
        self.errors = [
            {"collector_id": record.id, "status": record.status, "message": error}
            for record in self.collectors
            for error in record.errors
        ]
        self.finished_at = utc_now()
        self.total_artifact_bytes = sum(artifact.size_bytes for artifact in self.artifact_list())

    def artifact_list(self) -> tuple[Artifact, ...]:
        """Return all registered artifacts reconstructed from the manifest."""
        return tuple(Artifact.from_dict({key: value for key, value in item.items() if key != "producer_status"})
                     for item in self.artifact_catalog())

    def artifact_catalog(self) -> tuple[dict[str, Any], ...]:
        """Build the globally unique, ownership-validated run-wide evidence catalog."""
        catalog: list[dict[str, Any]] = []
        artifact_ids: set[str] = set()
        artifact_paths: set[str] = set()
        for record in self.collectors:
            for item in record.artifacts:
                artifact = Artifact.from_dict(item)
                if artifact.collector_id != record.id:
                    raise ValueError(f"manifest artifact collector ownership is invalid: {artifact.relative_path}")
                relative = PurePosixPath(artifact.relative_path)
                if (
                        "\\" in artifact.relative_path
                        or relative.is_absolute()
                        or ".." in relative.parts
                        or relative.as_posix() != artifact.relative_path
                        or len(relative.parts) < 2
                        or relative.parts[0] != record.id.replace(".", "_")
                ):
                    raise ValueError(f"manifest artifact escapes its collector-owned store: {artifact.relative_path}")
                if artifact.name != relative.name:
                    raise ValueError(f"manifest artifact name does not match its registered path: {artifact.name}")
                if artifact.id in artifact_ids:
                    raise ValueError(f"manifest contains duplicate artifact id: {artifact.id}")
                if artifact.relative_path in artifact_paths:
                    raise ValueError(f"manifest contains duplicate artifact path: {artifact.relative_path}")
                artifact_ids.add(artifact.id)
                artifact_paths.add(artifact.relative_path)
                catalog.append({**artifact.to_dict(), "producer_status": record.status})
        return tuple(catalog)

    def to_dict(self) -> dict[str, Any]:
        """Produce JSON-safe manifest data."""
        if (
                not isinstance(self.manifest_schema_version, int)
                or isinstance(self.manifest_schema_version, bool)
                or self.manifest_schema_version != MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported run manifest schema_version {self.manifest_schema_version!r}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        data = asdict(self)
        data["status"] = self.status.value
        data["artifact_catalog"] = self.artifact_catalog()
        data["package_sections"] = {
            "raw_evidence": "evidence/raw/",
            "derived_reports": "evidence/derived/",
            "reports": "reports/",
            "logs": "logs/",
            "hashes": "hashes/",
            "metadata": "metadata/",
        }
        return data


def sys_platform() -> str:
    """Expose the runtime platform without importing platform in consumers."""
    return platform.system().lower()


def _privilege_label() -> str:
    """Return the local elevation state without starting external tools."""
    state = windows_api_adapter.is_administrator()
    if state is None:
        return "unknown"
    return "true" if state else "false"


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Write a manifest atomically so partial writes never resemble completed runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
