"""Small, side-effect-free public application API for v4 collection runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from logicytics.configuration import AppConfig, load_config
from logicytics.contracts import Artifact, CollectorStatus, RunRequest, RunStatus
from logicytics.discovery import preflight
from logicytics.errors import ArtifactError, PlanError
from logicytics.planner import RunPlan, build_plan
from logicytics.runtime import RunOutcome, RunSupervisor

_RUN_ID = re.compile(r"run-[0-9a-f]{32}")
_ARTIFACT_ID = re.compile(r"artifact\.[0-9a-f]{32}")
_COLLECTOR_ID = re.compile(r"(?:core|plugin)\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?")
_MAXIMUM_ARTIFACT_READ_BYTES = 64 * 1024 * 1024
_DEFAULT_ARTIFACT_READ_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CollectorSnapshot:
    """Immutable public status for one collector recorded in a persisted run."""

    collector_id: str
    status: str


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Immutable, manifest-validated view of one current or completed run."""

    run_id: str
    status: RunStatus
    run_directory: Path
    manifest_path: Path
    collectors: tuple[CollectorSnapshot, ...]
    artifacts: tuple[Artifact, ...]
    finished_at: str | None


def _unique_manifest_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous persisted manifest keys instead of replacing prior values."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate run manifest key {key!r}")
        result[key] = value
    return result


def load_configuration(project_root: Path | str, config_path: Path | str | None = None) -> AppConfig:
    """Load strictly validated configuration without creating collection output."""
    root = Path(project_root).resolve()
    selected = None if config_path is None else Path(config_path)
    if selected is not None and not selected.is_absolute():
        selected = root / selected
    return load_config(root, selected)


def _configuration(
        project_root: Path | str,
        configuration: AppConfig | None,
        config_path: Path | str | None,
) -> tuple[Path, AppConfig]:
    """Resolve one unambiguous, typed configuration source."""
    root = Path(project_root).resolve()
    if configuration is not None and config_path is not None:
        raise PlanError("provide either configuration or config_path, not both")
    if configuration is not None and not isinstance(configuration, AppConfig):
        raise PlanError("configuration must be a validated AppConfig instance")
    return root, configuration if configuration is not None else load_configuration(root, config_path)


def plan_run(
        project_root: Path | str,
        request: RunRequest,
        *,
        configuration: AppConfig | None = None,
        config_path: Path | str | None = None,
) -> RunPlan:
    """Preflight and resolve a strict run without starting collectors or writing output."""
    root, settings = _configuration(project_root, configuration, config_path)
    if not isinstance(request, RunRequest):
        raise PlanError("request must be an immutable RunRequest instance")
    if request.max_workers > settings.runtime.maximum_workers:
        raise PlanError("requested workers exceed configured maximum_workers")
    return build_plan(preflight(root), request)


def run_collection(
        project_root: Path | str,
        request: RunRequest,
        *,
        configuration: AppConfig | None = None,
        config_path: Path | str | None = None,
) -> RunOutcome:
    """Execute only a strictly validated, authorized, independently isolated run."""
    root, settings = _configuration(project_root, configuration, config_path)
    plan = plan_run(root, request, configuration=settings)
    return RunSupervisor(root, settings).run(plan)


def _artifact(raw: object, collector_id: str) -> Artifact:
    """Reconstruct one strict artifact while enforcing collector-owned paths."""
    if not isinstance(raw, dict):
        raise PlanError("run manifest contains a malformed registered artifact")
    values = dict(raw)
    transformations = values.get("transformations")
    if not isinstance(transformations, list):
        raise PlanError("run manifest artifact transformations must be an array")
    values["transformations"] = tuple(transformations)
    try:
        artifact = Artifact(**values)
    except (TypeError, ValueError) as error:
        raise PlanError(f"run manifest contains an invalid registered artifact: {error}") from error
    relative = PurePosixPath(artifact.relative_path)
    owner = collector_id.replace(".", "_")
    if (
            artifact.collector_id != collector_id
            or "\\" in artifact.relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != artifact.relative_path
            or len(relative.parts) < 2
            or relative.parts[0] != owner
            or artifact.name != relative.name
    ):
        raise PlanError("run manifest artifact violates collector ownership or path boundaries")
    return artifact


def _manifest_timestamp(value: object, field: str, *, optional: bool = False) -> str | None:
    """Require timezone-aware persisted timestamps, allowing null only when declared."""
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise PlanError(f"run manifest {field} must be a timezone-aware timestamp")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise PlanError(f"run manifest {field} must be a timezone-aware timestamp") from error
    if timestamp.tzinfo is None:
        raise PlanError(f"run manifest {field} must be a timezone-aware timestamp")
    return value


def query_run(
        project_root: Path | str,
        run_id: str,
        *,
        configuration: AppConfig | None = None,
        config_path: Path | str | None = None,
) -> RunSnapshot:
    """Read a run-owned manifest and return immutable, ownership-validated status."""
    _, settings = _configuration(project_root, configuration, config_path)
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise PlanError("run_id must be a canonical run identifier")
    output_root = settings.runtime.output_root.resolve()
    run_directory = output_root / run_id
    manifest_path = run_directory / "manifest.json"
    try:
        if run_directory.resolve(strict=True) != run_directory:
            raise PlanError("run directory escapes its configured output root")
        if manifest_path.resolve(strict=True) != manifest_path or not manifest_path.is_file():
            raise PlanError("run manifest escapes its run-owned directory")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_manifest_object)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise PlanError(f"unable to read run manifest for {run_id}: {error}") from error
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise PlanError("run manifest identity does not match its owned directory")
    _manifest_timestamp(payload.get("requested_at"), "requested_at")
    if not isinstance(payload.get("status"), str):
        raise PlanError("run manifest contains an unsupported run status")
    try:
        status = RunStatus(payload.get("status"))
    except ValueError as error:
        raise PlanError("run manifest contains an unsupported run status") from error
    records = payload.get("collectors")
    catalog = payload.get("artifact_catalog")
    if not isinstance(records, list) or not isinstance(catalog, list):
        raise PlanError("run manifest must contain collector and artifact catalogs")

    collectors: list[CollectorSnapshot] = []
    artifacts: list[Artifact] = []
    collector_ids: set[str] = set()
    artifact_ids: set[str] = set()
    expected_catalog: list[dict[str, Any]] = []
    valid_collector_statuses = {"planned", "running", *(item.value for item in CollectorStatus)}
    for record in records:
        if not isinstance(record, dict):
            raise PlanError("run manifest contains a malformed collector record")
        collector_id = record.get("id")
        collector_status = record.get("status")
        if (
                not isinstance(collector_id, str)
                or _COLLECTOR_ID.fullmatch(collector_id) is None
                or collector_id in collector_ids
                or not isinstance(collector_status, str)
                or collector_status not in valid_collector_statuses
                or not isinstance(record.get("artifacts"), list)
        ):
            raise PlanError("run manifest contains an invalid or duplicate collector record")
        collector_ids.add(collector_id)
        collectors.append(CollectorSnapshot(collector_id, collector_status))
        for raw in record["artifacts"]:
            artifact = _artifact(raw, collector_id)
            if artifact.id in artifact_ids:
                raise PlanError("run manifest contains duplicate registered artifact IDs")
            artifact_ids.add(artifact.id)
            artifacts.append(artifact)
            expected_catalog.append({**raw, "producer_status": collector_status})
    if payload.get("resolved_plan") != [item.collector_id for item in collectors]:
        raise PlanError("run manifest collector records do not match the resolved plan")
    if catalog != expected_catalog:
        raise PlanError("run manifest artifact catalog does not match collector-owned evidence")
    if status not in {RunStatus.PLANNED, RunStatus.RUNNING} and payload.get("total_artifact_bytes") != sum(
            artifact.size_bytes for artifact in artifacts
    ):
        raise PlanError("run manifest total_artifact_bytes does not match registered evidence")
    finished_at = _manifest_timestamp(payload.get("finished_at"), "finished_at", optional=True)
    if status not in {RunStatus.PLANNED, RunStatus.RUNNING} and finished_at is None:
        raise PlanError("finalized run manifest must contain finished_at")
    return RunSnapshot(
        run_id=run_id,
        status=status,
        run_directory=run_directory,
        manifest_path=manifest_path,
        collectors=tuple(collectors),
        artifacts=tuple(artifacts),
        finished_at=finished_at,
    )


def read_artifact(
        project_root: Path | str,
        run_id: str,
        artifact_id: str,
        *,
        maximum_bytes: int = _DEFAULT_ARTIFACT_READ_BYTES,
        configuration: AppConfig | None = None,
        config_path: Path | str | None = None,
) -> bytes:
    """Read bounded, registered evidence only after ownership and SHA-256 verification."""
    if not isinstance(artifact_id, str) or _ARTIFACT_ID.fullmatch(artifact_id) is None:
        raise ArtifactError("artifact_id must be a canonical registered artifact identifier")
    if (
            not isinstance(maximum_bytes, int)
            or isinstance(maximum_bytes, bool)
            or not 1 <= maximum_bytes <= _MAXIMUM_ARTIFACT_READ_BYTES
    ):
        raise ArtifactError("maximum_bytes must be an integer from 1 to 67108864")
    snapshot = query_run(project_root, run_id, configuration=configuration, config_path=config_path)
    artifact = next((item for item in snapshot.artifacts if item.id == artifact_id), None)
    if artifact is None:
        raise ArtifactError("artifact is not registered in the selected run manifest")
    if artifact.size_bytes > maximum_bytes:
        raise ArtifactError("registered artifact exceeds the requested bounded read limit")
    root = snapshot.run_directory / "artifacts"
    source = root.joinpath(*PurePosixPath(artifact.relative_path).parts)
    owner = root / artifact.collector_id.replace(".", "_")
    try:
        if root.resolve(strict=True) != root or owner.resolve(strict=True) != owner:
            raise ValueError("artifact ownership roots must not be redirected")
        resolved = source.resolve(strict=True)
        resolved.relative_to(owner)
        if not source.is_file():
            raise ArtifactError("registered artifact must be a regular run-owned file")
        with source.open("rb") as stream:
            contents = stream.read(maximum_bytes + 1)
    except (OSError, ValueError) as error:
        raise ArtifactError("registered artifact escapes its collector-owned store or cannot be read") from error
    if (
            len(contents) > maximum_bytes
            or len(contents) != artifact.size_bytes
            or hashlib.sha256(contents).hexdigest() != artifact.sha256
    ):
        raise ArtifactError("registered artifact failed manifest size or SHA-256 verification")
    return contents
