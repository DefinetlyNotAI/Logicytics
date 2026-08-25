"""Evidence artifact registration with workspace-bound path validation."""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from logicytics.contracts import Artifact, ArtifactWriter
from logicytics.errors import ArtifactError


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large evidence does not fill memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WorkspaceArtifactWriter(ArtifactWriter):
    """Copies approved files from one collector workspace to the run artifact tree."""

    def __init__(
            self,
            collector_id: str,
            workspace: Path,
            artifact_root: Path,
            maximum_output_bytes: int,
            maximum_artifact_files: int,
            *,
            source_category: str | None = None,
            maximum_artifact_bytes: int | None = None,
            run_output_budget_bytes: int | None = None,
            cancellation_file: Path | None = None,
    ) -> None:
        self._collector_id = collector_id
        collector_parts = collector_id.split(".", 2)
        resolved_category = source_category if source_category is not None else (
            collector_parts[1] if len(collector_parts) > 1 else ""
        )
        if not isinstance(resolved_category, str) or not resolved_category.strip():
            raise ArtifactError("artifact source_category must be a non-empty string")
        self._source_category = resolved_category
        self._workspace = workspace.resolve()
        self._artifact_root = artifact_root.resolve()
        self._maximum_output_bytes = maximum_output_bytes
        self._maximum_artifact_bytes = (
            maximum_output_bytes if maximum_artifact_bytes is None else maximum_artifact_bytes
        )
        if not isinstance(self._maximum_artifact_bytes, int) or isinstance(
                self._maximum_artifact_bytes,
                bool,
        ) or not 1 <= self._maximum_artifact_bytes <= maximum_output_bytes:
            raise ArtifactError("maximum_artifact_bytes must be a positive integer within maximum_output_bytes")
        self._maximum_artifact_files = maximum_artifact_files
        self._run_output_budget_bytes = run_output_budget_bytes
        if run_output_budget_bytes is not None and (
                not isinstance(run_output_budget_bytes, int)
                or isinstance(run_output_budget_bytes, bool)
                or run_output_budget_bytes < 1
        ):
            raise ArtifactError("run_output_budget_bytes must be a positive integer")
        self._cancellation_file = cancellation_file
        self._registration_lock = RLock()
        self._bytes_registered = 0
        self._artifacts: list[Artifact] = []

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        """Return artifacts registered by this worker in registration order."""
        with self._registration_lock:
            return tuple(self._artifacts)

    def register_file(
            self,
            source: Path,
            *,
            media_type: str = "application/octet-stream",
            transformations: tuple[str, ...] = (),
    ) -> Artifact:
        """Serialize destination allocation, quota checks, publication, and catalog updates."""
        with self._registration_lock:
            return self._register_file(source, media_type=media_type, transformations=transformations)

    def _register_file(
            self,
            source: Path,
            *,
            media_type: str,
            transformations: tuple[str, ...],
    ) -> Artifact:
        if not isinstance(transformations, tuple) or any(
                not isinstance(step, str) or not step.strip() for step in transformations
        ):
            raise ArtifactError("artifact transformations must be a tuple of non-empty strings")
        self._check_cancellation()
        source = source.resolve()
        if not source.is_file() or not _is_within(source, self._workspace):
            raise ArtifactError("artifacts must be regular files inside the collector workspace")
        source_stat = source.stat()
        size_bytes = source_stat.st_size
        self._check_output_limits(size_bytes)
        if len(self._artifacts) >= self._maximum_artifact_files:
            raise ArtifactError("collector artifact count exceeds its declared maximum_artifact_files")

        relative_source = source.relative_to(self._workspace)
        safe_collector_id = self._collector_id.replace(".", "_")
        destination = self._artifact_root / safe_collector_id / relative_source
        if not _is_within(destination, self._artifact_root):
            raise ArtifactError("artifact destination must remain inside the run artifact store")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not _is_within(destination, self._artifact_root):
            raise ArtifactError("artifact destination must remain inside the run artifact store")
        if destination.exists():
            destination = destination.with_name(f"{destination.stem}-{uuid4().hex[:8]}{destination.suffix}")
        digest = self._copy_artifact(source, destination, size_bytes, source_stat.st_mtime_ns)
        artifact = Artifact(
            id=f"artifact.{uuid4().hex}",
            relative_path=destination.relative_to(self._artifact_root).as_posix(),
            sha256=digest,
            size_bytes=size_bytes,
            media_type=media_type,
            collector_id=self._collector_id,
            source_category=self._source_category,
            collected_at=datetime.now(timezone.utc).isoformat(),
            transformations=(*transformations, "copied into run artifact store"),
        )
        self._bytes_registered += size_bytes
        self._artifacts.append(artifact)
        return artifact

    def _check_cancellation(self) -> None:
        if self._cancellation_file is not None and self._cancellation_file.exists():
            raise ArtifactError("artifact registration was cancelled")

    def _check_output_limits(self, size_bytes: int) -> None:
        if size_bytes > self._maximum_artifact_bytes:
            raise ArtifactError("collector artifact exceeds its declared maximum_artifact_bytes")
        if self._bytes_registered + size_bytes > self._maximum_output_bytes:
            raise ArtifactError("collector output exceeds its declared maximum_output_bytes")
        if self._run_output_budget_bytes is not None and (
                self._bytes_registered + size_bytes > self._run_output_budget_bytes
        ):
            raise ArtifactError("run output exceeds configured maximum_run_output_bytes")

    def _copy_artifact(
            self,
            source: Path,
            destination: Path,
            expected_size: int,
            expected_modified_at: int,
    ) -> str:
        """Stream bounded evidence to an atomic destination while observing cancellation."""
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        digest = hashlib.sha256()
        copied_bytes = 0
        try:
            with source.open("rb") as source_stream, temporary.open("xb") as destination_stream:
                while block := source_stream.read(1024 * 1024):
                    self._check_cancellation()
                    copied_bytes += len(block)
                    self._check_output_limits(copied_bytes)
                    if copied_bytes > expected_size:
                        raise ArtifactError("artifact source changed during registration")
                    destination_stream.write(block)
                    digest.update(block)
            final_stat = source.stat()
            if (
                    copied_bytes != expected_size
                    or final_stat.st_size != expected_size
                    or final_stat.st_mtime_ns != expected_modified_at
            ):
                raise ArtifactError("artifact source changed during registration")
            shutil.copystat(source, temporary)
            self._check_cancellation()
            os.replace(temporary, destination)
            return digest.hexdigest()
        finally:
            if temporary.exists():
                temporary.unlink()
