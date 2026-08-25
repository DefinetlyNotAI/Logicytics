"""Evidence artifact registration with workspace-bound path validation."""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
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
        self._bytes_registered = 0
        self._artifacts: list[Artifact] = []

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        """Return artifacts registered by this worker in registration order."""
        return tuple(self._artifacts)

    def register_file(
            self,
            source: Path,
            *,
            media_type: str = "application/octet-stream",
            transformations: tuple[str, ...] = (),
    ) -> Artifact:
        if not isinstance(transformations, tuple) or any(
                not isinstance(step, str) or not step.strip() for step in transformations
        ):
            raise ArtifactError("artifact transformations must be a tuple of non-empty strings")
        source = source.resolve()
        if not source.is_file() or not _is_within(source, self._workspace):
            raise ArtifactError("artifacts must be regular files inside the collector workspace")
        size_bytes = source.stat().st_size
        if size_bytes > self._maximum_artifact_bytes:
            raise ArtifactError("collector artifact exceeds its declared maximum_artifact_bytes")
        if self._bytes_registered + size_bytes > self._maximum_output_bytes:
            raise ArtifactError("collector output exceeds its declared maximum_output_bytes")
        if len(self._artifacts) >= self._maximum_artifact_files:
            raise ArtifactError("collector artifact count exceeds its declared maximum_artifact_files")

        relative_source = source.relative_to(self._workspace)
        safe_collector_id = self._collector_id.replace(".", "_")
        destination = self._artifact_root / safe_collector_id / relative_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = destination.with_name(f"{destination.stem}-{uuid4().hex[:8]}{destination.suffix}")
        shutil.copy2(source, destination)
        artifact = Artifact(
            id=f"artifact.{uuid4().hex}",
            relative_path=destination.relative_to(self._artifact_root).as_posix(),
            sha256=sha256_file(destination),
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
