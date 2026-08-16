"""Safe local Sysinternals archive discovery and extraction for v4."""

from __future__ import annotations

import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SysinternalsState:
    """Archive, ignore-marker, and extraction state for the local project."""

    status: str
    archive: Path
    extraction_directory: Path

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe state summary."""
        data = asdict(self)
        data["archive"] = str(self.archive)
        data["extraction_directory"] = str(self.extraction_directory)
        return data


def ensure_sysinternals(project_root: Path) -> SysinternalsState:
    """Safely extract a bundled archive unless an explicit local marker opts out."""
    archive = project_root / "SysinternalsSuite.zip"
    extraction_directory = project_root / "tools" / "sysinternals"
    if (project_root / ".ignore-sysinternals").is_file():
        return SysinternalsState("ignored", archive, extraction_directory)
    if extraction_directory.is_dir():
        return SysinternalsState("extracted", archive, extraction_directory)
    if not archive.is_file():
        return SysinternalsState("missing", archive, extraction_directory)
    extraction_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (extraction_directory / member.filename).resolve()
            try:
                destination.relative_to(extraction_directory.resolve())
            except ValueError as error:
                raise ValueError("Sysinternals archive contains an unsafe path") from error
        bundle.extractall(extraction_directory)
    return SysinternalsState("extracted", archive, extraction_directory)
