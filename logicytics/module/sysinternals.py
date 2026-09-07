"""Safe local Sysinternals archive discovery and extraction for v4."""

from __future__ import annotations

import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlopen

from logicytics.module.configuration import MaintenanceSettings


@dataclass(frozen=True, slots=True)
class SysinternalsState:
    """Configured download, archive, and extraction state for the local project."""

    status: str
    archive: Path
    extraction_directory: Path

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe state summary."""
        data = asdict(self)
        data["archive"] = str(self.archive)
        data["extraction_directory"] = str(self.extraction_directory)
        return data


def ensure_sysinternals(
        project_root: Path,
        settings: MaintenanceSettings | None = None,
) -> SysinternalsState:
    """Honor YAML opt-out or securely download, validate, and extract Sysinternals."""
    settings = settings or MaintenanceSettings()
    archive = project_root / "tools" / "SysinternalsSuite.zip"
    extraction_directory = project_root / "tools" / "sysinternals"
    if not settings.sysinternals_enabled:
        return SysinternalsState("disabled", archive, extraction_directory)
    if extraction_directory.is_dir():
        return SysinternalsState("extracted", archive, extraction_directory)
    if not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(settings.sysinternals_download_url, timeout=30) as response, NamedTemporaryFile(
                    mode="wb", dir=archive.parent, delete=False
            ) as temporary:
                temporary.write(response.read())
                downloaded = Path(temporary.name)
        except (OSError, URLError) as error:
            return SysinternalsState(f"download_failed: {error}", archive, extraction_directory)
        try:
            if not zipfile.is_zipfile(downloaded):
                return SysinternalsState("download_failed: archive is not a ZIP file", archive, extraction_directory)
            downloaded.replace(archive)
        finally:
            downloaded.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination = (extraction_directory / member.filename).resolve()
                try:
                    destination.relative_to(extraction_directory.resolve())
                except ValueError as error:
                    raise ValueError("Sysinternals archive contains an unsafe path") from error
            extraction_directory.mkdir(parents=True, exist_ok=True)
            bundle.extractall(extraction_directory)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        return SysinternalsState(f"extract_failed: {error}", archive, extraction_directory)
    return SysinternalsState("extracted", archive, extraction_directory)
