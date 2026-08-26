"""Find and copy bounded sensitive-named files after explicit approval."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, EvidenceKind, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus

KEYWORDS = ("password", "secret", "code", "login", "api", "key", "token", "auth", "credential", "private",
            "certificate", "ssh", "pgp", "wallet")
EXTENSIONS = {".txt", ".csv", ".json", ".xml", ".yml", ".yaml", ".ini", ".cfg", ".conf", ".log", ".pdf", ".doc",
              ".docx", ".xls", ".xlsx", ".zip", ".db", ".sqlite", ".pem", ".key", ".ppk"}
MAX_FILE_BYTES = 10 * 1024 * 1024


def _copy(source: Path, destination: Path) -> Path | None:
    """Copy one regular source file into the private workspace, preserving metadata."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    except OSError:
        return None
    return destination


class SensitiveFileInventoryCollector(CoreCollector):
    """Search a bounded system-drive subset for sensitive-named supported files."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the explicit-consent sensitive-file inventory artifact contract."""
        return CollectorMetadata(
            id="core.filesystem.sensitive_file_inventory", name="Sensitive file inventory", version="4.0.0",
            specialty=Specialty.FILESYSTEM,
            description="Finds and copies bounded supported files with sensitive-data keywords in their names.",
            author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.FILESYSTEM_READ, Capability.SENSITIVE_FILES),
            sensitive_data_categories=("credentials", "private_keys", "personal_documents"), default_profiles=("deep",),
            timeout_seconds=300, maximum_output_bytes=256 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Validate bounded search settings before walking the requested root."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        try:
            if int(context.settings.get("max_directories", 5_000)) < 1 or int(
                    context.settings.get("max_matches", 500)) < 1:
                return ValidationResult(False, reasons=("max_directories and max_matches must be positive",))
        except (TypeError, ValueError):
            return ValidationResult(False, reasons=("max_directories and max_matches must be integers",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Find matches and copy them concurrently into a dedicated evidence directory."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before sensitive-file inventory")
        root = Path(str(context.settings.get("root", os.environ.get("SystemDrive", "C:") + "\\")))
        max_directories = int(context.settings.get("max_directories", 5_000))
        max_matches = int(context.settings.get("max_matches", 500))
        matches: list[Path] = []
        scanned_directories = 0
        context.report_progress("sensitive_file_inventory_started", root=str(root))
        for directory, directories, filenames in os.walk(root, onerror=lambda _: None):
            if context.is_cancelled:
                return CollectorResult(CollectorStatus.CANCELLED, "cancelled during sensitive-file inventory")
            scanned_directories += 1
            if scanned_directories > max_directories or len(matches) >= max_matches:
                directories.clear()
                break
            for filename in filenames:
                candidate = Path(directory) / filename
                name = filename.casefold()
                if candidate.suffix.casefold() not in EXTENSIONS or not any(keyword in name for keyword in KEYWORDS):
                    continue
                try:
                    if candidate.is_symlink() or candidate.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                matches.append(candidate)
                if len(matches) >= max_matches:
                    break
        destination_root = context.workspace / "sensitive_file_inventory"
        with ThreadPoolExecutor(max_workers=4) as executor:
            copied = [item for item in executor.map(lambda source: _copy(source,
                                                                         destination_root / source.drive.replace(":",
                                                                                                                 "") / source.relative_to(
                                                                             root)), matches) if item is not None]
        if not copied:
            return CollectorResult(CollectorStatus.SKIPPED,
                                   "no sensitive-named supported files met the bounded inventory policy")
        artifacts = tuple(
            context.artifacts.register_file(path, evidence_kind=EvidenceKind.RAW) for path in copied
        )
        context.report_progress("sensitive_file_inventory_finished", scanned_directories=scanned_directories,
                                copied_files=len(artifacts), bytes_written=sum(item.size_bytes for item in artifacts))
        return CollectorResult.succeeded("sensitive file inventory collected", artifacts)

    def cleanup(self, context: CollectorContext) -> None:
        """Leave copied evidence removal to the isolated workspace lifecycle."""
