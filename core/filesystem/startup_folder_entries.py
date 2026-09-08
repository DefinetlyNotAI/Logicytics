"""Export bounded metadata for files in the standard Windows Startup folders."""

from __future__ import annotations

import json
import os
from pathlib import Path

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import filesystem_adapter

_MAXIMUM_ENTRIES = 500


def _startup_folders() -> tuple[Path, ...]:
    """Return the standard per-user and all-users Startup folders without creating them."""
    folders: list[Path] = []
    appdata = os.environ.get("APPDATA")
    programdata = os.environ.get("PROGRAMDATA")
    if appdata:
        folders.append(Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup")
    if programdata:
        folders.append(Path(programdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup")
    return tuple(folders)


class StartupFolderEntriesCollector(CoreCollector):
    """Capture read-only metadata for entries configured to start through Startup folders."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the filesystem-read, bounded Startup-folder JSON artifact contract."""
        return CollectorMetadata(
            id="core.filesystem.startup_folder_entries",
            name="Startup folder entries",
            version="4.0.0",
            specialty=Specialty.FILESYSTEM,
            output_media_types=("application/json",),
            description="Exports names, locations, sizes, and timestamps for up to 500 Startup-folder entries.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.FILESYSTEM_READ,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",),
            timeout_seconds=30,
            maximum_output_bytes=256 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and whether at least one standard folder is available."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if not _startup_folders():
            return ValidationResult(False, reasons=("Windows Startup-folder environment variables are unavailable",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Record bounded metadata for direct files in existing Startup folders."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Startup-folder collection")
        context.report_progress("startup_folder_entries_started")
        entries: list[dict[str, int | str]] = []
        missing_folders = 0
        for folder in _startup_folders():
            if not folder.is_dir():
                missing_folders += 1
                continue
            try:
                children = sorted(filesystem_adapter.children(folder), key=lambda path: path.name.casefold())
            except OSError as error:
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "Startup-folder access was denied",
                    errors=(str(error),),
                )
            for child in children:
                if context.is_cancelled:
                    return CollectorResult(CollectorStatus.CANCELLED, "cancelled during Startup-folder collection")
                if len(entries) >= _MAXIMUM_ENTRIES:
                    break
                try:
                    stat = child.stat()
                except OSError:
                    continue
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "is_directory": str(child.is_dir()).lower(),
                        "size_bytes": stat.st_size,
                        "modified_epoch": int(stat.st_mtime),
                    }
                )
        output = context.workspace / "startup_folder_entries.json"
        output.write_text(
            json.dumps({"entries": entries, "missing_folders": missing_folders}, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress(
            "startup_folder_entries_finished",
            entry_count=len(entries),
            bytes_written=artifact.size_bytes,
        )
        return CollectorResult.succeeded("Startup-folder entries collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because collection only reads directory metadata."""
