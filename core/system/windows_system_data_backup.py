"""Copy bounded Windows policy, event-log, and security-support evidence."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, EvidenceKind, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus

MAX_FILE_BYTES = 100 * 1024 * 1024


class WindowsSystemDataBackupCollector(CoreCollector):
    """Back up bounded Windows system-data paths into labeled private directories."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the explicit-consent Windows system-data backup contract."""
        return CollectorMetadata(
            id="core.system.windows_system_data_backup", name="Windows system-data backup", version="4.0.0",
            specialty=Specialty.SYSTEM,
            description="Copies bounded Group Policy, event-log, and Windows security-support evidence.",
            author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.FILESYSTEM_READ, Capability.SENSITIVE_FILES),
            sensitive_data_categories=("security_logs", "group_policy", "system_configuration"),
            default_profiles=("deep",), timeout_seconds=180, maximum_output_bytes=256 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state before system-data copying starts."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Copy supported Windows system-data files and preserve source metadata."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Windows system-data backup")
        windows = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        program_data = Path(os.environ.get("ProgramData", r"C:\ProgramData"))
        candidates = [
            ("group_policy", windows / "System32" / "GroupPolicy" / "Machine" / "Registry.pol"),
            ("group_policy", windows / "System32" / "GroupPolicy" / "User" / "Registry.pol"),
            ("event_logs", windows / "System32" / "winevt" / "Logs" / "System.evtx"),
            ("event_logs", windows / "System32" / "winevt" / "Logs" / "Application.evtx"),
            ("event_logs", windows / "System32" / "winevt" / "Logs" / "Security.evtx"),
        ]
        candidates.extend(("security_support", path) for path in
                          (program_data / "Microsoft" / "Windows Defender" / "Support").glob("*.log"))
        copied: list[Path] = []
        context.report_progress("windows_system_data_backup_started")
        for label, source in candidates:
            if context.is_cancelled:
                return CollectorResult(CollectorStatus.CANCELLED, "cancelled during Windows system-data backup")
            try:
                if not source.is_file() or source.is_symlink() or source.stat().st_size > MAX_FILE_BYTES:
                    continue
                destination = context.workspace / "windows_system_data" / label / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            except OSError:
                continue
            if context.is_cancelled:
                destination.unlink(missing_ok=True)
                for copied_path in copied:
                    copied_path.unlink(missing_ok=True)
                return CollectorResult(CollectorStatus.CANCELLED, "cancelled during Windows system-data backup")
            copied.append(destination)
        if not copied:
            return CollectorResult(CollectorStatus.SKIPPED,
                                   "no configured Windows system-data files met the bounded backup policy")
        artifacts = []
        for path in copied:
            if context.is_cancelled:
                for unpublished in copied[len(artifacts):]:
                    unpublished.unlink(missing_ok=True)
                return CollectorResult.cancelled("cancelled during system-data registration", tuple(artifacts))
            artifacts.append(context.artifacts.register_file(path, evidence_kind=EvidenceKind.RAW))
        artifact_tuple = tuple(artifacts)
        context.report_progress("windows_system_data_backup_finished", copied_files=len(artifact_tuple),
                                bytes_written=sum(item.size_bytes for item in artifact_tuple))
        return CollectorResult.succeeded("Windows system-data backup collected", artifact_tuple)

    def cleanup(self, context: CollectorContext) -> None:
        """Leave copied evidence removal to the isolated workspace lifecycle."""
