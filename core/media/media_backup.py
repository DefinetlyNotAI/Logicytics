"""Back up bounded current-user Pictures and Videos evidence after explicit approval."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    EvidenceKind,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import filesystem_adapter

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4"}
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 1_000


class MediaBackupCollector(CoreCollector):
    """Copy supported current-user media into the collector's private workspace."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the explicit-consent media backup artifact contract."""
        return CollectorMetadata(
            id="core.media.media_backup",
            name="Pictures and Videos backup",
            version="4.0.0",
            specialty=Specialty.MEDIA,
            output_media_types=("application/octet-stream",),
            description="Copies bounded JPG, JPEG, PNG, and MP4 files from current-user Pictures and Videos folders.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.FILESYSTEM_READ, Capability.SENSITIVE_FILES),
            sensitive_data_categories=("personal_media",),
            default_profiles=("deep",),
            timeout_seconds=300,
            maximum_output_bytes=512 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state before the bounded media scan starts."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Copy supported regular media files while preserving source metadata."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before media backup")
        home = filesystem_adapter.home()
        source_roots = (("pictures", home / "Pictures"), ("videos", home / "Videos"))
        destination_root = context.workspace / "media_backup"
        copied: list[Path] = []
        copied_bytes = 0
        skipped_files = 0
        inaccessible_roots = 0
        context.report_progress("media_backup_started")
        for category, source_root in source_roots:
            try:
                exists = source_root.is_dir()
            except OSError:
                inaccessible_roots += 1
                continue
            if not exists:
                continue
            try:
                candidates = filesystem_adapter.recursive(source_root)
                for candidate in candidates:
                    if context.is_cancelled:
                        return CollectorResult(CollectorStatus.CANCELLED, "cancelled during media backup")
                    if len(copied) >= MAX_FILES:
                        break
                    try:
                        if not candidate.is_file() or candidate.is_symlink() or candidate.suffix.casefold() not in SUPPORTED_EXTENSIONS:
                            continue
                        size = candidate.stat().st_size
                    except OSError:
                        skipped_files += 1
                        continue
                    if size > MAX_FILE_BYTES or copied_bytes + size > MAX_TOTAL_BYTES:
                        skipped_files += 1
                        continue
                    timestamp = datetime.fromtimestamp(candidate.stat().st_mtime, UTC).strftime("%Y%m%dT%H%M%SZ")
                    destination = destination_root / category / f"{candidate.stem}_{timestamp}{candidate.suffix.casefold()}"
                    suffix = 1
                    while destination.exists():
                        destination = destination_root / category / f"{candidate.stem}_{timestamp}_{suffix}{candidate.suffix.casefold()}"
                        suffix += 1
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    filesystem_adapter.copy_file(candidate, destination)
                    if context.is_cancelled:
                        destination.unlink(missing_ok=True)
                        for copied_path in copied:
                            copied_path.unlink(missing_ok=True)
                        return CollectorResult(CollectorStatus.CANCELLED, "cancelled during media backup")
                    copied.append(destination)
                    copied_bytes += size
            except OSError:
                inaccessible_roots += 1
        if not copied:
            reason = "no supported media files met the bounded backup policy"
            if inaccessible_roots == len(source_roots):
                reason = "the current user's Pictures and Videos folders are inaccessible"
            return CollectorResult(CollectorStatus.SKIPPED, reason)
        artifacts = []
        for path in copied:
            if context.is_cancelled:
                for unpublished in copied[len(artifacts):]:
                    unpublished.unlink(missing_ok=True)
                return CollectorResult.cancelled("cancelled during media registration", tuple(artifacts))
            artifacts.append(context.artifacts.register_file(path, evidence_kind=EvidenceKind.RAW))
        artifact_tuple = tuple(artifacts)
        context.report_progress(
            "media_backup_finished",
            copied_files=len(artifacts),
            skipped_files=skipped_files,
            bytes_written=sum(item.size_bytes for item in artifact_tuple),
        )
        return CollectorResult.succeeded("current-user media backup collected", artifact_tuple)

    def cleanup(self, context: CollectorContext) -> None:
        """Leave copied evidence removal to the isolated workspace lifecycle."""
