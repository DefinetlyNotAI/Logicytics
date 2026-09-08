"""Create a bounded recursive tree of the Windows system drive without reading file contents."""

from __future__ import annotations

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

_DEFAULT_MAX_ENTRIES = 5_000
_DEFAULT_MAX_DEPTH = 12
_HARD_MAX_ENTRIES = 50_000
_HARD_MAX_DEPTH = 32


def _bounded_setting(settings: object, key: str, default: int, maximum: int) -> int:
    """Return one integer traversal setting constrained to a safe positive range."""
    value = settings.get(key, default) if isinstance(settings, dict) else default
    return value if isinstance(value, int) and 1 <= value <= maximum else default


class SystemDriveTreeCollector(CoreCollector):
    """Capture a bounded recursive system-drive tree without reading file contents."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the filesystem-read, bounded system-drive tree artifact contract."""
        return CollectorMetadata(
            id="core.filesystem.system_drive_tree",
            name="System drive tree",
            version="4.0.0",
            specialty=Specialty.FILESYSTEM,
            output_media_types=("text/plain",),
            description="Exports a configurable bounded recursive directory tree for the Windows system drive.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.FILESYSTEM_READ,),
            sensitive_data_categories=("filesystem_metadata",),
            default_profiles=("deep",),
            timeout_seconds=120,
            maximum_output_bytes=8 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and system-drive availability before traversal."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        system_drive = filesystem_adapter.system_drive_root()
        if not system_drive.is_dir():
            return ValidationResult(False, reasons=(f"system drive is unavailable: {system_drive}",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Walk a bounded tree and register the metadata-only text evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before system-drive tree collection")
        maximum_entries = _bounded_setting(context.settings, "max_entries", _DEFAULT_MAX_ENTRIES, _HARD_MAX_ENTRIES)
        maximum_depth = _bounded_setting(context.settings, "max_depth", _DEFAULT_MAX_DEPTH, _HARD_MAX_DEPTH)
        root = filesystem_adapter.system_drive_root()
        lines = [
            f"# system_drive={root}",
            f"# max_entries={maximum_entries}",
            f"# max_depth={maximum_depth}",
        ]
        entries = 0
        skipped = 0
        truncated = False
        context.report_progress("system_drive_tree_started", max_entries=maximum_entries, max_depth=maximum_depth)
        for current, directories, filenames in filesystem_adapter.walk(root, topdown=True, followlinks=False, onerror=lambda _error: None):
            if context.is_cancelled:
                return CollectorResult(CollectorStatus.CANCELLED, "cancelled during system-drive tree collection")
            relative = Path(current).relative_to(root)
            depth = len(relative.parts)
            if depth >= maximum_depth:
                skipped += len(directories)
                directories[:] = []
            for name, marker in [(name, "/") for name in directories] + [(name, "") for name in filenames]:
                if entries >= maximum_entries:
                    truncated = True
                    break
                lines.append(f"{relative / name}{marker}")
                entries += 1
            if truncated:
                break
        if truncated:
            lines.append("# truncated=true")
        output = context.workspace / "system_drive_tree.txt"
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        context.report_progress(
            "system_drive_tree_finished",
            entry_count=entries,
            skipped_directories=skipped,
            truncated=str(truncated).lower(),
            bytes_written=artifact.size_bytes,
        )
        summary = "system-drive tree collected" if not truncated else "system-drive tree collected with configured entry limit"
        return CollectorResult.succeeded(summary, (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because traversal holds no persistent resources."""
