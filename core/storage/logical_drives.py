"""Collect bounded Windows logical-drive capacity and type metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from logicytics import CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.ctypes_collector import get_logical_drives, ularge_integer, get_disk_free_space, get_drive_type

_DRIVE_TYPES = {
    0: "unknown",
    1: "no_root_directory",
    2: "removable",
    3: "fixed",
    4: "remote",
    5: "optical",
    6: "ram_disk",
}


def _logical_drives() -> list[dict[str, int | str]]:
    """Return Windows logical drive metadata without walking any filesystem contents."""
    mask = get_logical_drives()
    drives: list[dict[str, int | str]] = []

    for offset in range(26):
        if not mask & (1 << offset):
            continue

        root = f"{chr(ord('A') + offset)}:\\"
        available = ularge_integer()
        total = ularge_integer()
        free = ularge_integer()

        if not get_disk_free_space(
                root,
                available,
                total,
                free,
        ):
            continue

        drive_type_code = get_drive_type(root)

        drives.append(
            {
                "root": root,
                "type": _DRIVE_TYPES.get(drive_type_code, "unknown"),
                "total_bytes": int(total.value),
                "free_bytes": int(free.value),
                "available_bytes": int(available.value),
            }
        )

    return drives


class LogicalDrivesCollector(CoreCollector):
    """Write a JSON inventory of mounted logical-drive capacity and type metadata."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare a bounded, capability-free storage metadata collector."""
        return CollectorMetadata(
            id="core.storage.logical_drives",
            name="Logical drives",
            version="4.0.0",
            specialty=Specialty.STORAGE,
            output_media_types=("application/json",),
            description="Records mounted logical-drive types and aggregate capacity metadata.",
            author="Logicytics",
            supported_platforms=("win32",),
            default_profiles=("minimal", "standard", "deep", "offline"),
            timeout_seconds=10,
            maximum_output_bytes=64 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation and ensure the Windows logical-drive API is available."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        try:
            _logical_drives()
        except OSError as error:
            return ValidationResult(False, reasons=(f"logical-drive API is unavailable: {error}",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Capture logical-drive metadata and register a bounded JSON report."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before storage collection")
        context.report_progress("logical_drives_started")
        try:
            drives = _logical_drives()
        except OSError as error:
            return CollectorResult(CollectorStatus.FAILED, "could not read logical drives", errors=(str(error),))
        report = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "drives": drives,
        }
        output = context.workspace / "logical_drives.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("logical_drives_finished", drive_count=len(drives), bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("logical-drive metadata collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the Windows storage API is synchronous."""
