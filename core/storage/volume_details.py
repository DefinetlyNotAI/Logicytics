"""Export detailed mounted Windows volume metadata through bounded native API calls."""

from __future__ import annotations

import ctypes
import json
from ctypes import wintypes
from datetime import datetime, timezone

from logicytics import CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus

_DRIVE_TYPES = {0: "unknown", 1: "no_root_directory", 2: "removable", 3: "fixed", 4: "remote", 5: "optical",
                6: "ram_disk"}


def _volume_details() -> list[dict[str, int | str]]:
    """Read mounted logical-volume capacity, label, and filesystem metadata only."""
    kernel32 = ctypes.windll.kernel32
    mask = kernel32.GetLogicalDrives()
    if mask == 0:
        raise ctypes.WinError()
    volumes: list[dict[str, int | str]] = []
    for offset in range(26):
        if not mask & (1 << offset):
            continue
        root = f"{chr(ord('A') + offset)}:\\"
        available = ctypes.c_ulonglong()
        total = ctypes.c_ulonglong()
        free = ctypes.c_ulonglong()
        if not kernel32.GetDiskFreeSpaceExW(root, ctypes.byref(available), ctypes.byref(total), ctypes.byref(free)):
            continue
        label = ctypes.create_unicode_buffer(261)
        filesystem = ctypes.create_unicode_buffer(261)
        serial = wintypes.DWORD()
        maximum_component_length = wintypes.DWORD()
        flags = wintypes.DWORD()
        information_available = kernel32.GetVolumeInformationW(
            root, label, len(label), ctypes.byref(serial), ctypes.byref(maximum_component_length), ctypes.byref(flags),
            filesystem, len(filesystem)
        )
        volumes.append({
            "root": root,
            "drive_type": _DRIVE_TYPES.get(kernel32.GetDriveTypeW(root), "unknown"),
            "filesystem": filesystem.value if information_available else "unavailable",
            "volume_name": label.value if information_available else "unavailable",
            "total_bytes": total.value,
            "free_bytes": free.value,
            "available_bytes": available.value,
        })
    return volumes


class VolumeDetailsCollector(CoreCollector):
    """Capture detailed mounted logical-volume metadata without walking file contents."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the capability-free detailed volume artifact contract."""
        return CollectorMetadata(
            id="core.storage.volume_details", name="Volume details", version="4.0.0", specialty=Specialty.STORAGE,
            output_media_types=("application/json",),
            description="Exports mounted drive type, filesystem, label, and capacity metadata.", author="Logicytics",
            supported_platforms=("win32",), default_profiles=("deep",), timeout_seconds=15,
            maximum_output_bytes=128 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation and native volume API availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        try:
            _volume_details()
        except OSError as error:
            return ValidationResult(False, reasons=(f"volume API is unavailable: {error}",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Read detailed volume metadata and register a JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before volume-details collection")
        context.report_progress("volume_details_started")
        try:
            volumes = _volume_details()
        except OSError as error:
            return CollectorResult(CollectorStatus.FAILED, "could not read detailed volume metadata",
                                   errors=(str(error),))
        output = context.workspace / "volume_details.json"
        output.write_text(
            json.dumps({"collected_at": datetime.now(timezone.utc).isoformat(), "volumes": volumes}, indent=2,
                       sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("volume_details_finished", volume_count=len(volumes), bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("detailed volume metadata collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because Windows volume API calls are synchronous."""
