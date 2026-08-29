"""Collect bounded aggregate physical and virtual memory statistics from Windows."""

from __future__ import annotations

import ctypes
import json
from datetime import datetime, timezone

from logicytics import CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


class _MemoryStatus(ctypes.Structure):
    """Mirror the Windows MEMORYSTATUSEX structure used by GlobalMemoryStatusEx."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _read_memory_status() -> _MemoryStatus:
    """Read Windows aggregate memory counters or raise the platform error."""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status


class MemorySnapshotCollector(CoreCollector):
    """Write a JSON report containing aggregate memory and page-file availability."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare this bounded, capability-free Windows memory collector."""
        return CollectorMetadata(
            id="core.memory.memory_snapshot",
            name="Memory snapshot",
            version="4.0.0",
            specialty=Specialty.MEMORY,
            output_media_types=("application/json",),
            description="Captures aggregate physical, virtual, and page-file memory statistics.",
            author="Logicytics",
            supported_platforms=("win32",),
            default_profiles=("minimal", "standard", "deep", "offline"),
            timeout_seconds=10,
            maximum_output_bytes=64 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation and Windows memory API availability without writing artifacts."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        try:
            _read_memory_status()
        except OSError as error:
            return ValidationResult(False, reasons=(f"memory API is unavailable: {error}",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Capture aggregate counters and register the resulting JSON report."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before memory collection")
        context.report_progress("memory_snapshot_started")
        try:
            status = _read_memory_status()
        except OSError as error:
            return CollectorResult(CollectorStatus.FAILED, "could not read Windows memory status", errors=(str(error),))
        report = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "memory_load_percent": status.dwMemoryLoad,
            "physical_memory": {
                "total_bytes": status.ullTotalPhys,
                "available_bytes": status.ullAvailPhys,
                "used_bytes": status.ullTotalPhys - status.ullAvailPhys,
            },
            "page_file": {
                "total_bytes": status.ullTotalPageFile,
                "available_bytes": status.ullAvailPageFile,
                "used_bytes": status.ullTotalPageFile - status.ullAvailPageFile,
            },
            "virtual_memory": {
                "total_bytes": status.ullTotalVirtual,
                "available_bytes": status.ullAvailVirtual,
                "used_bytes": status.ullTotalVirtual - status.ullAvailVirtual,
            },
        }
        output = context.workspace / "memory_snapshot.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("memory_snapshot_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("memory snapshot collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the Windows memory API is synchronous."""
