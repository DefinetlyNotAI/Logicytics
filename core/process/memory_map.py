"""Export readable memory-region metadata for the isolated collector process."""

from __future__ import annotations

import ctypes
import json
import shutil
from ctypes import wintypes
from pathlib import Path

from logicytics import CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


class _MemoryBasicInformation(ctypes.Structure):
    _fields_ = [("BaseAddress", wintypes.LPVOID), ("AllocationBase", wintypes.LPVOID),
                ("AllocationProtect", wintypes.DWORD), ("PartitionId", wintypes.WORD), ("RegionSize", ctypes.c_size_t),
                ("State", wintypes.DWORD), ("Protect", wintypes.DWORD), ("Type", wintypes.DWORD)]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def _permissions(protection: int) -> str:
    """Return a readable Windows page-protection summary."""
    values = {0x02: "read", 0x04: "read_write", 0x08: "write_copy", 0x20: "execute_read", 0x40: "execute_read_write",
              0x80: "execute_write_copy"}
    return values.get(protection & 0xFF, "unreadable")


class MemoryMapCollector(CoreCollector):
    """Capture bounded virtual-memory metadata from the isolated worker process."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the bounded memory-region JSON artifact contract."""
        return CollectorMetadata(
            id="core.process.memory_map", name="Process memory map", version="4.0.0", specialty=Specialty.PROCESS,
            output_media_types=("application/json",),
            description="Exports readable virtual-memory region addresses, sizes, permissions, paths, and process RSS.",
            author="Logicytics",
            supported_platforms=("win32",), sensitive_data_categories=("process_metadata",), default_profiles=("deep",),
            timeout_seconds=90, maximum_output_bytes=64 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Validate cancellation state and bounded region settings."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        try:
            maximum = int(context.settings.get("max_regions", 5_000))
            output_limit = int(context.settings.get("output_limit_bytes", 64 * 1024 * 1024))
            safety_margin = int(context.settings.get("disk_safety_margin_bytes", 100 * 1024 * 1024))
        except (TypeError, ValueError):
            return ValidationResult(False, reasons=("memory-map limits must be integers",))
        if not 1 <= maximum <= 100_000 or not 1_024 <= output_limit <= 64 * 1024 * 1024 or safety_margin < 0:
            return ValidationResult(False, reasons=("invalid memory-map region, output, or safety limits",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query readable local memory regions and write metadata-only JSON evidence."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before memory-map collection")
        maximum = int(context.settings.get("max_regions", 5_000))
        output_limit = int(context.settings.get("output_limit_bytes", 64 * 1024 * 1024))
        safety_margin = int(context.settings.get("disk_safety_margin_bytes", 100 * 1024 * 1024))
        configured_directory = Path(str(context.settings.get("dump_directory", "memory_maps")))
        output_directory = (context.workspace / configured_directory).resolve()
        try:
            output_directory.relative_to(context.workspace.resolve())
        except ValueError:
            return CollectorResult(CollectorStatus.FAILED,
                                   "memory-map dump_directory must stay inside the collector workspace")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        process = kernel32.GetCurrentProcess()
        counters = _ProcessMemoryCounters()
        psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), ctypes.sizeof(counters))
        memory = _MemoryBasicInformation()
        address = 0
        regions: list[dict[str, object]] = []
        context.report_progress("memory_map_started", max_regions=maximum)
        while len(regions) < maximum:
            if context.is_cancelled:
                return CollectorResult(CollectorStatus.CANCELLED, "cancelled during memory-map collection")
            queried = kernel32.VirtualQuery(ctypes.c_void_p(address), ctypes.byref(memory), ctypes.sizeof(memory))
            if not queried or memory.RegionSize == 0:
                break
            base = ctypes.cast(memory.BaseAddress, ctypes.c_void_p).value or address
            readable = memory.State == 0x1000 and not (memory.Protect & 0x101)
            if readable:
                mapped = ctypes.create_unicode_buffer(32_768)
                mapped_length = psapi.GetMappedFileNameW(process, ctypes.c_void_p(base), mapped, len(mapped))
                regions.append({"index": len(regions), "address": f"0x{base:016X}", "size_bytes": memory.RegionSize,
                                "rss_bytes": counters.WorkingSetSize, "permissions": _permissions(memory.Protect),
                                "mapped_path": mapped.value if mapped_length else None, "state": memory.State,
                                "type": memory.Type})
            next_address = base + memory.RegionSize
            if next_address <= address:
                break
            address = next_address
        truncated = False
        while True:
            if context.is_cancelled:
                return CollectorResult(CollectorStatus.CANCELLED, "cancelled during memory-map serialization")
            serialized = json.dumps({"region_count": len(regions), "truncated": truncated, "regions": regions},
                                    indent=2) + "\n"
            if len(serialized.encode("utf-8")) <= output_limit or not regions:
                break
            regions.pop()
            truncated = True
        if shutil.disk_usage(context.workspace).free < len(serialized.encode("utf-8")) + safety_margin:
            return CollectorResult(CollectorStatus.SKIPPED,
                                   "insufficient free disk space after configured memory-map safety margin")
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before memory-map publication")
        output_directory.mkdir(parents=True, exist_ok=True)
        output = output_directory / "memory_map.json"
        output.write_text(serialized, encoding="utf-8")
        if context.is_cancelled:
            output.unlink(missing_ok=True)
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled during memory-map publication")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("memory_map_finished", region_count=len(regions), truncated=str(truncated).lower(),
                                bytes_written=artifact.size_bytes)
        summary = "process memory map collected" if not truncated else "process memory map collected with configured output truncation"
        return CollectorResult.succeeded(summary, (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the collector only queries its own process."""
