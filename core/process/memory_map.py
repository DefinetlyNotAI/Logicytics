"""Export readable memory-region metadata for the isolated collector process."""

from __future__ import annotations

import json
from pathlib import Path

from logicytics import CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics import ProcessMemoryCounters, MemoryBasicInformation, get_process_memory_info, \
    virtual_query, get_mapped_file_name, get_current_process, pointer_value
from logicytics.platform_adapters import filesystem_adapter


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
            return ValidationResult(
                False,
                reasons=("run cancellation was requested",),
            )

        def setting_int(name: str, default: int) -> int:
            value = context.settings.get(name, default)

            if isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")

            if isinstance(value, (int, str, bytes, bytearray)):
                return int(value)

            raise ValueError(f"{name} must be an integer")

        try:
            maximum = setting_int("max_regions", 5_000)
            output_limit = setting_int(
                "output_limit_bytes",
                64 * 1024 * 1024,
            )
            safety_margin = setting_int(
                "disk_safety_margin_bytes",
                100 * 1024 * 1024,
            )
        except (TypeError, ValueError):
            return ValidationResult(
                False,
                reasons=("memory-map limits must be integers",),
            )

        if (
                not 1 <= maximum <= 100_000
                or not 1_024 <= output_limit <= 64 * 1024 * 1024
                or safety_margin < 0
        ):
            return ValidationResult(
                False,
                reasons=("invalid memory-map region, output, or safety limits",),
            )

        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query readable local memory regions and write metadata-only JSON evidence."""
        if context.is_cancelled:
            return CollectorResult(
                CollectorStatus.CANCELLED,
                "cancelled before memory-map collection",
            )

        maximum = context.setting_int("max_regions", 5_000)
        output_limit = context.setting_int(
            "output_limit_bytes",
            64 * 1024 * 1024,
        )
        safety_margin = context.setting_int(
            "disk_safety_margin_bytes",
            100 * 1024 * 1024,
        )

        configured_directory = Path(
            context.setting_str("dump_directory", "memory_maps")
        )
        output_directory = (
                context.workspace / configured_directory
        ).resolve()

        try:
            output_directory.relative_to(context.workspace.resolve())
        except ValueError:
            return CollectorResult(
                CollectorStatus.FAILED,
                "memory-map dump_directory must stay inside the collector workspace",
            )

        process = get_current_process()

        counters = ProcessMemoryCounters()
        if not get_process_memory_info(process, counters):
            return CollectorResult(
                CollectorStatus.FAILED,
                "could not query process memory counters",
            )

        memory = MemoryBasicInformation()
        address = 0
        regions: list[dict[str, object]] = []

        context.report_progress(
            "memory_map_started",
            max_regions=maximum,
        )

        while len(regions) < maximum:
            if context.is_cancelled:
                return CollectorResult(
                    CollectorStatus.CANCELLED,
                    "cancelled during memory-map collection",
                )

            queried = virtual_query(address, memory)
            region_size = int(memory.RegionSize)

            if not queried or region_size == 0:
                break

            base_address = pointer_value(memory.BaseAddress)
            if base_address is None:
                base_address = address

            readable = (
                    memory.State == 0x1000
                    and not memory.Protect & 0x101
            )

            if readable:
                mapped_path = get_mapped_file_name(
                    process,
                    base_address,
                )

                regions.append(
                    {
                        "index": len(regions),
                        "address": f"0x{base_address:016X}",
                        "size_bytes": region_size,
                        "rss_bytes": int(counters.WorkingSetSize),
                        "permissions": _permissions(int(memory.Protect)),
                        "mapped_path": mapped_path,
                        "state": int(memory.State),
                        "type": int(memory.Type),
                    }
                )

            next_address = base_address + region_size

            if next_address <= address:
                break

            address = next_address

        truncated = False

        while True:
            if context.is_cancelled:
                return CollectorResult(
                    CollectorStatus.CANCELLED,
                    "cancelled during memory-map serialization",
                )

            serialized = json.dumps(
                {
                    "region_count": len(regions),
                    "truncated": truncated,
                    "regions": regions,
                },
                indent=2,
            ) + "\n"

            serialized_size = len(serialized.encode("utf-8"))

            if serialized_size <= output_limit or not regions:
                break

            regions.pop()
            truncated = True

        if (
                filesystem_adapter.disk_usage(context.workspace).free
                < serialized_size + safety_margin
        ):
            return CollectorResult(
                CollectorStatus.SKIPPED,
                "insufficient free disk space after configured memory-map safety margin",
            )

        if context.is_cancelled:
            return CollectorResult(
                CollectorStatus.CANCELLED,
                "cancelled before memory-map publication",
            )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = output_directory / "memory_map.json"
        output.write_text(
            serialized,
            encoding="utf-8",
        )

        if context.is_cancelled:
            output.unlink(missing_ok=True)

            return CollectorResult(
                CollectorStatus.CANCELLED,
                "cancelled during memory-map publication",
            )

        artifact = context.artifacts.register_file(
            output,
            media_type="application/json",
        )

        context.report_progress(
            "memory_map_finished",
            region_count=len(regions),
            truncated=str(truncated).lower(),
            bytes_written=artifact.size_bytes,
        )

        summary = (
            "process memory map collected"
            if not truncated
            else "process memory map collected with configured output truncation"
        )

        return CollectorResult.succeeded(
            summary,
            (artifact,),
        )

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the collector only queries its own process."""
