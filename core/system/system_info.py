"""Collect a bounded, non-sensitive inventory of the current Windows system."""

from __future__ import annotations

import getpass
import json
import os
import platform
from datetime import UTC, datetime

from logicytics import (
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import network_adapter as socket


class SystemInfoCollector(CoreCollector):
    """Create a JSON system inventory through the v4 artifact boundary."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the collector identity, scope, and bounded output contract."""
        return CollectorMetadata(
            id="core.system.system_info",
            name="System information",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("application/json",),
            description="Collects a bounded operating-system and hardware inventory.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(),
            default_profiles=("minimal", "standard", "deep", "offline"),
            timeout_seconds=15,
            maximum_output_bytes=128 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Confirm that the standard-library system APIs are available."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Write and register the system inventory as a JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before system inventory collection")
        context.report_progress("system_inventory_started")
        inventory = {
            "collected_at": datetime.now(UTC).isoformat(),
            "hostname": socket.gethostname(),
            "username": getpass.getuser(),
            "operating_system": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "architecture": platform.architecture(),
            },
            "hardware": {
                "machine": platform.machine(),
                "processor": platform.processor() or "unavailable",
                "cpu_count": os.cpu_count(),
            },
            "python": platform.python_version(),
        }
        output = context.workspace / "system_info.json"
        output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("system_inventory_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded(
            "system inventory collected",
            (artifact,),
        )

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because this collector owns none beyond its workspace."""
