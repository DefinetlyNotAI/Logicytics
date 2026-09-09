"""Capture a bounded legacy WMIC computer-system inventory when WMIC is installed."""

from __future__ import annotations

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which


class WmicInventoryCollector(CoreCollector):
    """Preserve the optional WMIC integration without making it a host prerequisite."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare a deep-profile, subprocess-gated text evidence contract."""
        return CollectorMetadata(
            id="core.system.wmic_inventory",
            name="WMIC computer-system inventory",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("text/plain",),
            description=("Exports bounded computer-system identity and hardware fields through the optional legacy WMIC executable."),
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",),
            timeout_seconds=30,
            maximum_output_bytes=256 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Treat an absent optional Windows feature as an explicit skip condition."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("wmic") is None:
            return ValidationResult(
                False,
                reasons=("WMIC is not installed; enable the optional WMIC capability to collect this view",),
            )
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run one read-only WMIC query and register its normalized text output."""
        if context.is_cancelled:
            return CollectorResult(
                CollectorStatus.CANCELLED,
                "cancelled before WMIC inventory collection",
            )
        executable = which("wmic")
        if executable is None:
            return CollectorResult(
                CollectorStatus.SKIPPED,
                "WMIC is not installed on this Windows system",
                errors=("enable the optional WMIC capability to collect the legacy view",),
            )
        context.report_progress("wmic_inventory_started")
        completed = subprocess.run(
            [
                executable,
                "computersystem",
                "get",
                ("Name,Manufacturer,Model,SystemType,NumberOfProcessors,NumberOfLogicalProcessors,TotalPhysicalMemory"),
                "/format:list",
            ],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-16",
            errors="replace",
            timeout=25,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            detail = detail or f"WMIC exit code {completed.returncode}"
            if "access" in detail.casefold() and "denied" in detail.casefold():
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "WMIC access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(
                CollectorStatus.FAILED,
                "WMIC computer-system query failed",
                errors=(detail,),
            )
        normalized = completed.stdout.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return CollectorResult(
                CollectorStatus.FAILED,
                "WMIC computer-system query returned no data",
            )
        output = context.workspace / "wmic_inventory.txt"
        output.write_text(normalized + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        context.report_progress("wmic_inventory_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("WMIC computer-system inventory collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because WMIC exits before the result is returned."""
