"""Export local display-adapter metadata as bounded, read-only JSON evidence."""

from __future__ import annotations

import json
import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


class DisplayAdaptersCollector(CoreCollector):
    """Capture read-only local display-adapter identities, drivers, and memory through CIM."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated display-adapter JSON artifact contract."""
        return CollectorMetadata(
            id="core.hardware.display_adapters", name="Display adapters", version="4.0.0", specialty=Specialty.HARDWARE,
            output_media_types=("application/json",),
            description="Exports local display-adapter names, driver versions, resolution, and memory metadata.",
            author="Logicytics", supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("hardware_inventory",), default_profiles=("deep",), timeout_seconds=30,
            maximum_output_bytes=256 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query display-adapter CIM metadata and register a bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before display-adapter collection")
        context.report_progress("display_adapters_started")
        command = (
            "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterCompatibility, DriverVersion, "
            "VideoProcessor, AdapterRAM, CurrentHorizontalResolution, CurrentVerticalResolution, CurrentRefreshRate "
            "| ConvertTo-Json -Depth 3"
        )
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                   capture_output=True, check=False, text=True, timeout=25)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            status = CollectorStatus.SKIPPED if "denied" in detail.casefold() else CollectorStatus.FAILED
            return CollectorResult(status, "display-adapter query failed", errors=(detail,))
        try:
            adapters = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "display-adapter query returned invalid JSON", errors=(str(error),))
        if not isinstance(adapters, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "display-adapter query returned an unexpected result")
        output = context.workspace / "display_adapters.json"
        output.write_text(json.dumps(adapters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(adapters) if isinstance(adapters, list) else 1
        context.report_progress("display_adapters_finished", adapter_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("display adapters collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
