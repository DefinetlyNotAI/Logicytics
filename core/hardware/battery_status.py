"""Export local Windows battery status as bounded read-only JSON evidence."""

from __future__ import annotations

import json
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class BatteryStatusCollector(CoreCollector):
    """Capture read-only battery capacity and charge state through Windows CIM."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated battery-status JSON artifact contract."""
        return CollectorMetadata(
            id="core.hardware.battery_status", name="Battery status", version="4.0.0", specialty=Specialty.HARDWARE,
            output_media_types=("application/json",),
            description="Exports local battery name, status, charge, capacity, and estimated runtime metadata.",
            author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",), timeout_seconds=45, maximum_output_bytes=128 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query battery CIM metadata and register a bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before battery-status collection")
        context.report_progress("battery_status_started")
        command = (
            "Get-CimInstance -ClassName Win32_Battery | "
            "Select-Object Name, BatteryStatus, EstimatedChargeRemaining, EstimatedRunTime, DesignCapacity, FullChargeCapacity, Status | "
            "ConvertTo-Json -Depth 3"
        )
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                   capture_output=True, check=False, text=True, timeout=40)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "battery-status access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "battery-status query failed", errors=(detail,))
        try:
            batteries = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "battery-status query returned invalid JSON",
                                   errors=(str(error),))
        if not isinstance(batteries, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "battery-status query returned an unexpected result")
        output = context.workspace / "battery_status.json"
        output.write_text(json.dumps(batteries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(batteries) if isinstance(batteries, list) else 1
        context.report_progress("battery_status_finished", battery_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("battery status collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
