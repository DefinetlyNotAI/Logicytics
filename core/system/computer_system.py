"""Export local Windows computer model, manufacturer, and processor-count CIM data."""

from __future__ import annotations

import json
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class ComputerSystemCollector(CoreCollector):
    """Capture read-only local computer-system identity and hardware metadata."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated computer-system JSON artifact contract."""
        return CollectorMetadata(
            id="core.system.computer_system",
            name="Computer system",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("application/json",),
            description="Exports local computer model, manufacturer, and processor-count CIM data.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=128 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query computer-system CIM data and register a JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before computer-system collection")
        context.report_progress("computer_system_started")
        command = (
            "Get-CimInstance -ClassName Win32_ComputerSystem | "
            "Select-Object Name, Manufacturer, Model, SystemType, NumberOfProcessors, "
            "NumberOfLogicalProcessors, TotalPhysicalMemory | ConvertTo-Json -Depth 3"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=40,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "computer-system CIM access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "computer-system CIM query failed", errors=(detail,))
        try:
            computer_system = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(
                CollectorStatus.FAILED,
                "computer-system CIM query returned invalid JSON",
                errors=(str(error),),
            )
        if not isinstance(computer_system, dict):
            return CollectorResult(CollectorStatus.FAILED, "computer-system CIM query returned an unexpected result")
        output = context.workspace / "computer_system.json"
        output.write_text(json.dumps(computer_system, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("computer_system_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("computer-system details collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
