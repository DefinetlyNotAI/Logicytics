"""Export detailed Windows operating-system CIM data as a bounded JSON artifact."""

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


class OperatingSystemCollector(CoreCollector):
    """Capture read-only Windows operating-system details through CIM."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated operating-system JSON artifact contract."""
        return CollectorMetadata(
            id="core.system.operating_system",
            name="Operating system details",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("application/json",),
            description="Exports detailed local Windows operating-system CIM information.",
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
        """Query operating-system CIM data and register a JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before operating-system collection")
        context.report_progress("operating_system_started")
        command = (
            "Get-CimInstance -ClassName Win32_OperatingSystem | "
            "Select-Object Caption, CSDVersion, Version, BuildNumber, OSArchitecture, "
            "InstallDate, LastBootUpTime, Locale, MUILanguages, SystemDirectory, WindowsDirectory | "
            "ConvertTo-Json -Depth 3"
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
                    "operating-system CIM access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "operating-system CIM query failed", errors=(detail,))
        try:
            operating_system = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(
                CollectorStatus.FAILED,
                "operating-system CIM query returned invalid JSON",
                errors=(str(error),),
            )
        if not isinstance(operating_system, dict):
            return CollectorResult(CollectorStatus.FAILED, "operating-system CIM query returned an unexpected result")
        output = context.workspace / "operating_system.json"
        output.write_text(json.dumps(operating_system, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("operating_system_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("operating-system details collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
