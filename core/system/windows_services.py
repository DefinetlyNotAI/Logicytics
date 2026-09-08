"""Export local Windows service metadata as bounded, read-only JSON evidence."""

from __future__ import annotations

import json

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


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class WindowsServicesCollector(CoreCollector):
    """Capture read-only service names, states, startup modes, and paths through CIM."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated Windows-services JSON artifact contract."""
        return CollectorMetadata(
            id="core.system.windows_services",
            name="Windows services",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("application/json",),
            description="Exports local Windows service names, states, start modes, accounts, and executable paths.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",),
            timeout_seconds=60,
            maximum_output_bytes=4 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query service CIM metadata and register a bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Windows-services collection")
        context.report_progress("windows_services_started")
        command = (
            "Get-CimInstance -ClassName Win32_Service | "
            "Select-Object Name, DisplayName, State, StartMode, StartName, PathName, ProcessId | ConvertTo-Json -Depth 3"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=55,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "Windows-service access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "Windows-service query failed", errors=(detail,))
        try:
            services = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(
                CollectorStatus.FAILED,
                "Windows-service query returned invalid JSON",
                errors=(str(error),),
            )
        if not isinstance(services, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "Windows-service query returned an unexpected result")
        output = context.workspace / "windows_services.json"
        output.write_text(json.dumps(services, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(services) if isinstance(services, list) else 1
        context.report_progress("windows_services_finished", service_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Windows services collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
