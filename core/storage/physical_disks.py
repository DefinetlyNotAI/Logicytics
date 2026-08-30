"""Export local Windows physical disk model and capacity CIM data as bounded JSON."""

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


class PhysicalDisksCollector(CoreCollector):
    """Capture a read-only inventory of locally attached physical disk hardware."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated physical-disk JSON artifact contract."""
        return CollectorMetadata(
            id="core.storage.physical_disks",
            name="Physical disks",
            version="4.0.0",
            specialty=Specialty.STORAGE,
            output_media_types=("application/json",),
            description="Exports local physical disk models, media types, interface types, and sizes.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query physical-disk CIM data and register a JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before physical-disk collection")
        context.report_progress("physical_disks_started")
        command = (
            "Get-CimInstance -ClassName Win32_DiskDrive | "
            "Select-Object Model, Size, MediaType, InterfaceType, DeviceID, Partitions | "
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
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "physical-disk CIM access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "physical-disk CIM query failed", errors=(detail,))
        try:
            physical_disks = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "physical-disk CIM query returned invalid JSON",
                                   errors=(str(error),))
        if not isinstance(physical_disks, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "physical-disk CIM query returned an unexpected result")
        output = context.workspace / "physical_disks.json"
        output.write_text(json.dumps(physical_disks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(physical_disks) if isinstance(physical_disks, list) else 1
        context.report_progress("physical_disks_finished", disk_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("physical-disk inventory collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
