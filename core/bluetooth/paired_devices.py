"""Collect Windows Plug and Play metadata for Bluetooth devices."""

from __future__ import annotations

import json
import subprocess
from shutil import which

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
    """Recognize Windows and PowerShell access-denied messages."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class PairedDevicesCollector(CoreCollector):
    """Export Bluetooth PnP device metadata without changing Bluetooth state."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the read-only PnP query, subprocess permission, and bounded output."""
        return CollectorMetadata(
            id="core.bluetooth.paired_devices",
            name="Bluetooth devices",
            version="4.0.0",
            specialty=Specialty.BLUETOOTH,
            output_media_types=("application/json",),
            description="Captures available Bluetooth Plug and Play device metadata through PowerShell.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            default_profiles=("deep",),
            timeout_seconds=30,
            maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before issuing the PnP query."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run a read-only Bluetooth PnP query and register its normalized JSON output."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Bluetooth collection")
        context.report_progress("bluetooth_devices_started")
        command = (
            "Get-PnpDevice -Class Bluetooth | "
            "Select-Object FriendlyName, InstanceId, Status, Class, Problem, Present | "
            "ConvertTo-Json -Depth 2"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "Bluetooth PnP access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "Bluetooth PnP query failed", errors=(detail,))
        try:
            payload = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(
                CollectorStatus.FAILED,
                "Bluetooth PnP query returned invalid JSON",
                errors=(str(error),),
            )
        devices = payload if isinstance(payload, list) else [payload]
        output = context.workspace / "bluetooth_devices.json"
        output.write_text(json.dumps(devices, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("bluetooth_devices_finished", device_count=len(devices),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Bluetooth device metadata collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the PowerShell process has already exited."""
