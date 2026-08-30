"""Export paired Bluetooth names and address-like PnP identifiers as bounded JSON."""

from __future__ import annotations

import json
import re
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common access-denied wording from PowerShell PnP queries."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


def _format_address(instance_id: object) -> str | None:
    """Extract one conventional colon-separated address from a PnP instance identifier."""
    match = re.search(r"(?<![0-9A-Fa-f])([0-9A-Fa-f]{12})(?![0-9A-Fa-f])", str(instance_id))
    return ":".join(match.group(1)[offset: offset + 2].upper() for offset in range(0, 12, 2)) if match else None


class BluetoothAddressesCollector(CoreCollector):
    """Capture Bluetooth friendly names and address-like identifiers without changing state."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated Bluetooth-address JSON artifact contract."""
        return CollectorMetadata(
            id="core.bluetooth.bluetooth_addresses", name="Bluetooth addresses", version="4.0.0",
            specialty=Specialty.BLUETOOTH,
            output_media_types=("application/json",),
            description="Exports paired Bluetooth friendly names and address-like identifiers from PnP data.",
            author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("wireless_identifiers",),
            default_profiles=("deep",), timeout_seconds=30, maximum_output_bytes=256 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query PnP data and register Bluetooth names and extracted addresses as JSON."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Bluetooth-address collection")
        context.report_progress("bluetooth_addresses_started")
        command = "Get-PnpDevice -Class Bluetooth | Select-Object FriendlyName, InstanceId, Present | ConvertTo-Json -Depth 2"
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                   capture_output=True, check=False, text=True, timeout=25)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "Bluetooth PnP access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "Bluetooth PnP query failed", errors=(detail,))
        try:
            raw = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "Bluetooth PnP query returned invalid JSON",
                                   errors=(str(error),))
        devices = raw if isinstance(raw, list) else [raw]
        if not all(isinstance(device, dict) for device in devices):
            return CollectorResult(CollectorStatus.FAILED, "Bluetooth PnP query returned an unexpected result")
        report = [{"friendly_name": device.get("FriendlyName"), "instance_id": device.get("InstanceId"),
                   "address": _format_address(device.get("InstanceId")), "present": device.get("Present")} for device in
                  devices]
        output = context.workspace / "bluetooth_addresses.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("bluetooth_addresses_finished", device_count=len(report),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Bluetooth names and address identifiers collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
