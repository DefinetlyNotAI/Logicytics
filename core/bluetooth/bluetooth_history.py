"""Write a timestamped Bluetooth-device snapshot for retained run-history evidence."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common Windows permission-denied wording."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class BluetoothHistoryCollector(CoreCollector):
    """Export a timestamped Bluetooth snapshot without mutating shared history state."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the bounded Bluetooth-history JSON artifact contract."""
        return CollectorMetadata(
            id="core.bluetooth.bluetooth_history", name="Bluetooth history snapshot", version="4.0.0",
            specialty=Specialty.BLUETOOTH,
            output_media_types=("application/json",),
            description="Exports a timestamped Bluetooth PnP snapshot; retained run packages provide historical evidence.",
            author="Logicytics", supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("device_identifiers",), default_profiles=("deep",), timeout_seconds=45,
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
        """Query Bluetooth PnP state and register a timestamped JSON snapshot."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Bluetooth-history collection")
        context.report_progress("bluetooth_history_started")
        command = "$ErrorActionPreference = 'Stop'; @(Get-PnpDevice -Class Bluetooth | Select-Object FriendlyName, InstanceId, Status, Problem, Present) | ConvertTo-Json -Depth 3"
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                   capture_output=True, check=False, text=True, timeout=40)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "Bluetooth-history access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "Bluetooth-history query failed", errors=(detail,))
        try:
            devices = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "Bluetooth-history query returned invalid JSON",
                                   errors=(str(error),))
        snapshot = {"collected_at": datetime.now(timezone.utc).isoformat(),
                    "devices": devices if isinstance(devices, list) else [devices]}
        output = context.workspace / "bluetooth_history.json"
        output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("bluetooth_history_finished", device_count=len(snapshot["devices"]),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Bluetooth history snapshot collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Leave temporary snapshot removal to the isolated workspace lifecycle."""
