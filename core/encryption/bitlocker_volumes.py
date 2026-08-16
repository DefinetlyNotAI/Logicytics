"""Export PowerShell BitLocker volume data as bounded, read-only JSON evidence."""

from __future__ import annotations

import ctypes
import getpass
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


def _is_administrator() -> bool | None:
    """Return the local administrator token state when Windows can report it."""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except OSError:
        return None


class BitlockerVolumesCollector(CoreCollector):
    """Capture local BitLocker-volume metadata without changing encryption configuration."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated BitLocker-volume JSON artifact contract."""
        return CollectorMetadata(
            id="core.encryption.bitlocker_volumes",
            name="BitLocker volumes",
            version="4.0.0",
            specialty=Specialty.ENCRYPTION,
            description="Exports local BitLocker volume metadata through read-only Get-BitLockerVolume.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("encryption_configuration",),
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
        """Query BitLocker volumes and register their JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before BitLocker-volume collection")
        context.report_progress("bitlocker_volumes_started")
        command = (
            "$ErrorActionPreference = 'Stop'; "
            "ConvertTo-Json -InputObject @(Get-BitLockerVolume | "
            "Select-Object MountPoint, VolumeType, VolumeStatus, ProtectionStatus, EncryptionMethod, "
            "EncryptionPercentage, LockStatus, AutoUnlockEnabled) -Depth 4"
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
                return CollectorResult(CollectorStatus.SKIPPED, "BitLocker volume access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "BitLocker volume query failed", errors=(detail,))
        try:
            volumes = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "BitLocker volume query returned invalid JSON", errors=(str(error),))
        if not isinstance(volumes, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "BitLocker volume query returned an unexpected result")
        report = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "user": getpass.getuser(),
            "is_administrator": _is_administrator(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "volumes": volumes,
        }
        output = context.workspace / "bitlocker_volumes.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(volumes) if isinstance(volumes, list) else 1
        context.report_progress("bitlocker_volumes_finished", volume_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("BitLocker volumes collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
