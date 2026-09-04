"""Export a bounded Windows session and system-state snapshot as JSON evidence."""

from __future__ import annotations

import json

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class SessionSnapshotCollector(CoreCollector):
    """Capture read-only host, user, memory, locale, and system-drive metadata."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated session snapshot artifact contract."""
        return CollectorMetadata(
            id="core.system.session_snapshot",
            name="Session snapshot",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("application/json",),
            description="Exports current user/SID, OS build, memory, language, host, time, and system-drive data.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("user_identity", "system_configuration"),
            default_profiles=("deep",),
            timeout_seconds=45,
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
        """Query bounded session fields and register their JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before session-snapshot collection")
        context.report_progress("session_snapshot_started")
        command = (
            "$os = Get-CimInstance -ClassName Win32_OperatingSystem; "
            "$identity = whoami /user /fo csv /nh | ConvertFrom-Csv; "
            "[pscustomobject]@{ "
            "ComputerName = $env:COMPUTERNAME; CurrentUser = $identity.'User Name'; Sid = $identity.SID; "
            "WindowsBuild = $os.BuildNumber; Version = $os.Version; "
            "PhysicalMemoryBytes = $os.TotalVisibleMemorySize * 1KB; "
            "VirtualMemoryBytes = $os.TotalVirtualMemorySize * 1KB; "
            "FreePhysicalMemoryBytes = $os.FreePhysicalMemory * 1KB; "
            "LanguageIds = [System.Globalization.CultureInfo]::CurrentCulture.LCID; "
            "UiLanguage = [System.Globalization.CultureInfo]::CurrentUICulture.Name; "
            "CollectedAt = [DateTime]::UtcNow.ToString('o'); SystemDrive = $env:SystemDrive "
            "} | ConvertTo-Json -Depth 3"
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
                                       "session-snapshot access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "session-snapshot query failed", errors=(detail,))
        try:
            snapshot = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "session-snapshot query returned invalid JSON",
                                   errors=(str(error),))
        if not isinstance(snapshot, dict):
            return CollectorResult(CollectorStatus.FAILED, "session-snapshot query returned an unexpected result")
        output = context.workspace / "session_snapshot.json"
        output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("session_snapshot_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("session snapshot collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
