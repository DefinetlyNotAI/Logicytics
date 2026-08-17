"""Export a bounded Windows Security event-log sample as CSV evidence."""

from __future__ import annotations

import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus

_MAX_EVENTS = 1_000


def _is_access_denied(detail: str) -> bool:
    """Recognize common access-denied wording emitted by PowerShell event queries."""
    normalized = detail.casefold()
    return (
            "permission denied" in normalized
            or "unauthorized" in normalized
            or ("access" in normalized and "denied" in normalized)
    )


class SecurityEventsCollector(CoreCollector):
    """Capture a bounded CSV sample of Security events without changing event logs."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated Security-event sample and output limit."""
        return CollectorMetadata(
            id="core.event_log.security_events", name="Security event log", version="4.0.0",
            specialty=Specialty.EVENT_LOG,
            description="Exports up to 1,000 local Windows Security events as CSV.", author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("event_logs",),
            default_profiles=("deep",), timeout_seconds=90, maximum_output_bytes=8 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before querying events."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query a bounded Security-event sample and register its CSV output."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Security event-log collection")
        context.report_progress("security_events_started", maximum_events=_MAX_EVENTS)
        command = (
            f"Get-WinEvent -LogName Security -MaxEvents {_MAX_EVENTS} | "
            "Select-Object TimeCreated, Id, LevelDisplayName, ProviderName, Message | ConvertTo-Csv -NoTypeInformation"
        )
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                   capture_output=True, check=False, text=True, timeout=75)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "Security event-log access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "Security event-log query failed", errors=(detail,))
        output = context.workspace / "security_events.csv"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/csv")
        event_count = max(0, len(completed.stdout.splitlines()) - 1)
        context.report_progress("security_events_finished", event_count=event_count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Security event-log sample collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the event query process has already exited."""
