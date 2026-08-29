"""Export a bounded Windows Application event-log sample as CSV evidence."""

from __future__ import annotations

import subprocess
from shutil import which

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    ResourceClass,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus

_MAX_EVENTS = 1_000


def _is_access_denied(detail: str) -> bool:
    """Recognize common access-denied wording emitted by PowerShell event queries."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class ApplicationEventsCollector(CoreCollector):
    """Capture a bounded CSV sample of Application events without changing event logs."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated Application-event sample and output limit."""
        return CollectorMetadata(
            id="core.event_log.application_events",
            name="Application event log",
            version="4.0.0",
            specialty=Specialty.EVENT_LOG,
            output_media_types=("text/csv",),
            description="Exports up to 1,000 local Windows Application events as CSV.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("event_logs",),
            parallel_safe=True,
            resource_class=ResourceClass.GENERAL,
            default_profiles=("deep",),
            timeout_seconds=90,
            maximum_output_bytes=8 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before querying events."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query a bounded Application-event sample and register its CSV output."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Application event-log collection")
        context.report_progress("application_events_started", maximum_events=_MAX_EVENTS)
        command = (
            f"Get-WinEvent -LogName Application -MaxEvents {_MAX_EVENTS} | "
            "Select-Object TimeCreated, Id, LevelDisplayName, ProviderName, Message | "
            "ConvertTo-Csv -NoTypeInformation"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=75,
        )
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled during Application event-log query")
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "Application event-log access was denied for the current account",
                                       errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "Application event-log query failed", errors=(detail,))
        output = context.workspace / "application_events.csv"
        output.write_text(completed.stdout, encoding="utf-8")
        if context.is_cancelled:
            output.unlink(missing_ok=True)
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled during Application event-log export")
        artifact = context.artifacts.register_file(output, media_type="text/csv")
        event_count = max(0, len(completed.stdout.splitlines()) - 1)
        context.report_progress("application_events_finished", event_count=event_count,
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Application event-log sample collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the event query process has already exited."""
