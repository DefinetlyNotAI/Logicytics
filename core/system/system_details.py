"""Export the complete read-only Windows systeminfo report as an evidence artifact."""

from __future__ import annotations

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
    """Recognize common permission-denied wording from Windows command output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class SystemDetailsCollector(CoreCollector):
    """Capture the full systeminfo report without changing local system state."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated system-details artifact contract."""
        return CollectorMetadata(
            id="core.system.system_details",
            name="System details",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("text/plain",),
            description="Exports the complete Windows systeminfo report as text.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=2 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and systeminfo availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("systeminfo") is None:
            return ValidationResult(False, reasons=("systeminfo is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run systeminfo and register its complete text output when permitted."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before system-details collection")
        context.report_progress("system_details_started")
        completed = subprocess.run(
            ["systeminfo"],
            capture_output=True,
            check=False,
            text=True,
            timeout=40,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"systeminfo exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "systeminfo access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "systeminfo failed", errors=(detail,))
        output = context.workspace / "system_details.txt"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        line_count = sum(1 for line in completed.stdout.splitlines() if line.strip())
        context.report_progress("system_details_finished", line_count=line_count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("system-details report collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because systeminfo exits before the result is returned."""
