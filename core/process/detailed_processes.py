"""Export the verbose Windows task list as bounded CSV process evidence."""

from __future__ import annotations

import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from tasklist output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class DetailedProcessesCollector(CoreCollector):
    """Capture a read-only verbose tasklist CSV report without changing processes."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated detailed process-report contract."""
        return CollectorMetadata(
            id="core.process.detailed_processes",
            name="Detailed running processes",
            version="4.0.0",
            specialty=Specialty.PROCESS,
            output_media_types=("text/csv",),
            description="Exports the verbose local Windows task list as CSV.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("process_metadata",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=8 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and tasklist availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("tasklist") is None:
            return ValidationResult(False, reasons=("tasklist is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run the verbose task list and register its bounded CSV evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before detailed-process collection")
        context.report_progress("detailed_processes_started")
        completed = subprocess.run(
            ["tasklist", "/v", "/fo", "csv", "/nh"],
            capture_output=True,
            check=False,
            text=True,
            timeout=40,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"tasklist exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED, "tasklist access was denied for the current account",
                                       errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "detailed tasklist query failed", errors=(detail,))
        output = context.workspace / "detailed_processes.csv"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/csv")
        process_count = sum(1 for line in completed.stdout.splitlines() if line.strip())
        context.report_progress("detailed_processes_finished", process_count=process_count,
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("detailed process list collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because tasklist exits before the result is returned."""
