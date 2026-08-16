"""Collect a bounded CSV inventory of running Windows processes."""

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


class RunningProcessesCollector(CoreCollector):
    """Capture the non-verbose Windows Tasklist report through the artifact boundary."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the local subprocess capability and bounded CSV output contract."""
        return CollectorMetadata(
            id="core.process.running_processes",
            name="Running processes",
            version="4.0.0",
            specialty=Specialty.PROCESS,
            description="Exports the non-verbose Windows Tasklist process inventory as CSV.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            default_profiles=("standard", "deep"),
            timeout_seconds=20,
            maximum_output_bytes=2 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Verify that the Windows Tasklist command is available before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("tasklist") is None:
            return ValidationResult(False, reasons=("tasklist is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run Tasklist, write its CSV output in the private workspace, and register it."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Tasklist execution")
        context.report_progress("tasklist_started")
        completed = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"tasklist exit code {completed.returncode}"
            if "access denied" in detail.casefold() or "access is denied" in detail.casefold():
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "Tasklist access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(
                CollectorStatus.FAILED,
                "Tasklist did not complete successfully",
                errors=(detail,),
            )
        output = context.workspace / "running_processes.csv"
        output.write_text(completed.stdout, encoding="utf-8", newline="")
        artifact = context.artifacts.register_file(output, media_type="text/csv")
        process_count = sum(1 for line in completed.stdout.splitlines() if line.strip())
        context.report_progress("tasklist_finished", processes=process_count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded(
            "running-process inventory collected",
            (artifact,),
        )

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because Tasklist completes before the result is returned."""
