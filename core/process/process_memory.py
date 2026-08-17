"""Export bounded per-process memory counters as read-only JSON evidence."""

from __future__ import annotations

import json
import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class ProcessMemoryCollector(CoreCollector):
    """Capture per-process aggregate memory counters without reading memory contents."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated process-memory JSON artifact contract."""
        return CollectorMetadata(
            id="core.process.process_memory", name="Process memory", version="4.0.0", specialty=Specialty.PROCESS,
            description="Exports aggregate working-set, private, and virtual memory counters for local processes.",
            author="Logicytics", supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("process_metadata",), default_profiles=("deep",), timeout_seconds=60,
            maximum_output_bytes=4 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query aggregate process memory counters and register a JSON artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before process-memory collection")
        context.report_progress("process_memory_started")
        command = (
            "$ErrorActionPreference = 'Continue'; Get-Process | "
            "Select-Object Id, ProcessName, WorkingSet64, PrivateMemorySize64, VirtualMemorySize64, HandleCount, CPU, StartTime | "
            "ConvertTo-Json -Depth 3"
        )
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                   capture_output=True, check=False, text=True, timeout=55)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "process-memory access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "process-memory query failed", errors=(detail,))
        try:
            processes = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "process-memory query returned invalid JSON",
                                   errors=(str(error),))
        if not isinstance(processes, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "process-memory query returned an unexpected result")
        output = context.workspace / "process_memory.json"
        output.write_text(json.dumps(processes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(processes) if isinstance(processes, list) else 1
        context.report_progress("process_memory_finished", process_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("process memory collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
