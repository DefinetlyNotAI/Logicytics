"""Export bounded local Scheduled Task metadata as read-only JSON evidence."""

from __future__ import annotations

import json
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class ScheduledTasksCollector(CoreCollector):
    """Capture read-only local scheduled-task identities and current states."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated scheduled-task JSON artifact contract."""
        return CollectorMetadata(
            id="core.system.scheduled_tasks", name="Scheduled tasks", version="4.0.0", specialty=Specialty.SYSTEM,
            output_media_types=("application/json",),
            description="Exports up to 1,000 local scheduled-task names, paths, authors, descriptions, and states.",
            author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",), timeout_seconds=60, maximum_output_bytes=2 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query bounded scheduled-task metadata and register a JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before scheduled-task collection")
        context.report_progress("scheduled_tasks_started")
        command = (
            "Get-ScheduledTask | Select-Object -First 1000 TaskName, TaskPath, State, Author, Description, URI | "
            "ConvertTo-Json -Depth 3"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=55,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "scheduled-task access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "scheduled-task query failed", errors=(detail,))
        try:
            tasks = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(
                CollectorStatus.FAILED,
                "scheduled-task query returned invalid JSON",
                errors=(str(error),),
            )
        if not isinstance(tasks, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "scheduled-task query returned an unexpected result")
        output = context.workspace / "scheduled_tasks.json"
        output.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(tasks) if isinstance(tasks, list) else 1
        context.report_progress("scheduled_tasks_finished", task_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("scheduled tasks collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
