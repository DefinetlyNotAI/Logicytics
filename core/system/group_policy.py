"""Collect the local Windows group-policy result report as bounded text evidence."""

from __future__ import annotations

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from Windows command output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class GroupPolicyCollector(CoreCollector):
    """Capture the local user/computer group-policy summary without changing policy state."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the read-only subprocess capability and bounded text output contract."""
        return CollectorMetadata(
            id="core.system.group_policy",
            name="Group policy result",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("text/plain",),
            description="Exports the local Windows Group Policy Result summary through gpresult.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=2 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and the presence of gpresult before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("gpresult") is None:
            return ValidationResult(False, reasons=("gpresult is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run the read-only group-policy summary and register its text output."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before group-policy collection")
        context.report_progress("group_policy_started")
        completed = subprocess.run(
            ["gpresult", "/r"],
            capture_output=True,
            check=False,
            text=True,
            timeout=40,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"gpresult exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "group-policy access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "gpresult failed", errors=(detail,))
        output = context.workspace / "group_policy.txt"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        line_count = sum(1 for line in completed.stdout.splitlines() if line.strip())
        context.report_progress("group_policy_finished", line_count=line_count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("group-policy summary collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because gpresult exits before the result is returned."""
