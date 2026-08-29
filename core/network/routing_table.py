"""Export the local Windows routing table as bounded, read-only network evidence."""

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


class RoutingTableCollector(CoreCollector):
    """Capture the local routing table without changing routes or sending traffic."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated, bounded routing-table report."""
        return CollectorMetadata(
            id="core.network.routing_table",
            name="Routing table",
            version="4.0.0",
            specialty=Specialty.NETWORK,
            output_media_types=("text/plain",),
            description="Exports the local IPv4 and IPv6 routing tables without changing routes.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("network_identifiers",),
            default_profiles=("deep",),
            timeout_seconds=30,
            maximum_output_bytes=1 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and route availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("route") is None:
            return ValidationResult(False, reasons=("route is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run the read-only route query and register its text artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before routing-table collection")
        context.report_progress("routing_table_started")
        completed = subprocess.run(
            ["route", "print"],
            capture_output=True,
            check=False,
            text=True,
            timeout=25,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"route exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "routing-table access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "routing-table query failed", errors=(detail,))
        output = context.workspace / "routing_table.txt"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        line_count = sum(1 for line in completed.stdout.splitlines() if line.strip())
        context.report_progress("routing_table_finished", line_count=line_count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("routing-table report collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because route exits before the result is returned."""
