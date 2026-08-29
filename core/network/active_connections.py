"""Export local active Windows connections and owning PIDs as bounded text evidence."""

from __future__ import annotations

import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from netstat output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class ActiveConnectionsCollector(CoreCollector):
    """Capture active TCP and UDP endpoint metadata without network probing or changes."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated active-connection artifact contract."""
        return CollectorMetadata(
            id="core.network.active_connections",
            name="Active network connections",
            version="4.0.0",
            specialty=Specialty.NETWORK,
            output_media_types=("text/plain",),
            description="Exports active TCP/UDP endpoints, states, and owning PIDs from netstat.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("network_identifiers", "process_metadata"),
            default_profiles=("deep",),
            timeout_seconds=30,
            maximum_output_bytes=4 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and netstat availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("netstat") is None:
            return ValidationResult(False, reasons=("netstat is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run the read-only connection query and register its text evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before connection collection")
        context.report_progress("active_connections_started")
        completed = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            check=False,
            text=True,
            timeout=25,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"netstat exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED, "netstat access was denied for the current account",
                                       errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "active-connection query failed", errors=(detail,))
        output = context.workspace / "active_connections.txt"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        connection_count = sum(1 for line in completed.stdout.splitlines() if line.lstrip().startswith(("TCP", "UDP")))
        context.report_progress("active_connections_finished", connection_count=connection_count,
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("active network connections collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because netstat exits before the result is returned."""
