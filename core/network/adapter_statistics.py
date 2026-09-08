"""Export local Windows network adapter I/O counters as bounded JSON evidence."""

from __future__ import annotations

import json

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
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class AdapterStatisticsCollector(CoreCollector):
    """Capture read-only per-interface byte, packet, error, and discard counters."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated adapter-statistics artifact contract."""
        return CollectorMetadata(
            id="core.network.adapter_statistics",
            name="Network adapter statistics",
            version="4.0.0",
            specialty=Specialty.NETWORK,
            output_media_types=("application/json",),
            description="Exports per-interface network byte, packet, error, and discard counters.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("network_identifiers",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query adapter counters and register their JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before adapter-statistics collection")
        context.report_progress("adapter_statistics_started")
        command = (
            "Get-NetAdapterStatistics | "
            "Select-Object Name, ReceivedBytes, SentBytes, ReceivedUnicastPackets, SentUnicastPackets, "
            "ReceivedDiscardedPackets, OutboundDiscardedPackets, ReceivedPacketErrors, OutboundPacketErrors | "
            "ConvertTo-Json -Depth 3"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=40,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "adapter-statistics access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "adapter-statistics query failed", errors=(detail,))
        try:
            statistics = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(
                CollectorStatus.FAILED,
                "adapter-statistics query returned invalid JSON",
                errors=(str(error),),
            )
        if not isinstance(statistics, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "adapter-statistics query returned an unexpected result")
        output = context.workspace / "adapter_statistics.json"
        output.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(statistics) if isinstance(statistics, list) else 1
        context.report_progress("adapter_statistics_finished", interface_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("network adapter statistics collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
