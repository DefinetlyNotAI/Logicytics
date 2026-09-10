"""Export local Microsoft Defender protection status as read-only JSON evidence."""

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


def _is_unavailable(detail: str) -> bool:
    """Recognize missing Defender-provider and access-denied results without failing a run."""
    normalized = detail.casefold()
    return (
            "permission denied" in normalized
            or ("access" in normalized and "denied" in normalized)
            or "not recognized" in normalized
            or "cannot find" in normalized
    )


class DefenderStatusCollector(CoreCollector):
    """Capture read-only Microsoft Defender configuration and protection-state metadata."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated Defender-status JSON artifact contract."""
        return CollectorMetadata(
            id="core.system.defender_status",
            name="Microsoft Defender status",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("application/json",),
            description="Exports installed Microsoft Defender engine, signature, and protection-state metadata.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("security_configuration",),
            default_profiles=("standard", "deep"),
            timeout_seconds=30,
            maximum_output_bytes=128 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query Defender status and register its bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Defender-status collection")
        context.report_progress("defender_status_started")
        command = (
            "Get-MpComputerStatus | Select-Object AMEngineVersion, AMProductVersion, AMServiceEnabled, "
            "AntispywareEnabled, AntivirusEnabled, BehaviorMonitorEnabled, IoavProtectionEnabled, "
            "NISEnabled, RealTimeProtectionEnabled, AntivirusSignatureVersion, AntivirusSignatureLastUpdated, "
            "QuickScanAge, FullScanAge | ConvertTo-Json -Depth 3"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=25,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_unavailable(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "Microsoft Defender status is unavailable",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "Microsoft Defender status query failed", errors=(detail,))
        try:
            status = json.loads(completed.stdout) if completed.stdout.strip() else {}
        except json.JSONDecodeError as error:
            return CollectorResult(
                CollectorStatus.FAILED,
                "Defender status query returned invalid JSON",
                errors=(str(error),),
            )
        if not isinstance(status, dict):
            return CollectorResult(CollectorStatus.FAILED, "Defender status query returned an unexpected result")
        output = context.workspace / "defender_status.json"
        output.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("defender_status_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Microsoft Defender status collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
