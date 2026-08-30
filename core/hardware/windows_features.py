"""Collect Windows optional-feature names and states through a bounded PowerShell query."""

from __future__ import annotations

import json
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which

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
    """Recognize common Windows and PowerShell permission-denied error wording."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class WindowsFeaturesCollector(CoreCollector):
    """Export Windows optional-feature state as a JSON artifact in the collector workspace."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated deep inventory and its bounded output limit."""
        return CollectorMetadata(
            id="core.hardware.windows_features",
            name="Windows optional features",
            version="4.0.0",
            specialty=Specialty.HARDWARE,
            output_media_types=("application/json",),
            description="Exports Windows optional-feature names and enabled states through PowerShell.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            default_profiles=("deep",),
            timeout_seconds=60,
            maximum_output_bytes=5 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Verify cancellation state and the PowerShell executable before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run a read-only feature query and register its JSON output when available."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before optional-feature collection")
        context.report_progress("windows_features_started")
        command = (
            "Get-WindowsOptionalFeature -Online | "
            "Select-Object FeatureName, State | ConvertTo-Json -Depth 2"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=45,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "optional-feature access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(
                CollectorStatus.FAILED,
                "PowerShell optional-feature query failed",
                errors=(detail,),
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(
                CollectorStatus.FAILED,
                "PowerShell optional-feature query returned invalid JSON",
                errors=(str(error),),
            )
        features = payload if isinstance(payload, list) else [payload]
        output = context.workspace / "windows_features.json"
        output.write_text(json.dumps(features, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("windows_features_finished", feature_count=len(features),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Windows optional features collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the PowerShell command is synchronous."""
