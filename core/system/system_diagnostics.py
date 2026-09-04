"""Export Windows hardware, CPU, page-size, and boot-time diagnostics as bounded JSON."""

from __future__ import annotations

import json

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class SystemDiagnosticsCollector(CoreCollector):
    """Capture read-only CPU, architecture, page-size, and boot-time diagnostics."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated system-diagnostics JSON artifact contract."""
        return CollectorMetadata(
            id="core.system.system_diagnostics", name="System diagnostics", version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("application/json",),
            description="Exports architecture, CPU, page-size, and boot-time diagnostics through Windows CIM.",
            author="Logicytics", supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",), default_profiles=("deep",),
            timeout_seconds=45, maximum_output_bytes=256 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query diagnostics and register their bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before system-diagnostics collection")
        context.report_progress("system_diagnostics_started")
        command = (
            "$ErrorActionPreference = 'Stop'; $os = Get-CimInstance -ClassName Win32_OperatingSystem; "
            "$cpu = @(Get-CimInstance -ClassName Win32_Processor | Select-Object Name, Architecture, AddressWidth, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed); "
            "[pscustomobject]@{ OperatingSystem = $os.Caption; Version = $os.Version; Architecture = $os.OSArchitecture; "
            "Machine = $env:PROCESSOR_IDENTIFIER; PageSizeBytes = [Environment]::SystemPageSize; ProcessorCount = $cpu.Count; "
            "Processors = $cpu; LastBootUpTime = $os.LastBootUpTime; CollectedAt = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json -Depth 5"
        )
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                   capture_output=True, check=False, text=True, timeout=40)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "system-diagnostics access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "system-diagnostics query failed", errors=(detail,))
        try:
            diagnostics = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "system-diagnostics query returned invalid JSON",
                                   errors=(str(error),))
        if not isinstance(diagnostics, dict):
            return CollectorResult(CollectorStatus.FAILED, "system-diagnostics query returned an unexpected result")
        output = context.workspace / "system_diagnostics.json"
        output.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("system_diagnostics_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("system diagnostics collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
