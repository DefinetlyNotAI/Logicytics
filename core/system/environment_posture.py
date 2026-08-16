"""Export local privilege, UAC, and PowerShell-policy posture as read-only JSON."""

from __future__ import annotations

import json
import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common Windows permission-denied wording."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class EnvironmentPostureCollector(CoreCollector):
    """Capture local execution and elevation posture without changing system state."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the bounded, subprocess-gated system posture artifact contract."""
        return CollectorMetadata(
            id="core.system.environment_posture", name="Environment posture", version="4.0.0",
            specialty=Specialty.SYSTEM,
            description="Exports administrator state, UAC settings, and PowerShell execution policies.",
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
        """Query local posture and register a bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before environment-posture collection")
        context.report_progress("environment_posture_started")
        command = (
            "$ErrorActionPreference = 'Stop'; "
            "$identity = [Security.Principal.WindowsIdentity]::GetCurrent(); "
            "$principal = [Security.Principal.WindowsPrincipal]::new($identity); "
            "$uac = Get-ItemProperty -Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System' "
            "-Name EnableLUA, ConsentPromptBehaviorAdmin, PromptOnSecureDesktop; "
            "[pscustomobject]@{ IsAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator); "
            "UacEnabled = [bool]$uac.EnableLUA; ConsentPromptBehaviorAdmin = $uac.ConsentPromptBehaviorAdmin; "
            "PromptOnSecureDesktop = $uac.PromptOnSecureDesktop; "
            "ExecutionPolicies = @(try { Get-ExecutionPolicy -List | Select-Object Scope, ExecutionPolicy } "
            "catch { [pscustomobject]@{ Error = $_.Exception.Message } }); "
            "CollectedAt = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json -Depth 4"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, check=False, text=True, timeout=40,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED, "environment-posture access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "environment-posture query failed", errors=(detail,))
        try:
            posture = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "environment-posture query returned invalid JSON", errors=(str(error),))
        if not isinstance(posture, dict):
            return CollectorResult(CollectorStatus.FAILED, "environment-posture query returned an unexpected result")
        output = context.workspace / "environment_posture.json"
        output.write_text(json.dumps(posture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("environment_posture_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("environment posture collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
