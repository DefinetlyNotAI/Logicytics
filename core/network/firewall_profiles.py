"""Export local Windows firewall profile settings as bounded read-only JSON evidence."""

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


class FirewallProfilesCollector(CoreCollector):
    """Capture read-only local firewall profile configuration without changing firewall state."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated firewall-profile JSON artifact contract."""
        return CollectorMetadata(
            id="core.network.firewall_profiles",
            name="Firewall profiles",
            version="4.0.0",
            specialty=Specialty.NETWORK,
            output_media_types=("application/json",),
            description="Exports local Domain, Private, and Public Windows firewall profile settings.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=256 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query firewall profiles and register their bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before firewall-profile collection")
        context.report_progress("firewall_profiles_started")
        command = (
            "Get-NetFirewallProfile | "
            "Select-Object Name, Enabled, DefaultInboundAction, DefaultOutboundAction, "
            "NotifyOnListen, AllowInboundRules, AllowLocalFirewallRules, "
            "AllowLocalIPsecRules, LogFileName | "
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
                    "firewall-profile access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "firewall-profile query failed", errors=(detail,))
        try:
            profiles = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(
                CollectorStatus.FAILED,
                "firewall-profile query returned invalid JSON",
                errors=(str(error),),
            )
        if not isinstance(profiles, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "firewall-profile query returned an unexpected result")
        output = context.workspace / "firewall_profiles.json"
        output.write_text(json.dumps(profiles, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(profiles) if isinstance(profiles, list) else 1
        context.report_progress("firewall_profiles_finished", profile_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("firewall profiles collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
