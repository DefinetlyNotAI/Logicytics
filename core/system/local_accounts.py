"""Export local Windows account metadata without accessing credentials or secrets."""

from __future__ import annotations

import json
from logicytics.platform_adapters import process_adapter as subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common local-account permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class LocalAccountsCollector(CoreCollector):
    """Capture read-only local-account names, enabled state, and timestamps through PowerShell."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated local-account JSON artifact contract."""
        return CollectorMetadata(
            id="core.system.local_accounts",
            name="Local accounts",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("application/json",),
            description="Exports local account names, SIDs, enabled state, descriptions, and login timestamps only.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("account_metadata",),
            default_profiles=("deep",),
            timeout_seconds=30,
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
        """Query local account metadata and register a bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before local-account collection")
        context.report_progress("local_accounts_started")
        command = (
            "Get-LocalUser | Select-Object Name, Enabled, Description, SID, LastLogon, PasswordLastSet, "
            "UserMayChangePassword | ConvertTo-Json -Depth 3"
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
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED, "local-account access was denied", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "local-account query failed", errors=(detail,))
        try:
            accounts = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "local-account query returned invalid JSON", errors=(str(error),))
        if not isinstance(accounts, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "local-account query returned an unexpected result")
        output = context.workspace / "local_accounts.json"
        output.write_text(json.dumps(accounts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(accounts) if isinstance(accounts, list) else 1
        context.report_progress("local_accounts_finished", account_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("local accounts collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
