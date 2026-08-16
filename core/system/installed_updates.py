"""Export installed Windows hotfix metadata as bounded, read-only JSON evidence."""

from __future__ import annotations

import json
import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class InstalledUpdatesCollector(CoreCollector):
    """Capture read-only local Windows hotfix metadata through the update provider."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated installed-updates JSON artifact contract."""
        return CollectorMetadata(
            id="core.system.installed_updates", name="Installed Windows updates", version="4.0.0", specialty=Specialty.SYSTEM,
            description="Exports local installed hotfix identifiers, descriptions, and install dates.", author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,), sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",), timeout_seconds=45, maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query installed hotfixes and register their bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before installed-update collection")
        context.report_progress("installed_updates_started")
        command = "Get-HotFix | Select-Object HotFixID, Description, InstalledBy, InstalledOn | ConvertTo-Json -Depth 3"
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, check=False, text=True, timeout=40)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED, "installed-update access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "installed-update query failed", errors=(detail,))
        try:
            updates = json.loads(completed.stdout) if completed.stdout.strip() else []
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "installed-update query returned invalid JSON", errors=(str(error),))
        if not isinstance(updates, (dict, list)):
            return CollectorResult(CollectorStatus.FAILED, "installed-update query returned an unexpected result")
        output = context.workspace / "installed_updates.json"
        output.write_text(json.dumps(updates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        count = len(updates) if isinstance(updates, list) else 1
        context.report_progress("installed_updates_finished", update_count=count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("installed Windows updates collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
