"""Export the local HKLM registry hive to a sensitive, read-only .reg artifact."""

from __future__ import annotations

import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, EvidenceKind, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize Windows permission-denied or elevation-required wording."""
    normalized = detail.casefold()
    return (
            "permission denied" in normalized
            or ("access" in normalized and "denied" in normalized)
            or "requires elevation" in normalized
    )


class HklmBackupCollector(CoreCollector):
    """Create a sensitive registry export solely inside the isolated workspace."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the explicit-consent HKLM registry-export artifact contract."""
        return CollectorMetadata(
            id="core.registry.hklm_backup", name="HKLM registry backup", version="4.0.0",
            specialty=Specialty.REGISTRY,
            output_media_types=("text/plain",),
            description="Exports the local HKLM hive as a .reg backup after explicit sensitive-data approval.",
            author="Logicytics", supported_platforms=("win32",),
            capabilities=(Capability.REGISTRY_READ, Capability.SUBPROCESS, Capability.SENSITIVE_FILES),
            sensitive_data_categories=("registry", "system_configuration", "credentials"), default_profiles=("deep",),
            timeout_seconds=300, maximum_output_bytes=2 * 1024 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and the Windows registry tool before export."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("reg") is None:
            return ValidationResult(False, reasons=("reg.exe is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Export HKLM to the private workspace and register the resulting artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before HKLM registry backup")
        context.report_progress("hklm_backup_started")
        output = context.workspace / "hklm_backup.reg"
        completed = subprocess.run(
            ["reg", "export", "HKLM", str(output), "/y"],
            capture_output=True, check=False, text=True, timeout=290,
        )
        detail = completed.stderr.strip() or completed.stdout.strip()
        if completed.returncode != 0:
            message = detail or f"reg.exe exit code {completed.returncode}"
            if _is_access_denied(message):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "HKLM registry export was denied for the current account", errors=(message,))
            return CollectorResult(CollectorStatus.FAILED, "HKLM registry export failed", errors=(message,))
        if not output.is_file() or output.stat().st_size == 0:
            return CollectorResult(CollectorStatus.FAILED, "HKLM registry export produced no backup artifact")
        artifact = context.artifacts.register_file(
            output,
            media_type="text/plain",
            evidence_kind=EvidenceKind.RAW,
            transformations=("exported from the HKLM registry hive",),
        )
        context.report_progress("hklm_backup_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("HKLM registry backup collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Leave removal to the isolated collector workspace lifecycle."""
