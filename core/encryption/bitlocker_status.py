"""Export local BitLocker status as bounded, read-only text evidence."""

from __future__ import annotations

from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str, return_code: int) -> bool:
    """Recognize textual and HRESULT access-denied results from manage-bde."""
    normalized = detail.casefold()
    return (
            return_code == 2147749891
            or "permission denied" in normalized
            or ("access" in normalized and "denied" in normalized)
    )


class BitlockerStatusCollector(CoreCollector):
    """Capture local BitLocker status without changing protectors or encryption state."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated BitLocker-status artifact contract."""
        return CollectorMetadata(
            id="core.encryption.bitlocker_status",
            name="BitLocker status",
            version="4.0.0",
            specialty=Specialty.ENCRYPTION,
            output_media_types=("text/plain",),
            description="Exports local drive BitLocker status through the read-only manage-bde command.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("encryption_configuration",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=2 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and manage-bde availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("manage-bde") is None:
            return ValidationResult(False, reasons=("manage-bde is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run read-only BitLocker status and register its text evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before BitLocker-status collection")
        context.report_progress("bitlocker_status_started")
        completed = subprocess.run(["manage-bde", "-status"], capture_output=True, check=False, text=True, timeout=40)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"manage-bde exit code {completed.returncode}"
            if _is_access_denied(detail, completed.returncode):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "BitLocker status access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "BitLocker status query failed", errors=(detail,))
        output = context.workspace / "bitlocker_status.txt"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        context.report_progress("bitlocker_status_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("BitLocker status collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because manage-bde exits before the result is returned."""
