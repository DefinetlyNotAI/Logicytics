"""Export mounted Windows volume GUID mappings as bounded, read-only text evidence."""

from __future__ import annotations

import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from mountvol output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class MountedVolumesCollector(CoreCollector):
    """Capture mounted volume GUID mappings without changing any mount points."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated mounted-volume text artifact contract."""
        return CollectorMetadata(
            id="core.storage.mounted_volumes",
            name="Mounted volumes",
            version="4.0.0",
            specialty=Specialty.STORAGE,
            description="Exports Windows mounted volume GUID and mount-point mappings from mountvol.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",),
            timeout_seconds=30,
            maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and mountvol availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("mountvol") is None:
            return ValidationResult(False, reasons=("mountvol is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run read-only mountvol and register its bounded text evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before mounted-volume collection")
        context.report_progress("mounted_volumes_started")
        completed = subprocess.run(["mountvol"], capture_output=True, check=False, text=True, timeout=25)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"mountvol exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED, "mounted-volume access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "mounted-volume query failed", errors=(detail,))
        output = context.workspace / "mounted_volumes.txt"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        volume_count = sum(1 for line in completed.stdout.splitlines() if line.strip().startswith("\\\\?\\Volume{"))
        context.report_progress("mounted_volumes_finished", volume_count=volume_count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("mounted-volume mappings collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because mountvol exits before the result is returned."""
