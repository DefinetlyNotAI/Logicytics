"""Collect a bounded CSV inventory of installed Windows device drivers."""

from __future__ import annotations

from logicytics.platform_adapters import process_adapter as subprocess
from shutil import which

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
    """Recognize common access-denied wording from Windows command output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class InstalledDriversCollector(CoreCollector):
    """Capture a read-only driverquery CSV report through the artifact boundary."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the read-only subprocess capability and bounded driver-report output."""
        return CollectorMetadata(
            id="core.system.installed_drivers",
            name="Installed drivers",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("text/csv",),
            description="Exports detailed Windows driver inventory information as CSV.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            default_profiles=("deep",),
            timeout_seconds=30,
            maximum_output_bytes=4 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Verify cancellation state and driverquery availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("driverquery") is None:
            return ValidationResult(False, reasons=("driverquery is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run driverquery and register its detailed CSV report when permitted."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before driver inventory collection")
        context.report_progress("installed_drivers_started")
        completed = subprocess.run(
            ["driverquery", "/v", "/fo", "csv", "/nh"],
            capture_output=True,
            check=False,
            text=True,
            timeout=25,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"driverquery exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "driverquery access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "driverquery failed", errors=(detail,))
        output = context.workspace / "installed_drivers.csv"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/csv")
        driver_count = sum(1 for line in completed.stdout.splitlines() if line.strip())
        context.report_progress("installed_drivers_finished", driver_count=driver_count,
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("installed-driver inventory collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because driverquery has completed before return."""
