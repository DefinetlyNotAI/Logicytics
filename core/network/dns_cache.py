"""Export the local Windows DNS resolver cache as bounded sensitive text evidence."""

from __future__ import annotations

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from ipconfig output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class DnsCacheCollector(CoreCollector):
    """Capture the local DNS resolver cache without modifying resolver state or sending traffic."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated sensitive DNS-cache artifact contract."""
        return CollectorMetadata(
            id="core.network.dns_cache", name="DNS resolver cache", version="4.0.0", specialty=Specialty.NETWORK,
            output_media_types=("text/plain",),
            description="Exports local DNS resolver cache records through read-only ipconfig output.",
            author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("dns_history",),
            default_profiles=("deep",), timeout_seconds=30, maximum_output_bytes=4 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and ipconfig availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("ipconfig") is None:
            return ValidationResult(False, reasons=("ipconfig is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Run the read-only DNS-cache query and register its bounded text artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before DNS-cache collection")
        context.report_progress("dns_cache_started")
        completed = subprocess.run(["ipconfig", "/displaydns"], capture_output=True, check=False, text=True, timeout=25)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"ipconfig exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED, "DNS-cache access was denied for the current account",
                                       errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "DNS-cache query failed", errors=(detail,))
        output = context.workspace / "dns_cache.txt"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        context.report_progress("dns_cache_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("DNS resolver cache collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because ipconfig exits before the result is returned."""
