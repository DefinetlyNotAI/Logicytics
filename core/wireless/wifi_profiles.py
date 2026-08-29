"""Export saved Windows Wi-Fi profile names as bounded, read-only text evidence."""

from __future__ import annotations

import subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from netsh output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class WifiProfilesCollector(CoreCollector):
    """Capture saved wireless profile names without reading profile key material."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated Wi-Fi profile-name artifact contract."""
        return CollectorMetadata(
            id="core.wireless.wifi_profiles",
            name="Saved Wi-Fi profiles",
            version="4.0.0",
            specialty=Specialty.WIRELESS,
            output_media_types=("text/plain",),
            description="Exports local saved Wi-Fi profile names without retrieving key material.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("wireless_profile_names",),
            default_profiles=("deep",),
            timeout_seconds=30,
            maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and netsh availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("netsh") is None:
            return ValidationResult(False, reasons=("netsh is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """List saved Wi-Fi profiles and register their read-only text artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Wi-Fi profile collection")
        context.report_progress("wifi_profiles_started")
        completed = subprocess.run(
            ["netsh", "wlan", "show", "profiles"],
            capture_output=True,
            check=False,
            text=True,
            timeout=25,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"netsh exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "Wi-Fi profile access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "Wi-Fi profile query failed", errors=(detail,))
        output = context.workspace / "wifi_profiles.txt"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        profile_count = sum(1 for line in completed.stdout.splitlines() if " : " in line)
        context.report_progress("wifi_profiles_finished", profile_count=profile_count,
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("saved Wi-Fi profiles collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because netsh exits before the result is returned."""
