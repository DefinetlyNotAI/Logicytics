"""Export saved Windows Wi-Fi profiles with key material after explicit approval."""

from __future__ import annotations

from logicytics.platform_adapters import process_adapter as subprocess
from pathlib import Path
from logicytics.platform_adapters import filesystem_adapter, which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, EvidenceKind, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common Windows permission-denied or elevation-required wording."""
    normalized = detail.casefold()
    return (
            "permission denied" in normalized
            or ("access" in normalized and "denied" in normalized)
            or "requires elevation" in normalized
    )


class WifiProfileKeysCollector(CoreCollector):
    """Export saved Wi-Fi profiles and credentials only after sensitive approval."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the explicit-consent wireless credential export contract."""
        return CollectorMetadata(
            id="core.wireless.wifi_profile_keys", name="Saved Wi-Fi profile keys", version="4.0.0",
            specialty=Specialty.WIRELESS,
            output_media_types=("application/xml",),
            description="Exports saved Wi-Fi profile XML including key material after explicit sensitive-data approval.",
            author="Logicytics", supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS, Capability.SENSITIVE_FILES),
            sensitive_data_categories=("wireless_profile_names", "credentials"), default_profiles=("deep",),
            timeout_seconds=60, maximum_output_bytes=2 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and netsh availability before credential export."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("netsh") is None:
            return ValidationResult(False, reasons=("netsh is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Export profile XML into this collector's private workspace and register it."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Wi-Fi profile-key collection")
        context.report_progress("wifi_profile_keys_started")
        export_directory = context.workspace / "wifi_profiles_with_keys"
        export_directory.mkdir()
        completed = subprocess.run(
            ["netsh", "wlan", "export", "profile", f"folder={export_directory}", "key=clear"],
            capture_output=True, check=False, text=True, timeout=55,
        )
        detail = completed.stderr.strip() or completed.stdout.strip()
        if completed.returncode != 0:
            message = detail or f"netsh exit code {completed.returncode}"
            if _is_access_denied(message):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "Wi-Fi profile-key access was denied for the current account", errors=(message,))
            return CollectorResult(CollectorStatus.FAILED, "Wi-Fi profile-key export failed", errors=(message,))
        profiles = sorted(path for path in filesystem_adapter.glob(export_directory, "*.xml") if path.is_file())
        if not profiles:
            return CollectorResult(CollectorStatus.SKIPPED, "no saved Wi-Fi profiles with key material were exported")
        artifacts = tuple(
            context.artifacts.register_file(
                Path(profile), media_type="application/xml", evidence_kind=EvidenceKind.RAW
            )
            for profile in profiles
        )
        context.report_progress("wifi_profile_keys_finished", profile_count=len(artifacts),
                                bytes_written=sum(item.size_bytes for item in artifacts))
        return CollectorResult.succeeded("saved Wi-Fi profile keys collected", artifacts)

    def cleanup(self, context: CollectorContext) -> None:
        """Leave cleanup to the isolated collector workspace lifecycle."""
