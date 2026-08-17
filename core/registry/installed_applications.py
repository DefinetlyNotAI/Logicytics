"""Export installed Windows application metadata from read-only uninstall registry keys."""

from __future__ import annotations

import json
import winreg
from datetime import datetime, timezone

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus

_UNINSTALL_PATHS = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
    r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
)
_MAX_APPLICATIONS = 5_000


def _registry_value(key: winreg.HKEYType, name: str) -> str | None:
    """Return a string registry value when present without exposing registry errors."""
    try:
        value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return str(value) if value is not None else None


def _installed_applications() -> list[dict[str, str | None]]:
    """Enumerate uninstall metadata from 64-bit and WOW6432Node registry views."""
    applications: list[dict[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for path in _UNINSTALL_PATHS:
        try:
            root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
        except OSError:
            continue
        with root:
            index = 0
            while len(applications) < _MAX_APPLICATIONS:
                try:
                    key_name = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                try:
                    with winreg.OpenKey(root, key_name) as key:
                        display_name = _registry_value(key, "DisplayName")
                        if not display_name:
                            continue
                        version = _registry_value(key, "DisplayVersion") or ""
                        identity = (display_name, version)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        applications.append({"display_name": display_name, "display_version": version or None,
                                             "publisher": _registry_value(key, "Publisher"),
                                             "install_date": _registry_value(key, "InstallDate"),
                                             "install_location": _registry_value(key, "InstallLocation"),
                                             "uninstall_key": key_name})
                except OSError:
                    continue
    return sorted(applications, key=lambda item: (item["display_name"] or "").casefold())


class InstalledApplicationsCollector(CoreCollector):
    """Capture read-only installed-application metadata from Windows registry views."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the registry-read, bounded installed-application artifact contract."""
        return CollectorMetadata(
            id="core.registry.installed_applications", name="Installed applications", version="4.0.0",
            specialty=Specialty.REGISTRY,
            description="Exports installed application names, versions, publishers, and install metadata from uninstall keys.",
            author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.REGISTRY_READ,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",), timeout_seconds=60, maximum_output_bytes=4 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation and whether at least one uninstall registry view is readable."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        for path in _UNINSTALL_PATHS:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path):
                    return ValidationResult(True)
            except OSError:
                continue
        return ValidationResult(False, reasons=("Windows uninstall registry keys are unavailable",))

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Read installed application metadata and register the bounded JSON artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before installed-application collection")
        context.report_progress("installed_applications_started")
        applications = _installed_applications()
        output = context.workspace / "installed_applications.json"
        output.write_text(
            json.dumps({"collected_at": datetime.now(timezone.utc).isoformat(), "applications": applications}, indent=2,
                       sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("installed_applications_finished", application_count=len(applications),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("installed applications collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because all registry handles are scoped and closed."""
