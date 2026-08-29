"""Export standard Windows startup Run-key entries as bounded read-only JSON evidence."""

from __future__ import annotations

import json
import winreg
from datetime import datetime, timezone

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus

_RUN_PATHS = (
    ("HKEY_CURRENT_USER", winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
    ("HKEY_CURRENT_USER", winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
    ("HKEY_LOCAL_MACHINE", winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
    ("HKEY_LOCAL_MACHINE", winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
)


def _startup_entries() -> list[dict[str, str]]:
    """Enumerate values from conventional per-user and machine startup Run keys."""
    entries: list[dict[str, str]] = []
    for hive_name, hive, path in _RUN_PATHS:
        try:
            key = winreg.OpenKey(hive, path)
        except OSError:
            continue
        with key:
            index = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, index)
                except OSError:
                    break
                index += 1
                entries.append({"hive": hive_name, "path": path, "name": name, "command": str(value)})
    return sorted(entries, key=lambda item: (item["hive"], item["path"], item["name"].casefold()))


class StartupApplicationsCollector(CoreCollector):
    """Capture startup Run-key commands without launching or modifying any application."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the registry-read startup-application artifact contract."""
        return CollectorMetadata(
            id="core.registry.startup_applications", name="Startup applications", version="4.0.0",
            specialty=Specialty.REGISTRY,
            output_media_types=("application/json",),
            description="Exports standard user and machine Run/RunOnce startup registry entries.", author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.REGISTRY_READ,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",), timeout_seconds=30, maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation before reading potentially available startup key paths."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Read startup metadata and register a bounded JSON evidence artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before startup-application collection")
        context.report_progress("startup_applications_started")
        entries = _startup_entries()
        output = context.workspace / "startup_applications.json"
        output.write_text(
            json.dumps({"collected_at": datetime.now(timezone.utc).isoformat(), "entries": entries}, indent=2,
                       sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("startup_applications_finished", entry_count=len(entries),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("startup applications collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because all registry handles are scoped and closed."""
