"""Copy bounded browser-profile evidence after explicit browser-data approval."""

from __future__ import annotations

from pathlib import Path

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, EvidenceKind, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import filesystem_adapter

CHROMIUM_FILES = ("History", "History-journal", "Cookies", "Cookies-journal", "Login Data", "Login Data-journal",
                  "Bookmarks", "Preferences")
FIREFOX_FILES = ("places.sqlite", "cookies.sqlite", "logins.json", "key4.db", "prefs.js")
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024


def _copy_file(source: Path, destination: Path, copied_bytes: int) -> tuple[Path | None, int]:
    """Copy one bounded regular file, preserving metadata, or return no artifact."""
    try:
        if not source.is_file() or source.is_symlink():
            return None, copied_bytes
        size = source.stat().st_size
    except OSError:
        return None, copied_bytes
    if size > MAX_FILE_BYTES or copied_bytes + size > MAX_TOTAL_BYTES:
        return None, copied_bytes
    destination.parent.mkdir(parents=True, exist_ok=True)
    filesystem_adapter.copy_file(source, destination)
    return destination, copied_bytes + size


class BrowserDataBackupCollector(CoreCollector):
    """Copy bounded data from supported local browser profiles into a private workspace."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the explicit-consent browser-data artifact contract."""
        return CollectorMetadata(
            id="core.browser.browser_data_backup", name="Browser data backup", version="4.0.0",
            specialty=Specialty.BROWSER,
            output_media_types=("application/octet-stream",),
            description="Copies bounded local profile evidence from Edge, Chrome, Firefox, Opera, and Opera GX.",
            author="Logicytics", supported_platforms=("win32",),
            capabilities=(Capability.FILESYSTEM_READ, Capability.BROWSER_DATA, Capability.SENSITIVE_FILES),
            sensitive_data_categories=("browser_history", "cookies", "credentials"), default_profiles=("deep",),
            timeout_seconds=300, maximum_output_bytes=256 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state before browser-profile copying starts."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Copy supported profile files from configured local browser locations."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before browser data backup")
        home = filesystem_adapter.home()
        local = filesystem_adapter.environment_path("LOCALAPPDATA", home / "AppData" / "Local")
        roaming = filesystem_adapter.environment_path("APPDATA", home / "AppData" / "Roaming")
        chromium_roots = (("chrome", local / "Google" / "Chrome" / "User Data"),
                          ("edge", local / "Microsoft" / "Edge" / "User Data"),
                          ("opera", roaming / "Opera Software" / "Opera Stable"),
                          ("opera_gx", roaming / "Opera Software" / "Opera GX Stable"))
        copied: list[Path] = []
        copied_bytes = 0
        context.report_progress("browser_data_backup_started")
        for browser, root in chromium_roots:
            try:
                profile_roots = [root] if browser.startswith("opera") else [path for path in filesystem_adapter.children(root) if
                                                                            path.is_dir() and (
                                                                                        path.name == "Default" or path.name.startswith(
                                                                                    "Profile "))]
            except OSError:
                continue
            for profile in profile_roots:
                for filename in CHROMIUM_FILES:
                    if context.is_cancelled:
                        return CollectorResult(CollectorStatus.CANCELLED, "cancelled during browser data backup")
                    destination, copied_bytes = _copy_file(profile / filename,
                                                           context.workspace / "browser_data" / browser / profile.name / filename,
                                                           copied_bytes)
                    if context.is_cancelled:
                        if destination is not None:
                            destination.unlink(missing_ok=True)
                        for copied_path in copied:
                            copied_path.unlink(missing_ok=True)
                        return CollectorResult(CollectorStatus.CANCELLED, "cancelled during browser data backup")
                    if destination is not None:
                        copied.append(destination)
        firefox_profiles = roaming / "Mozilla" / "Firefox" / "Profiles"
        try:
            profiles = [path for path in filesystem_adapter.children(firefox_profiles) if path.is_dir()]
        except OSError:
            profiles = []
        for profile in profiles:
            for filename in FIREFOX_FILES:
                if context.is_cancelled:
                    for copied_path in copied:
                        copied_path.unlink(missing_ok=True)
                    return CollectorResult(CollectorStatus.CANCELLED, "cancelled during browser data backup")
                destination, copied_bytes = _copy_file(profile / filename,
                                                       context.workspace / "browser_data" / "firefox" / profile.name / filename,
                                                       copied_bytes)
                if context.is_cancelled:
                    if destination is not None:
                        destination.unlink(missing_ok=True)
                    for copied_path in copied:
                        copied_path.unlink(missing_ok=True)
                    return CollectorResult(CollectorStatus.CANCELLED, "cancelled during browser data backup")
                if destination is not None:
                    copied.append(destination)
        if not copied:
            return CollectorResult(CollectorStatus.SKIPPED,
                                   "no supported local browser profile data met the bounded backup policy")
        artifacts = []
        for path in copied:
            if context.is_cancelled:
                for unpublished in copied[len(artifacts):]:
                    unpublished.unlink(missing_ok=True)
                return CollectorResult.cancelled("cancelled during browser-data registration", tuple(artifacts))
            artifacts.append(context.artifacts.register_file(path, evidence_kind=EvidenceKind.RAW))
        artifact_tuple = tuple(artifacts)
        context.report_progress("browser_data_backup_finished", copied_files=len(artifact_tuple),
                                bytes_written=sum(item.size_bytes for item in artifact_tuple))
        return CollectorResult.succeeded("browser data backup collected", artifact_tuple)

    def cleanup(self, context: CollectorContext) -> None:
        """Leave copied evidence removal to the isolated workspace lifecycle."""
