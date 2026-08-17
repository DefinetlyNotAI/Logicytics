"""Create a bounded threaded recursive listing of the Windows system drive."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus

_DEFAULT_MAX_ENTRIES = 10_000
_DEFAULT_MAX_DEPTH = 16
_DEFAULT_WORKERS = 4
_HARD_MAX_ENTRIES = 50_000
_HARD_MAX_DEPTH = 32
_HARD_MAX_WORKERS = 8


def _setting(settings: object, key: str, default: int, maximum: int) -> int:
    """Return one positive integer setting constrained to its documented maximum."""
    value = settings.get(key, default) if isinstance(settings, dict) else default
    return value if isinstance(value, int) and 1 <= value <= maximum else default


def _scan_directory(directory: Path) -> tuple[list[str], list[Path]]:
    """Return metadata-only entries and non-link child directories for one path."""
    entries: list[str] = []
    children: list[Path] = []
    try:
        with os.scandir(directory) as scan:
            for entry in scan:
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                entries.append(f"{entry.path}{'/' if is_directory else ''}")
                if is_directory:
                    children.append(Path(entry.path))
    except OSError:
        return [], []
    return sorted(entries), sorted(children)


class SystemDriveListingCollector(CoreCollector):
    """Capture a bounded threaded recursive directory listing without reading file contents."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the filesystem-read, threaded system-drive listing contract."""
        return CollectorMetadata(
            id="core.filesystem.system_drive_listing", name="System drive listing", version="4.0.0",
            specialty=Specialty.FILESYSTEM,
            description="Exports a configurable, bounded threaded recursive listing of the Windows system drive.",
            author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.FILESYSTEM_READ,),
            sensitive_data_categories=("filesystem_metadata",), default_profiles=("deep",), timeout_seconds=180,
            maximum_output_bytes=8 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and system-drive availability before traversal."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        root = Path(os.environ.get("SystemDrive", "C:") + "\\")
        if not root.is_dir():
            return ValidationResult(False, reasons=(f"system drive is unavailable: {root}",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Traverse the system drive in bounded parallel batches and register text evidence."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before system-drive listing collection")
        maximum_entries = _setting(context.settings, "max_entries", _DEFAULT_MAX_ENTRIES, _HARD_MAX_ENTRIES)
        maximum_depth = _setting(context.settings, "max_depth", _DEFAULT_MAX_DEPTH, _HARD_MAX_DEPTH)
        workers = _setting(context.settings, "workers", _DEFAULT_WORKERS, _HARD_MAX_WORKERS)
        root = Path(os.environ.get("SystemDrive", "C:") + "\\")
        lines = [f"# system_drive={root}", f"# max_entries={maximum_entries}", f"# max_depth={maximum_depth}",
                 f"# workers={workers}"]
        pending: dict[Future[tuple[list[str], list[Path]]], tuple[Path, int]] = {}
        entries = 0
        truncated = False
        context.report_progress("system_drive_listing_started", max_entries=maximum_entries, max_depth=maximum_depth,
                                workers=workers)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="LogicyticsList") as executor:
            pending[executor.submit(_scan_directory, root)] = (root, 0)
            while pending and entries < maximum_entries:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    directory, depth = pending.pop(future)
                    names, children = future.result()
                    for name in names:
                        if entries >= maximum_entries:
                            truncated = True
                            break
                        lines.append(str(Path(name).relative_to(root)))
                        entries += 1
                    if truncated:
                        break
                    if depth < maximum_depth:
                        for child in children:
                            if len(pending) >= maximum_entries:
                                truncated = True
                                break
                            pending[executor.submit(_scan_directory, child)] = (child, depth + 1)
                    if truncated:
                        break
                if context.is_cancelled:
                    for future in pending:
                        future.cancel()
                    return CollectorResult(CollectorStatus.CANCELLED,
                                           "cancelled during system-drive listing collection")
                if truncated:
                    break
        if truncated:
            lines.append("# truncated=true")
        output = context.workspace / "system_drive_listing.txt"
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        context.report_progress("system_drive_listing_finished", entry_count=entries, truncated=str(truncated).lower(),
                                bytes_written=artifact.size_bytes)
        summary = "threaded system-drive listing collected" if not truncated else "threaded system-drive listing collected with configured entry limit"
        return CollectorResult.succeeded(summary, (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the scoped thread pool has already shut down."""
