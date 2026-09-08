"""Discover and report bounded output from supported local Sysinternals tools."""

from __future__ import annotations

import os
from pathlib import Path

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import process_adapter as subprocess

TOOLS = ("psfile", "psgetsid", "psinfo", "pslist", "psloggedon", "psloglist")
MAX_OUTPUT_CHARS = 512_000


class SysinternalsReportCollector(CoreCollector):
    """Run available local Sysinternals tools and consolidate their diagnostic output."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated Sysinternals report artifact contract."""
        return CollectorMetadata(
            id="core.diagnostics.sysinternals_report",
            name="Sysinternals report",
            version="4.0.0",
            specialty=Specialty.DIAGNOSTICS,
            output_media_types=("text/plain",),
            description="Reports supported Sysinternals binary/archive state and consolidates available tool output.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_diagnostics",),
            default_profiles=("deep",),
            timeout_seconds=180,
            maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state before local tool discovery."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Discover local binaries, execute available tools, and write one text report."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Sysinternals collection")
        project_root = Path(__file__).resolve().parents[2]
        search_roots = (
            project_root / "sysinternals",
            project_root / "tools" / "sysinternals",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Sysinternals",
        )
        archive_paths = tuple(root.with_suffix(".zip") for root in search_roots)
        sections: list[str] = ["Sysinternals report", ""]
        context.report_progress("sysinternals_report_started")
        for tool in TOOLS:
            binary = next(
                (root / f"{tool}.exe" for root in search_roots if (root / f"{tool}.exe").is_file()),
                None,
            )
            sections.append(f"## {tool}")
            if binary is None:
                archive_state = "archive available" if any(path.is_file() for path in archive_paths) else "binary and archive missing"
                sections.extend((f"status: {archive_state}", ""))
                continue
            try:
                completed = subprocess.run([str(binary)], capture_output=True, check=False, text=True, timeout=25)
            except OSError as error:
                sections.extend((f"status: execution error: {error}", ""))
                continue
            output = (completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr).strip()
            sections.extend((f"status: executed (exit {completed.returncode})", output[:MAX_OUTPUT_CHARS], ""))
        report = "\n".join(sections)
        output_path = context.workspace / "sysinternals_report.txt"
        output_path.write_text(report[:MAX_OUTPUT_CHARS], encoding="utf-8")
        artifact = context.artifacts.register_file(output_path, media_type="text/plain")
        context.report_progress("sysinternals_report_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Sysinternals report collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because each tool exits before its output is returned."""
