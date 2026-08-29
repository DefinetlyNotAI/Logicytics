"""Export Windows BIOS manufacturer, name, and version as a bounded HTML table."""

from __future__ import annotations

import html
import json
import subprocess
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
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


def _render_bios_table(bios: dict[str, object]) -> str:
    """Render trusted structured BIOS fields into a portable evidence table."""
    rows = "\n".join(
        f"      <tr><th>{html.escape(label)}</th><td>{html.escape(str(bios.get(key) or 'unavailable'))}</td></tr>"
        for key, label in (
            ("Manufacturer", "Manufacturer"),
            ("Name", "Name"),
            ("SMBIOSBIOSVersion", "SMBIOS BIOS version"),
            ("Version", "Version"),
            ("ReleaseDate", "Release date"),
        )
    )
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en">',
            "  <head><meta charset=\"utf-8\"><title>BIOS information</title></head>",
            "  <body>",
            "    <h1>BIOS information</h1>",
            "    <table>",
            "      <thead><tr><th>Field</th><th>Value</th></tr></thead>",
            f"      <tbody>\n{rows}\n      </tbody>",
            "    </table>",
            "  </body>",
            "</html>",
            "",
        )
    )


class BiosInfoCollector(CoreCollector):
    """Capture a read-only BIOS inventory through modern Windows CIM data."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated BIOS HTML artifact contract."""
        return CollectorMetadata(
            id="core.system.bios_info",
            name="BIOS information",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("text/html",),
            description="Exports BIOS manufacturer, name, version, and release date as HTML.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("system_configuration",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=128 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query BIOS CIM data and register an HTML evidence table."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before BIOS collection")
        context.report_progress("bios_info_started")
        command = (
            "Get-CimInstance -ClassName Win32_BIOS | "
            "Select-Object Manufacturer, Name, SMBIOSBIOSVersion, Version, ReleaseDate | "
            "ConvertTo-Json -Compress"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=40,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "BIOS CIM access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "BIOS CIM query failed", errors=(detail,))
        try:
            bios = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "BIOS CIM query returned invalid JSON", errors=(str(error),))
        if not isinstance(bios, dict):
            return CollectorResult(CollectorStatus.FAILED, "BIOS CIM query returned an unexpected result")
        output = context.workspace / "bios_info.html"
        output.write_text(_render_bios_table(bios), encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/html")
        context.report_progress("bios_info_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("BIOS information collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
