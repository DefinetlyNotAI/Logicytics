"""Collect the local Windows IP configuration as a bounded text artifact."""

from __future__ import annotations

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


class NetworkAdaptersCollector(CoreCollector):
    """Run the local IP configuration command and register its output as evidence."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated adapter inventory and its fixed output bound."""
        return CollectorMetadata(
            id="core.network.network_adapters",
            name="Network adapters",
            version="4.0.0",
            specialty=Specialty.NETWORK,
            description="Captures local Windows IP configuration and adapter details through ipconfig.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            default_profiles=("standard", "deep"),
            timeout_seconds=20,
            maximum_output_bytes=2 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Confirm cancellation state and that ipconfig is available on the host."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("ipconfig") is None:
            return ValidationResult(False, reasons=("ipconfig is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Capture local adapter configuration and register the bounded plain-text report."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before adapter collection")
        context.report_progress("network_adapters_started")
        completed = subprocess.run(
            ["ipconfig", "/all"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"ipconfig exit code {completed.returncode}"
            if "permission denied" in detail.casefold() or (
                    "access" in detail.casefold() and "denied" in detail.casefold()
            ):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "ipconfig access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "ipconfig failed", errors=(detail,))
        output = context.workspace / "network_adapters.txt"
        output.write_text(completed.stdout, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        adapter_sections = completed.stdout.casefold().count("adapter ")
        context.report_progress(
            "network_adapters_finished",
            adapter_sections=adapter_sections,
            bytes_written=artifact.size_bytes,
        )
        return CollectorResult.succeeded("network adapter configuration collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because the ipconfig child process has completed."""
