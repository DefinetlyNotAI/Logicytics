"""Collect the hostname and resolver-provided addresses for the local host."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    NetworkAccess,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import network_adapter as socket


class NetworkIdentityCollector(CoreCollector):
    """Capture a bounded local network identity report without probing remote hosts."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare resolver access and the small JSON output produced by this collector."""
        return CollectorMetadata(
            id="core.network.network_identity",
            name="Network identity",
            version="4.0.0",
            specialty=Specialty.NETWORK,
            output_media_types=("application/json",),
            description="Records the local hostname and resolver-provided IP addresses without probing hosts.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.NETWORK,),
            network_access=NetworkAccess.LOCAL,
            default_profiles=("standard", "deep"),
            timeout_seconds=10,
            maximum_output_bytes=64 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Confirm that the run has not been cancelled before resolving local identity."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Resolve the local hostname and write a deterministic JSON report."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before network identity collection")
        context.report_progress("network_identity_started")
        hostname = socket.gethostname()
        addresses: list[str] = []
        resolver_error: str | None = None
        try:
            for result in socket.getaddrinfo(hostname, None):
                address = result[4][0]
                if address not in addresses:
                    addresses.append(address)
        except socket.gaierror as error:
            resolver_error = str(error)
        report = {
            "collected_at": datetime.now(UTC).isoformat(),
            "hostname": hostname,
            "addresses": sorted(addresses),
        }
        if resolver_error is not None:
            report["resolver_error"] = resolver_error
        output = context.workspace / "network_identity.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress(
            "network_identity_finished",
            address_count=len(addresses),
            bytes_written=artifact.size_bytes,
        )
        if resolver_error is not None:
            return CollectorResult(
                CollectorStatus.PARTIAL,
                "network identity collected without resolver addresses",
                (artifact,),
                errors=(resolver_error,),
            )
        return CollectorResult.succeeded("network identity collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because hostname resolution has already completed."""
