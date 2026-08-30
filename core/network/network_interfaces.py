"""Export local Windows interface addresses and link state as bounded JSON evidence."""

from __future__ import annotations

import ipaddress
import json
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


def _enrich_ipv4_networks(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add deterministic IPv4 netmask and broadcast fields from each prefix length."""
    enriched: list[dict[str, object]] = []
    for record in records:
        item = dict(record)
        address = item.get("IPAddress")
        prefix = item.get("PrefixLength")
        if isinstance(address, str) and isinstance(prefix, int):
            try:
                network = ipaddress.IPv4Network(f"{address}/{prefix}", strict=False)
            except ValueError:
                pass
            else:
                item["Netmask"] = str(network.netmask)
                item["BroadcastAddress"] = str(network.broadcast_address)
        enriched.append(item)
    return enriched


class NetworkInterfacesCollector(CoreCollector):
    """Capture read-only IPv4 addresses, masks, broadcasts, and adapter link metadata."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated network-interface JSON artifact contract."""
        return CollectorMetadata(
            id="core.network.network_interfaces",
            name="Network interfaces",
            version="4.0.0",
            specialty=Specialty.NETWORK,
            output_media_types=("application/json",),
            description="Exports IPv4 addresses, masks, broadcasts, link states, speeds, and duplex data.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("network_identifiers",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query interface address and link data, then register JSON evidence."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before network-interface collection")
        context.report_progress("network_interfaces_started")
        command = (
            "$adapters = Get-NetAdapter; "
            "Get-NetIPAddress -AddressFamily IPv4 | ForEach-Object { "
            "$adapter = $adapters | Where-Object ifIndex -eq $_.InterfaceIndex | Select-Object -First 1; "
            "[pscustomobject]@{ InterfaceAlias = $_.InterfaceAlias; IPAddress = $_.IPAddress; "
            "PrefixLength = $_.PrefixLength; AddressState = $_.AddressState.ToString(); "
            "Status = $adapter.Status.ToString(); LinkSpeed = $adapter.LinkSpeed; "
            "MediaConnectionState = $adapter.MediaConnectionState.ToString(); FullDuplex = $adapter.FullDuplex } "
            "} | ConvertTo-Json -Depth 3"
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
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "network-interface access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "network-interface query failed", errors=(detail,))
        try:
            interfaces = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return CollectorResult(CollectorStatus.FAILED, "network-interface query returned invalid JSON",
                                   errors=(str(error),))
        records = interfaces if isinstance(interfaces, list) else [interfaces]
        if not all(isinstance(record, dict) for record in records):
            return CollectorResult(CollectorStatus.FAILED, "network-interface query returned an unexpected result")
        output = context.workspace / "network_interfaces.json"
        output.write_text(json.dumps(_enrich_ipv4_networks(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("network_interfaces_finished", interface_count=len(records),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("network-interface inventory collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because PowerShell exits before the result is returned."""
