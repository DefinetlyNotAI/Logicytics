"""Export local Wi-Fi interface state with a normalized interface-name field."""

from __future__ import annotations

import json

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from netsh output."""
    normalized = detail.casefold()
    return (
            "permission denied" in normalized
            or ("access" in normalized and "denied" in normalized)
            or "requires elevation" in normalized
    )


def _normalize_interface_name(name: str) -> str:
    """Normalize the common Wi-Fi and WiFi spelling variation for joins."""
    return "Wi-Fi" if name.strip().casefold() in {"wifi", "wi-fi"} else name.strip()


class WifiInterfacesCollector(CoreCollector):
    """Capture Wi-Fi interface metadata without requesting profile key material."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the bounded, subprocess-gated wireless interface artifact contract."""
        return CollectorMetadata(
            id="core.wireless.wifi_interfaces", name="Wi-Fi interfaces", version="4.0.0",
            specialty=Specialty.WIRELESS,
            output_media_types=("application/json",),
            description="Exports local Wi-Fi interface state and normalized interface names without key material.",
            author="Logicytics", supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("network_configuration",), default_profiles=("deep",),
            timeout_seconds=30, maximum_output_bytes=256 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and netsh availability before collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("netsh") is None:
            return ValidationResult(False, reasons=("netsh is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Query Wi-Fi interface output and register normalized JSON evidence."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before Wi-Fi interface collection")
        context.report_progress("wifi_interfaces_started")
        completed = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, check=False, text=True, timeout=25,
        )
        detail = completed.stderr.strip()
        if completed.returncode != 0:
            message = detail or completed.stdout.strip() or f"netsh exit code {completed.returncode}"
            if _is_access_denied(message):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "Wi-Fi interface access was denied for the current account", errors=(message,))
            return CollectorResult(CollectorStatus.FAILED, "Wi-Fi interface query failed", errors=(message,))

        interfaces: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.casefold().replace(" ", "_")
            if key == "name" and current:
                interfaces.append(current)
                current = {}
            current[key] = value
        if current:
            interfaces.append(current)
        for interface in interfaces:
            name = interface.get("name", "")
            interface["normalized_name"] = _normalize_interface_name(name)

        report = {"interfaces": interfaces, "raw_output": completed.stdout.strip()}
        output = context.workspace / "wifi_interfaces.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("wifi_interfaces_finished", bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("Wi-Fi interfaces collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because netsh exits before the result is returned."""
