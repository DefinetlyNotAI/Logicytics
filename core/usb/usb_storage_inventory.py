"""Collect Windows USB storage device metadata from the read-only USBSTOR registry tree."""

from __future__ import annotations

import ctypes
import json
import winreg
from ctypes import wintypes
from datetime import datetime, timedelta, timezone

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus

_USBSTOR_PATH = r"SYSTEM\CurrentControlSet\Enum\USBSTOR"


def _registry_last_write(key: winreg.HKEYType) -> str | None:
    """Return one registry key's last-write timestamp in UTC when Windows reports it."""
    timestamp = wintypes.FILETIME()
    result = ctypes.windll.advapi32.RegQueryInfoKeyW(
        wintypes.HKEY(key.handle), None, None, None, None, None, None, None, None, None, None, ctypes.byref(timestamp)
    )
    if result != 0:
        return None
    ticks = (timestamp.dwHighDateTime << 32) | timestamp.dwLowDateTime
    return (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)).isoformat()


def _enumerate_usb_storage() -> list[dict[str, str | None]]:
    """Return USBSTOR device class, instance ID, and friendly name when readable."""
    devices: list[dict[str, str | None]] = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _USBSTOR_PATH) as root:
        class_index = 0
        while True:
            try:
                device_class = winreg.EnumKey(root, class_index)
            except OSError:
                break
            class_index += 1
            class_path = f"{_USBSTOR_PATH}\\{device_class}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_path) as class_key:
                instance_index = 0
                while True:
                    try:
                        instance_id = winreg.EnumKey(class_key, instance_index)
                    except OSError:
                        break
                    instance_index += 1
                    instance_path = f"{class_path}\\{instance_id}"
                    friendly_name = instance_id
                    try:
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, instance_path) as instance_key:
                            friendly_name = str(winreg.QueryValueEx(instance_key, "FriendlyName")[0])
                            last_write = _registry_last_write(instance_key)
                    except OSError:
                        last_write = None
                    devices.append(
                        {
                            "device_class": device_class,
                            "instance_id": instance_id,
                            "friendly_name": friendly_name,
                            "last_write": last_write,
                        }
                    )
    return devices


class UsbStorageInventoryCollector(CoreCollector):
    """Register a bounded JSON inventory of USB storage devices visible in the registry."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the registry-read capability and device-metadata-only output scope."""
        return CollectorMetadata(
            id="core.usb.usb_storage_inventory",
            name="USB storage inventory",
            version="4.0.0",
            specialty=Specialty.USB,
            output_media_types=("application/json",),
            description="Reads USB storage device class, instance ID, friendly name, and last-write time from USBSTOR.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.REGISTRY_READ,),
            default_profiles=("deep",),
            timeout_seconds=20,
            maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation and read access to the USBSTOR registry root."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _USBSTOR_PATH):
                return ValidationResult(True)
        except OSError as error:
            return ValidationResult(False, reasons=(f"USBSTOR registry access is unavailable: {error}",))

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Read USB storage metadata and register the resulting JSON artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before USB registry collection")
        context.report_progress("usb_storage_inventory_started")
        try:
            devices = _enumerate_usb_storage()
        except OSError as error:
            return CollectorResult(
                CollectorStatus.SKIPPED,
                "USBSTOR registry access was denied during collection",
                errors=(str(error),),
            )
        report = {"collected_at": datetime.now(timezone.utc).isoformat(), "devices": devices}
        output = context.workspace / "usb_storage_inventory.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("usb_storage_inventory_finished", device_count=len(devices),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("USB storage inventory collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because registry handles are closed during enumeration."""
