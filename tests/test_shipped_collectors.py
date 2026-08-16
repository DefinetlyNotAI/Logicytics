"""Contract-level checks for collectors shipped in the repository tree."""

from __future__ import annotations

import unittest
from pathlib import Path

from logicytics.discovery import preflight


class ShippedCollectorTests(unittest.TestCase):
    """Ensure every checked-in core collector passes strict discovery rules."""

    def test_shipped_collectors_pass_preflight(self) -> None:
        """The repository must never contain an invalid shipped collector."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)
        self.assertEqual((), report.invalid)
        self.assertEqual(
            {
                "core.bluetooth.paired_devices",
                "core.bluetooth.bluetooth_addresses",
                "core.network.network_identity",
                "core.network.active_connections",
                "core.network.connection_processes",
                "core.network.dns_cache",
                "core.network.firewall_profiles",
                "core.network.adapter_statistics",
                "core.network.bandwidth_sample",
                "core.network.network_interfaces",
                "core.network.network_adapters",
                "core.network.arp_cache",
                "core.network.routing_table",
                "core.memory.memory_snapshot",
                "core.process.running_processes",
                "core.process.detailed_processes",
                "core.process.process_memory",
                "core.registry.installed_applications",
                "core.registry.hklm_backup",
                "core.registry.startup_applications",
                "core.storage.logical_drives",
                "core.storage.physical_disks",
                "core.storage.mounted_volumes",
                "core.storage.volume_details",
                "core.system.system_info",
                "core.system.system_details",
                "core.system.bios_info",
                "core.system.operating_system",
                "core.system.computer_system",
                "core.system.session_snapshot",
                "core.system.environment_posture",
                "core.system.system_diagnostics",
                "core.system.installed_drivers",
                "core.system.installed_updates",
                "core.system.windows_services",
                "core.system.group_policy",
                "core.usb.usb_storage_inventory",
                "core.wireless.wifi_profiles",
                "core.wireless.wifi_interfaces",
                "core.wireless.wifi_profile_keys",
                "core.hardware.windows_features",
                "core.hardware.battery_status",
                "core.event_log.system_events",
                "core.event_log.application_events",
                "core.event_log.security_events",
                "core.filesystem.system_drive_tree",
                "core.filesystem.system_drive_listing",
                "core.encryption.bitlocker_status",
                "core.encryption.bitlocker_volumes",
            },
            {candidate.metadata.id for candidate in report.valid if candidate.metadata},
        )


if __name__ == "__main__":
    unittest.main()
