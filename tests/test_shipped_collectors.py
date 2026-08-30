"""Contract-level checks for collectors shipped in the repository tree."""

from __future__ import annotations

import unittest
from pathlib import Path

from logicytics import ResourceClass
from logicytics.discovery import preflight
from logicytics.modes import EXECUTION_MODES, mode_matrix


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
                "core.bluetooth.bluetooth_history",
                "core.browser.browser_data_backup",
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
                "core.packet.packet_capture",
                "core.packet.connection_graph",
                "core.memory.memory_snapshot",
                "core.media.media_backup",
                "core.process.running_processes",
                "core.process.detailed_processes",
                "core.process.process_memory",
                "core.process.memory_map",
                "core.registry.installed_applications",
                "core.registry.hklm_backup",
                "core.registry.startup_applications",
                "core.storage.logical_drives",
                "core.storage.physical_disks",
                "core.storage.mounted_volumes",
                "core.storage.volume_details",
                "core.ssh.ssh_backup",
                "core.system.system_info",
                "core.system.system_details",
                "core.system.bios_info",
                "core.system.operating_system",
                "core.system.computer_system",
                "core.system.defender_status",
                "core.system.session_snapshot",
                "core.system.environment_posture",
                "core.system.system_diagnostics",
                "core.system.installed_drivers",
                "core.system.installed_updates",
                "core.system.local_accounts",
                "core.system.scheduled_tasks",
                "core.system.windows_services",
                "core.system.windows_system_data_backup",
                "core.diagnostics.sysinternals_report",
                "core.system.group_policy",
                "core.usb.usb_storage_inventory",
                "core.wireless.wifi_profiles",
                "core.wireless.wifi_interfaces",
                "core.wireless.wifi_profile_keys",
                "core.hardware.windows_features",
                "core.hardware.display_adapters",
                "core.hardware.battery_status",
                "core.event_log.system_events",
                "core.event_log.application_events",
                "core.event_log.security_events",
                "core.filesystem.system_drive_tree",
                "core.filesystem.startup_folder_entries",
                "core.filesystem.sensitive_file_inventory",
                "core.filesystem.system_drive_listing",
                "core.encryption.bitlocker_status",
                "core.encryption.bitlocker_volumes",
            },
            {candidate.metadata.id for candidate in report.valid if candidate.metadata},
        )

    def test_event_log_collectors_explicitly_allow_bounded_parallel_scheduling(self) -> None:
        """The independent event channels may overlap while retaining separate identities."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)
        event_collectors = {
            candidate.metadata.id: candidate.metadata
            for candidate in report.valid
            if candidate.metadata and candidate.metadata.id.startswith("core.event_log.")
        }
        self.assertEqual(
            {
                "core.event_log.application_events",
                "core.event_log.security_events",
                "core.event_log.system_events",
            },
            set(event_collectors),
        )
        for metadata in event_collectors.values():
            self.assertTrue(metadata.parallel_safe)
            self.assertIs(ResourceClass.GENERAL, metadata.resource_class)
            self.assertEqual(("text/csv",), metadata.output_media_types)

    def test_mode_matrix_represents_every_discovered_collector_exactly_once(self) -> None:
        """Mode documentation covers selected, manual-only, and quarantined collectors."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)
        matrix = mode_matrix(report.candidates)
        collector_rows = matrix["collectors"]
        expected_ids = {
            candidate.metadata.id if candidate.metadata is not None else candidate.selection_id
            for candidate in report.candidates
        }
        represented_ids = [row["id"] for row in collector_rows]
        self.assertEqual(expected_ids, set(represented_ids))
        self.assertEqual(len(represented_ids), len(set(represented_ids)))

        modes = {row["name"]: row for row in matrix["modes"]}
        self.assertEqual(set(EXECUTION_MODES), set(modes))
        for collector in collector_rows:
            with self.subTest(collector=collector["id"]):
                self.assertEqual(
                    collector["valid"] and not collector["modes"],
                    collector["manual_only"],
                )
                for mode_name in collector["modes"]:
                    self.assertIn(collector["id"], modes[mode_name]["collector_ids"])
        for mode_name, mode in modes.items():
            expected = {
                row["id"] for row in collector_rows if mode_name in row["modes"]
            }
            self.assertEqual(expected, set(mode["collector_ids"]))


if __name__ == "__main__":
    unittest.main()
