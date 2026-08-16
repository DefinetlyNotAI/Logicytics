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
                "core.network.network_identity",
                "core.network.network_adapters",
                "core.network.arp_cache",
                "core.network.routing_table",
                "core.memory.memory_snapshot",
                "core.process.running_processes",
                "core.storage.logical_drives",
                "core.system.system_info",
                "core.system.system_details",
                "core.system.installed_drivers",
                "core.system.group_policy",
                "core.usb.usb_storage_inventory",
                "core.hardware.windows_features",
                "core.event_log.system_events",
            },
            {candidate.metadata.id for candidate in report.valid if candidate.metadata},
        )


if __name__ == "__main__":
    unittest.main()
