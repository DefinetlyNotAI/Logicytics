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
                "core.network.network_identity",
                "core.network.network_adapters",
                "core.memory.memory_snapshot",
                "core.process.running_processes",
                "core.storage.logical_drives",
                "core.system.system_info",
                "core.hardware.windows_features",
            },
            {candidate.metadata.id for candidate in report.valid if candidate.metadata},
        )


if __name__ == "__main__":
    unittest.main()
