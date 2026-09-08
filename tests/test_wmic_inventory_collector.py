"""Focused behavior checks for the optional legacy WMIC integration."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.system import wmic_inventory
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.module.artifacts import WorkspaceArtifactWriter


class WmicInventoryCollectorTests(unittest.TestCase):
    def test_wmic_inventory_runs_the_bounded_read_only_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifacts = root / "artifacts"
            workspace.mkdir()
            artifacts.mkdir()
            context = CollectorContext(
                run_id="run-" + "0" * 32,
                collector_id="core.system.wmic_inventory",
                workspace=workspace,
                temporary_directory=workspace / "tmp",
                artifacts=WorkspaceArtifactWriter("core.system.wmic_inventory", workspace, artifacts, 256 * 1024, 1),
                logger=Mock(),
                settings={},
                cancellation_file=workspace / ".cancelled",
            )
            response = "Manufacturer=Example Corp\r\nModel=Example Model\r\n"
            with (
                patch.object(
                    wmic_inventory,
                    wmic_inventory.which.__name__,
                    return_value="C:/Windows/System32/wbem/WMIC.exe",
                ),
                patch.object(
                    wmic_inventory.subprocess,
                    wmic_inventory.subprocess.run.__name__,
                    return_value=subprocess.CompletedProcess((), 0, response, ""),
                ) as run,
            ):
                collector = wmic_inventory.WmicInventoryCollector()
                self.assertTrue(collector.validate(context).valid)
                result = collector.collect(context)

            self.assertIs(CollectorStatus.SUCCEEDED, result.status)
            self.assertEqual(
                "Manufacturer=Example Corp\nModel=Example Model\n",
                (workspace / "wmic_inventory.txt").read_text(encoding="utf-8"),
            )
            command = run.call_args.args[0]
            self.assertEqual("C:/Windows/System32/wbem/WMIC.exe", command[0])
            self.assertEqual(("computersystem", "get"), tuple(command[1:3]))
            self.assertEqual(25, run.call_args.kwargs["timeout"])
            self.assertFalse(run.call_args.kwargs["check"])

    def test_absent_wmic_is_an_explicit_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = CollectorContext(
                run_id="run-" + "0" * 32,
                collector_id="core.system.wmic_inventory",
                workspace=root,
                temporary_directory=root / "tmp",
                artifacts=Mock(),
                logger=Mock(),
                settings={},
                cancellation_file=root / ".cancelled",
            )
            with patch.object(
                wmic_inventory,
                wmic_inventory.which.__name__,
                return_value=None,
            ):
                collector = wmic_inventory.WmicInventoryCollector()
                validation = collector.validate(context)
                result = collector.collect(context)
        self.assertFalse(validation.valid)
        self.assertIn("not installed", validation.reasons[0])
        self.assertIs(CollectorStatus.SKIPPED, result.status)


if __name__ == "__main__":
    unittest.main()
