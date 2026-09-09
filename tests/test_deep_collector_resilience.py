"""Regression checks for recoverable deep-collection platform failures."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.diagnostics import sysinternals_report
from core.network import network_interfaces
from core.process import detailed_processes
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.module.artifacts import WorkspaceArtifactWriter


def _context(root: Path, collector_id: str) -> CollectorContext:
    """Build a direct collector context with an isolated artifact destination."""
    workspace = root / "workspace"
    artifacts = root / "artifacts"
    workspace.mkdir()
    artifacts.mkdir()
    return CollectorContext(
        run_id="run-" + "0" * 32,
        collector_id=collector_id,
        workspace=workspace,
        temporary_directory=workspace / "tmp",
        artifacts=WorkspaceArtifactWriter(collector_id, workspace, artifacts, 1024 * 1024, 10),
        logger=Mock(),
        settings={},
        cancellation_file=workspace / ".cancelled",
    )


class DeepCollectorResilienceTests(unittest.TestCase):
    """Ensure environmental probe failures do not crash deep collection workers."""

    def test_detailed_tasklist_timeout_is_skipped(self) -> None:
        """A busy tasklist command must leave the remainder of a run usable."""
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), "core.process.detailed_processes")
            with patch.object(
                detailed_processes.subprocess,
                detailed_processes.subprocess.run.__name__,
                side_effect=subprocess.TimeoutExpired(["tasklist"], 40),
            ):
                result = detailed_processes.DetailedProcessesCollector().collect(context)
        self.assertIs(CollectorStatus.SKIPPED, result.status)
        self.assertIn("timed out", result.summary)

    def test_sysinternals_timeouts_are_recorded_in_the_report(self) -> None:
        """One hanging optional Sysinternals tool must not crash its worker."""
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), "core.diagnostics.sysinternals_report")
            with patch.object(
                sysinternals_report.subprocess,
                sysinternals_report.subprocess.run.__name__,
                side_effect=subprocess.TimeoutExpired(["psfile.exe"], 25),
            ):
                result = sysinternals_report.SysinternalsReportCollector().collect(context)
            report = (context.workspace / "sysinternals_report.txt").read_text(encoding="utf-8")
        self.assertIs(CollectorStatus.SUCCEEDED, result.status)
        self.assertIn("timed out after 25 seconds", report)

    def test_network_interfaces_accepts_unmatched_adapter_records(self) -> None:
        """IPv4 records without a current adapter must receive safe placeholder fields."""
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), "core.network.network_interfaces")
            response = '[{"InterfaceAlias":"stale","IPAddress":"192.0.2.1","PrefixLength":24}]'
            with patch.object(
                network_interfaces.subprocess,
                network_interfaces.subprocess.run.__name__,
                return_value=subprocess.CompletedProcess((), 0, response, ""),
            ) as run:
                result = network_interfaces.NetworkInterfacesCollector().collect(context)
        self.assertIs(CollectorStatus.SUCCEEDED, result.status)
        command = run.call_args.args[0][-1]
        self.assertIn("$null -eq $adapter", command)
        self.assertIn("Status = $status", command)


if __name__ == "__main__":
    unittest.main()
