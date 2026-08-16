"""Behavior checks for the capability-gated process collector."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from logicytics.contracts import ArtifactWriter, CollectorContext, CollectorStatus, EventLogger


class _UnusedArtifactWriter(ArtifactWriter):
    """Raise if a skipped collector unexpectedly attempts artifact registration."""

    def register_file(self, source: Path, *, media_type: str = "application/octet-stream"):
        raise AssertionError("a skipped collector must not register artifacts")


class _NoopLogger(EventLogger):
    """Discard events while directly exercising a collector method."""

    def event(self, level: str, message: str, **fields: int | float | str) -> None:
        return None


def _collector_type():
    path = Path(__file__).resolve().parent.parent / "core" / "process" / "running_processes.py"
    spec = importlib.util.spec_from_file_location("running_processes_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.RunningProcessesCollector


class RunningProcessesCollectorTests(unittest.TestCase):
    """Ensure expected local permission restrictions are reported as a skip."""

    def test_access_denied_is_skipped_not_failed(self) -> None:
        """Tasklist permission denials should preserve the rest of the run."""
        module, collector_type = _collector_type()
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            context = CollectorContext(
                run_id="test-run",
                collector_id="core.process.running_processes",
                workspace=workspace,
                artifacts=_UnusedArtifactWriter(),
                logger=_NoopLogger(),
                settings={},
                cancellation_file=workspace / ".cancelled",
            )
            with patch.object(
                module.subprocess,
                "run",
                return_value=CompletedProcess([], 1, "", "ERROR: Access denied"),
            ):
                result = collector_type().collect(context)
        self.assertEqual(CollectorStatus.SKIPPED, result.status)
        self.assertIn("denied", result.summary.casefold())


if __name__ == "__main__":
    unittest.main()
