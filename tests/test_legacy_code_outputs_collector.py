"""Behavioral coverage for importing generated evidence from historical CODE checkouts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.integration.legacy_code_outputs import LegacyCodeOutputsCollector
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.module.artifacts import WorkspaceArtifactWriter


class LegacyCodeOutputsCollectorTests(unittest.TestCase):
    """Prove that only bounded generated evidence reaches the artifact catalog."""

    def test_generated_evidence_is_imported_while_project_material_is_excluded(self) -> None:
        """Source, executables, models, configuration, caches, and libraries never enter artifacts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "CODE"
            code.mkdir()
            (code / "report.txt").write_text("text evidence\n", encoding="utf-8")
            nested = code / "reports"
            nested.mkdir()
            (nested / "events.csv").write_text("event_id\n1\n", encoding="utf-8")
            excluded = {
                "collector.py": "source",
                "script.ps1": "source",
                "tool.exe": "binary",
                "model.pth": "model",
                "config.ini": "secret configuration",
            }
            for name, contents in excluded.items():
                (code / name).write_text(contents, encoding="utf-8")
            cache = code / "__pycache__"
            cache.mkdir()
            (cache / "cached.json").write_text("{}", encoding="utf-8")
            library = code / "lib"
            library.mkdir()
            (library / "internal.log").write_text("internal", encoding="utf-8")

            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            cancellation = root / "cancel.requested"
            metadata = LegacyCodeOutputsCollector.metadata()
            writer = WorkspaceArtifactWriter(
                metadata.id,
                workspace,
                artifact_root,
                metadata.maximum_output_bytes,
                metadata.maximum_artifact_files,
                maximum_artifact_bytes=metadata.maximum_artifact_bytes,
            )
            context = CollectorContext(
                run_id="run-test",
                collector_id=metadata.id,
                workspace=workspace,
                temporary_directory=workspace / "tmp",
                artifacts=writer,
                logger=MagicMock(),
                settings={},
                cancellation_file=cancellation,
            )
            collector = LegacyCodeOutputsCollector()
            with patch.object(collector, "_code_root", return_value=code):
                self.assertTrue(collector.validate(context).valid)
                result = collector.collect(context)

            self.assertIs(CollectorStatus.SUCCEEDED, result.status)
            self.assertEqual({"report.txt", "events.csv"}, {artifact.name for artifact in result.artifacts})
            stored = {path.name for path in artifact_root.rglob("*") if path.is_file()}
            self.assertEqual({"report.txt", "events.csv"}, stored)
            for name in excluded:
                self.assertFalse(any(path.name == name for path in workspace.rglob("*")))
                self.assertFalse(any(path.name == name for path in artifact_root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
