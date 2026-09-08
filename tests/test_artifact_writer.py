from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from logicytics.contracts import (
    EvidenceKind,
    RunRequest,
)
from logicytics.module.artifacts import WorkspaceArtifactWriter
from logicytics.module.configuration import (
    default_config,
)
from logicytics.module.discovery import preflight
from logicytics.module.errors import ArtifactError
from logicytics.module.planner import build_plan
from logicytics.module.runtime import RunSupervisor
from tests.fixtures.collectors import COLLECTOR


class ArtifactWriterTests(unittest.TestCase):
    """Artifact registration, ownership, limits, and concurrent budget behavior."""

    def test_artifacts_cannot_escape_collector_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            outside = root / "outside.txt"
            outside.write_text("no", encoding="utf-8")
            writer = WorkspaceArtifactWriter("core.system.test", workspace, artifact_root, 1024, 1)
            with self.assertRaises(ArtifactError):
                writer.register_file(outside)
            linked_source = workspace / "linked-source.txt"
            linked_source.symlink_to(outside)
            with self.assertRaisesRegex(ArtifactError, "collector workspace"):
                writer.register_file(linked_source)

    def test_artifact_destination_symlink_cannot_escape_run_store(self) -> None:
        """Pre-existing artifact junctions or symlinks must never redirect evidence outside the run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            outside = root / "outside"
            workspace.mkdir()
            artifact_root.mkdir()
            outside.mkdir()
            source = workspace / "report.txt"
            source.write_text("evidence\n", encoding="utf-8")
            (artifact_root / "core_system_test").symlink_to(outside, target_is_directory=True)
            writer = WorkspaceArtifactWriter("core.system.test", workspace, artifact_root, 1024, 1)

            with self.assertRaisesRegex(ArtifactError, "destination"):
                writer.register_file(source)
            self.assertEqual([], list(outside.iterdir()))

    def test_artifact_writer_enforces_declared_output_name_and_format(self) -> None:
        """A core collector cannot silently drift from its published output contract."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            expected = workspace / "system_info.json"
            unexpected = workspace / "renamed.json"
            expected.write_text("{}\n", encoding="utf-8")
            unexpected.write_text("{}\n", encoding="utf-8")
            writer = WorkspaceArtifactWriter(
                "core.system.system_info",
                workspace,
                artifact_root,
                1024,
                2,
                allowed_relative_paths=("system_info.json",),
                allowed_media_types=("application/json",),
            )

            with self.assertRaisesRegex(ArtifactError, "output contract"):
                writer.register_file(unexpected, media_type="application/json")
            with self.assertRaisesRegex(ArtifactError, "media type"):
                writer.register_file(expected, media_type="text/plain")
            artifact = writer.register_file(expected, media_type="application/json")
            self.assertEqual("core_system_system_info/system_info.json", artifact.relative_path)

    def test_cancelled_artifact_copy_never_publishes_partial_evidence(self) -> None:
        """Cancellation must stop staging without creating a final or temporary artifact."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            source = workspace / "report.txt"
            source.write_text("evidence\n", encoding="utf-8")
            cancellation_file = root / ".cancelled"
            cancellation_file.touch()
            writer = WorkspaceArtifactWriter(
                "core.system.test",
                workspace,
                artifact_root,
                1024,
                1,
                cancellation_file=cancellation_file,
            )

            with self.assertRaisesRegex(ArtifactError, "cancelled"):
                writer.register_file(source)
            self.assertEqual((), writer.artifacts)
            self.assertEqual([], list(artifact_root.rglob("*")))

    def test_mutating_artifact_source_cannot_bypass_declared_byte_limits(self) -> None:
        """Growing an evidence file after its initial size check must not publish partial data."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            source = workspace / "report.bin"
            source.write_bytes(b"1234")
            writer = WorkspaceArtifactWriter(
                "core.system.test",
                workspace,
                artifact_root,
                4,
                1,
            )
            original_copy = writer._copy_artifact

            def mutate_before_copy(*arguments):
                source.write_bytes(b"12345")
                return original_copy(*arguments)

            with patch.object(writer, "_copy_artifact", side_effect=mutate_before_copy):
                with self.assertRaisesRegex(ArtifactError, "maximum_artifact_bytes"):
                    writer.register_file(source)
            self.assertEqual((), writer.artifacts)
            self.assertEqual([], list((artifact_root / "core_system_test").iterdir()))

    def test_cancellation_during_streaming_removes_incomplete_artifact(self) -> None:
        """Mid-transfer cancellation must remove temporary evidence and publish nothing."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            source = workspace / "large.bin"
            source.write_bytes(b"x" * (2 * 1024 * 1024))
            cancellation_file = root / ".cancelled"
            writer = WorkspaceArtifactWriter(
                "core.system.test",
                workspace,
                artifact_root,
                3 * 1024 * 1024,
                1,
                cancellation_file=cancellation_file,
            )
            original_check = writer._check_cancellation
            checks = 0

            def cancel_during_second_chunk() -> None:
                nonlocal checks
                checks += 1
                if checks == 3:
                    cancellation_file.touch()
                original_check()

            with patch.object(writer, "_check_cancellation", side_effect=cancel_during_second_chunk):
                with self.assertRaisesRegex(ArtifactError, "cancelled"):
                    writer.register_file(source)
            self.assertEqual((), writer.artifacts)
            self.assertEqual([], list((artifact_root / "core_system_test").iterdir()))

    def test_artifact_registration_enforces_collector_file_count_limit(self) -> None:
        """A collector cannot exceed its declared artifact count even with tiny files."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            first = workspace / "first.txt"
            second = workspace / "second.txt"
            first.write_text("one", encoding="utf-8")
            second.write_text("two", encoding="utf-8")
            writer = WorkspaceArtifactWriter("core.system.test", workspace, artifact_root, 1024, 1)
            writer.register_file(first, media_type="text/plain")
            with self.assertRaisesRegex(ArtifactError, "maximum_artifact_files"):
                writer.register_file(second, media_type="text/plain")

    def test_concurrent_artifact_registration_allocates_unique_owned_paths(self) -> None:
        """Simultaneous registration of one source cannot overwrite another catalog entry."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            source = workspace / "shared.txt"
            source.write_text("evidence", encoding="utf-8")
            writer = WorkspaceArtifactWriter("core.system.test", workspace, artifact_root, 1024, 20)

            with ThreadPoolExecutor(max_workers=10) as executor:
                artifacts = list(executor.map(lambda _: writer.register_file(source), range(20)))

            paths = [artifact.relative_path for artifact in artifacts]
            self.assertEqual(20, len(set(paths)))
            self.assertEqual(20, len(writer.artifacts))
            self.assertTrue(all((artifact_root / path).read_text(encoding="utf-8") == "evidence" for path in paths))

    def test_concurrent_artifact_registration_cannot_bypass_file_or_byte_limits(self) -> None:
        """Parallel callers observe one atomic quota and cannot overfill evidence storage."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            source = workspace / "shared.txt"
            source.write_text("123", encoding="utf-8")
            writer = WorkspaceArtifactWriter("core.system.test", workspace, artifact_root, 6, 2)

            def register(_: int) -> bool:
                try:
                    writer.register_file(source)
                except ArtifactError:
                    return False
                return True

            with ThreadPoolExecutor(max_workers=12) as executor:
                outcomes = list(executor.map(register, range(24)))

            self.assertEqual(2, sum(outcomes))
            self.assertEqual(2, len(writer.artifacts))
            self.assertEqual(6, sum(artifact.size_bytes for artifact in writer.artifacts))
            self.assertEqual(2, len(list((artifact_root / "core_system_test").iterdir())))

    def test_artifact_registration_enforces_individual_file_size_limit(self) -> None:
        """A collector-specific file ceiling rejects large evidence before it is copied."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            oversized = workspace / "oversized.bin"
            oversized.write_bytes(b"12345")
            writer = WorkspaceArtifactWriter(
                "core.system.test",
                workspace,
                artifact_root,
                1024,
                5,
                maximum_artifact_bytes=4,
            )
            with self.assertRaisesRegex(ArtifactError, "maximum_artifact_bytes"):
                writer.register_file(oversized)
            self.assertEqual((), writer.artifacts)
            self.assertEqual([], list(artifact_root.rglob("*")))

    def test_artifact_registration_enforces_reserved_run_output_budget(self) -> None:
        """A worker cannot copy evidence beyond its supervisor-reserved run quota."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            source = workspace / "oversized.bin"
            source.write_bytes(b"12345")
            writer = WorkspaceArtifactWriter(
                "core.system.test",
                workspace,
                artifact_root,
                1024,
                5,
                run_output_budget_bytes=4,
            )
            with self.assertRaisesRegex(ArtifactError, "maximum_run_output_bytes"):
                writer.register_file(source)
            self.assertEqual([], list(artifact_root.rglob("*")))

    def test_isolated_worker_enforces_declared_individual_artifact_limit(self) -> None:
        """Metadata file limits must survive preflight and reach the isolated worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            maximum_artifact_bytes=2,',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)

            record = outcome.manifest.collectors[0]
            self.assertEqual("failed", record.status)
            self.assertTrue(any("maximum_artifact_bytes" in error for error in record.errors))
            self.assertEqual([], record.artifacts)

    def test_artifact_registration_records_validated_provenance(self) -> None:
        """Artifact provenance is typed, immutable, and completed by the writer."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            source = workspace / "report.json"
            source.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "source_category"):
                WorkspaceArtifactWriter("invalid", workspace, artifact_root, 1024, 1)
            writer = WorkspaceArtifactWriter(
                "plugin.example",
                workspace,
                artifact_root,
                1024,
                1,
                source_category="evidence_graph",
            )
            with self.assertRaisesRegex(ArtifactError, "transformations"):
                writer.register_file(source, transformations=["normalized"])  # type: ignore[arg-type]
            with self.assertRaisesRegex(ArtifactError, "media_type"):
                writer.register_file(source, media_type="not a mime type")
            with self.assertRaisesRegex(ArtifactError, "evidence_kind"):
                writer.register_file(source, evidence_kind="raw")  # type: ignore[arg-type]
            self.assertEqual((), writer.artifacts)
            artifact = writer.register_file(
                source,
                evidence_kind=EvidenceKind.RAW,
                transformations=("normalized",),
            )
            self.assertEqual("report.json", artifact.name)
            self.assertEqual("registered", artifact.status)
            self.assertEqual("evidence_graph", artifact.source_category)
            self.assertEqual(EvidenceKind.RAW, artifact.evidence_kind)
            datetime.fromisoformat(artifact.collected_at)
            self.assertEqual(
                ("normalized", "copied into run artifact store"),
                artifact.transformations,
            )


if __name__ == "__main__":
    unittest.main()
