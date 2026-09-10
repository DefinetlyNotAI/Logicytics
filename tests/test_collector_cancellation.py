"""Direct cooperative-cancellation tests for long-running collector phases."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from core.browser import browser_data_backup
from core.event_log import application_events, security_events, system_events
from core.filesystem import sensitive_file_inventory
from core.media import media_backup
from core.packet import packet_capture
from core.process import memory_map
from core.ssh import ssh_backup
from core.system import windows_system_data_backup
from logicytics.contracts import (
    Artifact,
    ArtifactWriter,
    CollectorContext,
    CollectorStatus,
    EventLogger,
    EvidenceKind,
)


class _RejectingArtifactWriter(ArtifactWriter):
    """Reject artifact publication during cancellation tests."""

    def register_file(
            self,
            source: Path,
            *,
            media_type: str = "application/octet-stream",
            evidence_kind: EvidenceKind = EvidenceKind.DERIVED,
            transformations: tuple[str, ...] = (),
    ) -> Artifact:
        raise AssertionError(f"cancelled collector unexpectedly registered {source}")


class _NullEventLogger(EventLogger):
    """Discard structured progress events emitted during tests."""

    def event(
            self,
            level: str,
            message: str,
            **fields: float | str,
    ) -> None:
        return None


def _context(
        workspace: Path,
        settings: dict[str, object] | None = None,
        *,
        collector_id: str = "core.test.cancellation",
) -> tuple[CollectorContext, Path]:
    """Create a real CollectorContext and its cancellation sentinel."""

    workspace.mkdir(parents=True, exist_ok=True)

    temporary_directory = workspace / "tmp"
    temporary_directory.mkdir(parents=True, exist_ok=True)

    cancellation_file = workspace / ".cancel"

    context = CollectorContext(
        run_id="test-cancellation",
        collector_id=collector_id,
        workspace=workspace,
        temporary_directory=temporary_directory,
        artifacts=_RejectingArtifactWriter(),
        logger=_NullEventLogger(),
        settings=settings or {},
        cancellation_file=cancellation_file,
    )

    return context, cancellation_file


class CollectorCancellationTests(unittest.TestCase):
    """Verify cooperative cancellation across long-running collectors."""

    def test_sensitive_copy_cancellation_removes_unpublished_bytes(self) -> None:
        """Cancellation during a sensitive-file copy removes unpublished data."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            source_root = root / "source"
            source_root.mkdir()

            (source_root / "password.txt").write_text(
                "sensitive",
                encoding="utf-8",
            )

            context, cancellation_file = _context(
                root / "workspace",
                {
                    "root": str(source_root),
                    "max_directories": 10,
                    "max_matches": 10,
                },
                collector_id="core.filesystem.sensitive_file_inventory",
            )

            def cancel_after_copy(
                    source: Path,
                    destination: Path,
            ) -> Path:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
                cancellation_file.touch()
                return destination

            with patch(
                    "core.filesystem.sensitive_file_inventory._copy",
                    side_effect=cancel_after_copy,
            ):
                result = sensitive_file_inventory.SensitiveFileInventoryCollector().collect(context)

            self.assertEqual(
                CollectorStatus.CANCELLED,
                result.status,
            )

            self.assertEqual(
                [],
                [path for path in context.workspace.rglob("*") if path.is_file() and path != cancellation_file],
            )

    def test_browser_and_media_copy_cancellation_remove_unpublished_bytes(
            self,
    ) -> None:
        """Browser and media collectors remove incomplete private copies."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            local = root / "local"
            roaming = root / "roaming"

            profile = local / "Google" / "Chrome" / "User Data" / "Default"
            profile.mkdir(parents=True)

            (profile / "History").write_bytes(b"history")

            browser_context, browser_cancellation = _context(
                root / "browser-workspace",
                collector_id="core.browser.browser_data_backup",
            )

            def cancel_browser_copy(
                    _source: Path,
                    destination: Path,
                    copied_bytes: int,
            ) -> tuple[Path, int]:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"history")

                browser_cancellation.touch()

                return destination, copied_bytes + 7

            with (
                patch.dict(
                    os.environ,
                    {
                        "LOCALAPPDATA": str(local),
                        "APPDATA": str(roaming),
                    },
                ),
                patch(
                    "core.browser.browser_data_backup._copy_file",
                    side_effect=cancel_browser_copy,
                ),
            ):
                browser_result = browser_data_backup.BrowserDataBackupCollector().collect(browser_context)

            self.assertEqual(
                CollectorStatus.CANCELLED,
                browser_result.status,
            )

            self.assertEqual(
                [],
                [path for path in browser_context.workspace.rglob("*") if
                 path.is_file() and path != browser_cancellation],
            )

            home = root / "home"
            pictures = home / "Pictures"
            pictures.mkdir(parents=True)

            (pictures / "photo.jpg").write_bytes(b"photo")

            media_context, media_cancellation = _context(
                root / "media-workspace",
                collector_id="core.media.media_backup",
            )

            def cancel_media_copy(
                    source: Path,
                    destination: Path,
            ) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())

                media_cancellation.touch()

            with (
                patch.object(
                    media_backup.filesystem_adapter,
                    media_backup.filesystem_adapter.home.__name__,
                    return_value=home,
                ),
                patch.object(
                    media_backup.filesystem_adapter,
                    media_backup.filesystem_adapter.copy_file.__name__,
                    side_effect=cancel_media_copy,
                ),
            ):
                media_result = media_backup.MediaBackupCollector().collect(media_context)

            self.assertEqual(
                CollectorStatus.CANCELLED,
                media_result.status,
            )

            self.assertEqual(
                [],
                [path for path in media_context.workspace.rglob("*") if path.is_file() and path != media_cancellation],
            )

    def test_event_queries_cannot_publish_after_cancellation(self) -> None:
        """Event-log collectors recheck cancellation after PowerShell exits."""

        collectors = (
            (
                application_events.ApplicationEventsCollector,
                "core.event_log.application_events.subprocess.run",
                "core.event_log.application_events",
            ),
            (
                security_events.SecurityEventsCollector,
                "core.event_log.security_events.subprocess.run",
                "core.event_log.security_events",
            ),
            (
                system_events.SystemEventsCollector,
                "core.event_log.system_events.subprocess.run",
                "core.event_log.system_events",
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            for collector_type, subprocess_target, collector_id in collectors:
                with self.subTest(
                        collector=collector_type.__name__,
                ):
                    context, cancellation_file = _context(
                        root / collector_type.__name__,
                        collector_id=collector_id,
                    )

                    def finish_after_cancellation(
                            *_args: Any,
                            **_kwargs: Any,
                    ) -> subprocess.CompletedProcess[str]:
                        cancellation_file.touch()

                        return subprocess.CompletedProcess(
                            args=[],
                            returncode=0,
                            stdout="header\nrow\n",
                            stderr="",
                        )

                    with patch(
                            subprocess_target,
                            side_effect=finish_after_cancellation,
                    ):
                        result = collector_type().collect(context)

                    self.assertEqual(
                        CollectorStatus.CANCELLED,
                        result.status,
                    )

                    self.assertEqual(
                        [],
                        [path for path in context.workspace.rglob("*") if path.is_file() and path != cancellation_file],
                    )

    def test_system_data_and_ssh_copy_cancellation_remove_partial_backups(
            self,
    ) -> None:
        """System and SSH collectors delete private incomplete copies."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            windows = root / "Windows"

            event_log = windows / "System32" / "winevt" / "Logs" / "System.evtx"

            event_log.parent.mkdir(parents=True)
            event_log.write_bytes(b"event log")

            system_context, system_cancellation = _context(
                root / "system-workspace",
                collector_id="core.system.windows_system_data_backup",
            )

            def cancel_system_copy(
                    source: Path,
                    destination: Path,
            ) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())

                system_cancellation.touch()

            with (
                patch.dict(
                    os.environ,
                    {
                        "SystemRoot": str(windows),
                        "ProgramData": str(root / "ProgramData"),
                    },
                ),
                patch(
                    ("core.system.windows_system_data_backup.filesystem_adapter.copy_file"),
                    side_effect=cancel_system_copy,
                ),
            ):
                system_result = windows_system_data_backup.WindowsSystemDataBackupCollector().collect(system_context)

            self.assertEqual(
                CollectorStatus.CANCELLED,
                system_result.status,
            )

            self.assertEqual(
                [],
                [path for path in system_context.workspace.rglob("*") if
                 path.is_file() and path != system_cancellation],
            )

            home = root / "home"

            private_key = home / ".ssh" / "id_test"
            private_key.parent.mkdir(parents=True)
            private_key.write_bytes(b"private key")

            ssh_context, ssh_cancellation = _context(
                root / "ssh-workspace",
                collector_id="core.ssh.ssh_backup",
            )

            original_write = zipfile.ZipFile.write

            def cancel_archive_write(
                    archive: zipfile.ZipFile,
                    filename: str | os.PathLike[str],
                    arcname: str | os.PathLike[str] | None = None,
                    compress_type: int | None = None,
                    compresslevel: int | None = None,
            ) -> None:
                bound_write = original_write.__get__(archive, type(archive))

                bound_write(
                    filename,
                    arcname=arcname,
                    compress_type=compress_type,
                    compresslevel=compresslevel,
                )

                ssh_cancellation.touch()

            with (
                patch.object(
                    ssh_backup.filesystem_adapter,
                    ssh_backup.filesystem_adapter.home.__name__,
                    return_value=home,
                ),
                patch.object(
                    ssh_backup.zipfile.ZipFile,
                    ssh_backup.zipfile.ZipFile.write.__name__,
                    new=cancel_archive_write,
                ),
            ):
                ssh_result = ssh_backup.SshBackupCollector().collect(ssh_context)

            self.assertEqual(
                CollectorStatus.CANCELLED,
                ssh_result.status,
            )

            self.assertFalse((ssh_context.workspace / "ssh_backup.zip").exists())

    def test_packet_and_memory_loops_stop_before_serialization(
            self,
    ) -> None:
        """Packet and memory loops stop before serializing evidence."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            packet_context, packet_cancellation = _context(
                root / "packet",
                {
                    "packet_count": 2,
                    "timeout_seconds": 2,
                    "retry_window_seconds": 0,
                    "interface": "127.0.0.1",
                },
                collector_id="core.packet.packet_capture",
            )

            class FakeSocket:
                """Minimal raw-socket replacement."""

                @staticmethod
                def bind(_address: object) -> None:
                    return None

                @staticmethod
                def setsockopt(
                        *_args: object,
                ) -> None:
                    return None

                @staticmethod
                def ioctl(
                        *_args: object,
                ) -> None:
                    return None

                @staticmethod
                def recvfrom(
                        _buffer_size: int,
                ) -> tuple[bytes, tuple[str, int]]:
                    return b"", ("127.0.0.1", 0)

                @staticmethod
                def close() -> None:
                    return None

            def cancel_wait(
                    *_args: object,
                    **_kwargs: object,
            ) -> tuple[
                list[object],
                list[object],
                list[object],
            ]:
                packet_cancellation.touch()

                return [], [], []

            with (
                patch.object(
                    packet_capture.socket,
                    packet_capture.socket.socket.__name__,
                    return_value=FakeSocket(),
                ),
                patch.object(
                    packet_capture.select,
                    packet_capture.select.select.__name__,
                    side_effect=cancel_wait,
                ),
            ):
                packet_result = packet_capture.PacketCaptureCollector().collect(packet_context)

            self.assertEqual(
                CollectorStatus.CANCELLED,
                packet_result.status,
            )

            self.assertFalse((packet_context.workspace / "packet_capture.csv").exists())

            memory_context, memory_cancellation = _context(
                root / "memory",
                {
                    "max_regions": 10,
                    "output_limit_bytes": 4096,
                    "disk_safety_margin_bytes": 0,
                    "dump_directory": "maps",
                },
                collector_id="core.process.memory_map",
            )

            def cancel_on_current_process() -> int:
                """Signal cancellation as soon as the memory query begins."""
                memory_cancellation.touch()
                return 1

            with (
                patch.object(
                    memory_map,
                    "get_current_process",
                    side_effect=cancel_on_current_process,
                ),
                patch.object(
                    memory_map,
                    "get_process_memory_info",
                    return_value=True,
                ),
                patch.object(
                    memory_map,
                    "virtual_query",
                    return_value=0,
                ),
            ):
                memory_result = memory_map.MemoryMapCollector().collect(memory_context)

            self.assertEqual(
                CollectorStatus.CANCELLED,
                memory_result.status,
            )

            self.assertEqual(
                [],
                [path for path in memory_context.workspace.rglob("*") if
                 path.is_file() and path != memory_cancellation],
            )


if __name__ == "__main__":
    unittest.main()
