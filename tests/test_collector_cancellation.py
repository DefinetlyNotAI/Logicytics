"""Direct cooperative-cancellation tests for long-running collector phases."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.browser import browser_data_backup
from core.event_log import application_events, security_events, system_events
from core.filesystem import sensitive_file_inventory
from core.media import media_backup
from core.packet import packet_capture
from core.process import memory_map
from core.ssh import ssh_backup
from core.system import windows_system_data_backup
from logicytics.contracts import CollectorStatus


class _RejectingArtifacts:
    def register_file(self, *_args, **_kwargs):
        raise AssertionError("cancelled collection must not register unpublished evidence")


class _Context:
    def __init__(self, workspace: Path, settings: dict[str, object] | None = None) -> None:
        self.workspace = workspace
        self.temporary_directory = workspace / "tmp"
        self.settings = settings or {}
        self.artifacts = _RejectingArtifacts()
        self.cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled

    def report_progress(self, _event: str, **_metrics: int | float | str) -> None:
        return None


class CollectorCancellationTests(unittest.TestCase):
    def test_sensitive_copy_cancellation_removes_unpublished_bytes(self) -> None:
        """Cancellation after a concurrent sensitive copy leaves no unregistered evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            (source_root / "password.txt").write_text("sensitive", encoding="utf-8")
            context = _Context(root / "workspace", {"root": str(source_root), "max_directories": 10,
                                                    "max_matches": 10})
            context.workspace.mkdir()

            def cancel_after_copy(source: Path, destination: Path) -> Path:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
                context.cancelled = True
                return destination

            with patch.object(sensitive_file_inventory, "_copy", side_effect=cancel_after_copy):
                result = sensitive_file_inventory.SensitiveFileInventoryCollector().collect(context)

            self.assertEqual(CollectorStatus.CANCELLED, result.status)
            self.assertEqual([], [path for path in context.workspace.rglob("*") if path.is_file()])

    def test_browser_and_media_copy_cancellation_remove_unpublished_bytes(self) -> None:
        """Source-data collectors clean copied bytes when cancellation arrives during a file copy."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            browser_context = _Context(root / "browser-workspace")
            browser_context.workspace.mkdir()
            local = root / "local"
            roaming = root / "roaming"
            profile = local / "Google" / "Chrome" / "User Data" / "Default"
            profile.mkdir(parents=True)
            (profile / "History").write_bytes(b"history")

            def cancel_browser_copy(_source: Path, destination: Path, copied_bytes: int):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"history")
                browser_context.cancelled = True
                return destination, copied_bytes + 7

            with patch.dict(os.environ, {"LOCALAPPDATA": str(local), "APPDATA": str(roaming)}), patch.object(
                    browser_data_backup, "_copy_file", side_effect=cancel_browser_copy):
                browser_result = browser_data_backup.BrowserDataBackupCollector().collect(browser_context)
            self.assertEqual(CollectorStatus.CANCELLED, browser_result.status)
            self.assertEqual([], [path for path in browser_context.workspace.rglob("*") if path.is_file()])

            home = root / "home"
            pictures = home / "Pictures"
            pictures.mkdir(parents=True)
            (pictures / "photo.jpg").write_bytes(b"photo")
            media_context = _Context(root / "media-workspace")
            media_context.workspace.mkdir()

            def cancel_media_copy(source: Path, destination: Path) -> None:
                destination.write_bytes(source.read_bytes())
                media_context.cancelled = True

            with patch.object(media_backup.filesystem_adapter, "home", return_value=home), patch.object(
                    media_backup.filesystem_adapter, "copy_file", side_effect=cancel_media_copy):
                media_result = media_backup.MediaBackupCollector().collect(media_context)
            self.assertEqual(CollectorStatus.CANCELLED, media_result.status)
            self.assertEqual([], [path for path in media_context.workspace.rglob("*") if path.is_file()])

    def test_event_queries_cannot_publish_after_cancellation(self) -> None:
        """All bounded event-log queries recheck cancellation after PowerShell exits."""
        collectors = (
            (application_events, application_events.ApplicationEventsCollector),
            (security_events, security_events.SecurityEventsCollector),
            (system_events, system_events.SystemEventsCollector),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for module, collector_type in collectors:
                with self.subTest(collector=collector_type.__name__):
                    workspace = Path(temporary) / collector_type.__name__
                    workspace.mkdir()
                    context = _Context(workspace)

                    def finish_after_cancellation(*_args, **_kwargs):
                        context.cancelled = True
                        return subprocess.CompletedProcess([], 0, "header\nrow\n", "")

                    with patch.object(module.subprocess, "run", side_effect=finish_after_cancellation):
                        result = collector_type().collect(context)
                    self.assertEqual(CollectorStatus.CANCELLED, result.status)
                    self.assertEqual([], [path for path in workspace.rglob("*") if path.is_file()])

    def test_system_data_and_ssh_copy_cancellation_remove_partial_backups(self) -> None:
        """System-data and archive loops remove private copies that were never registered."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            windows = root / "Windows"
            event_log = windows / "System32" / "winevt" / "Logs" / "System.evtx"
            event_log.parent.mkdir(parents=True)
            event_log.write_bytes(b"event log")
            system_context = _Context(root / "system-workspace")
            system_context.workspace.mkdir()

            def cancel_system_copy(source: Path, destination: Path) -> None:
                destination.write_bytes(source.read_bytes())
                system_context.cancelled = True

            with patch.dict(os.environ, {"SystemRoot": str(windows), "ProgramData": str(root / "ProgramData")}), \
                    patch.object(windows_system_data_backup.filesystem_adapter, "copy_file",
                                 side_effect=cancel_system_copy):
                system_result = windows_system_data_backup.WindowsSystemDataBackupCollector().collect(system_context)
            self.assertEqual(CollectorStatus.CANCELLED, system_result.status)
            self.assertEqual([], [path for path in system_context.workspace.rglob("*") if path.is_file()])

            home = root / "home"
            private_key = home / ".ssh" / "id_test"
            private_key.parent.mkdir(parents=True)
            private_key.write_bytes(b"private key")
            ssh_context = _Context(root / "ssh-workspace")
            ssh_context.workspace.mkdir()
            original_write = ssh_backup.zipfile.ZipFile.write

            def cancel_archive_write(archive, *args, **kwargs):
                result = original_write(archive, *args, **kwargs)
                ssh_context.cancelled = True
                return result

            with patch.object(ssh_backup.filesystem_adapter, "home", return_value=home), patch.object(
                    ssh_backup.zipfile.ZipFile, "write", new=cancel_archive_write):
                ssh_result = ssh_backup.SshBackupCollector().collect(ssh_context)
            self.assertEqual(CollectorStatus.CANCELLED, ssh_result.status)
            self.assertFalse((ssh_context.workspace / "ssh_backup.zip").exists())

    def test_packet_and_memory_loops_stop_before_serialization(self) -> None:
        """Capture and memory enumeration honor cancellation before materializing reports."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet_context = _Context(root / "packet", {"packet_count": 2, "timeout_seconds": 2,
                                                        "retry_window_seconds": 0, "interface": "127.0.0.1"})
            packet_context.workspace.mkdir()

            class FakeSocket:
                def bind(self, _address) -> None:
                    return None

                def setsockopt(self, *_args) -> None:
                    return None

                def ioctl(self, *_args) -> None:
                    return None

                def close(self) -> None:
                    return None

            def cancel_wait(*_args):
                packet_context.cancelled = True
                return [], [], []

            with patch.object(packet_capture.socket, "socket", return_value=FakeSocket()), patch.object(
                    packet_capture.select, "select", side_effect=cancel_wait):
                packet_result = packet_capture.PacketCaptureCollector().collect(packet_context)
            self.assertEqual(CollectorStatus.CANCELLED, packet_result.status)
            self.assertFalse((packet_context.workspace / "packet_capture.csv").exists())

            memory_context = _Context(root / "memory", {"max_regions": 10, "output_limit_bytes": 4096,
                                                        "disk_safety_margin_bytes": 0,
                                                        "dump_directory": "maps"})
            memory_context.workspace.mkdir()

            class FakeKernel:
                def GetCurrentProcess(self):
                    memory_context.cancelled = True
                    return 1

            class FakePsapi:
                @staticmethod
                def GetProcessMemoryInfo(*_args):
                    return 1

            with patch.object(memory_map.ctypes, "WinDLL", side_effect=(FakeKernel(), FakePsapi())):
                memory_result = memory_map.MemoryMapCollector().collect(memory_context)
            self.assertEqual(CollectorStatus.CANCELLED, memory_result.status)
            self.assertEqual([], [path for path in memory_context.workspace.rglob("*") if path.is_file()])


if __name__ == "__main__":
    unittest.main()
