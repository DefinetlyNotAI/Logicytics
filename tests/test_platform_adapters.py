"""Contract checks for centralized, injectable host platform seams."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from logicytics.platform_adapters import NetworkAdapter, ProcessAdapter, RegistryAdapter


class ProcessAdapterTests(unittest.TestCase):
    def test_process_adapter_normalizes_and_delegates_shell_free_commands(self) -> None:
        completed = subprocess.CompletedProcess(("tool", "argument"), 0, "output", "")
        with patch("logicytics.platform_adapters.subprocess.run", return_value=completed) as invoke:
            result = ProcessAdapter().run(["tool", Path("argument")], capture_output=True,
                                          check=False, text=True, timeout=5)
        self.assertIs(completed, result)
        invoke.assert_called_once_with(("tool", "argument"), capture_output=True,
                                       check=False, text=True, timeout=5)

    def test_process_adapter_rejects_shell_empty_and_unbounded_timeout_inputs(self) -> None:
        adapter = ProcessAdapter()
        for command, options, message in (
            ([], {}, "at least one"),
            (["tool"], {"shell": True}, "shell"),
            (["tool"], {"timeout": 0}, "timeout"),
        ):
            with self.subTest(command=command, options=options):
                with self.assertRaisesRegex(ValueError, message):
                    adapter.run(command, **options)

    def test_core_collectors_cannot_import_the_process_module_directly(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        offenders = [
            str(path.relative_to(project_root))
            for path in (project_root / "core").rglob("*.py")
            if "import subprocess" in path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([], offenders)

    def test_registry_adapter_delegates_only_read_operations(self) -> None:
        adapter = RegistryAdapter()
        fake = Mock()
        fake.OpenKey.return_value = "key"
        fake.QueryValueEx.return_value = ("value", 1)
        fake.EnumKey.return_value = "child"
        fake.EnumValue.return_value = ("name", "value", 1)
        with patch("logicytics.platform_adapters._winreg", fake):
            self.assertEqual("key", adapter.OpenKey(1, "Software"))
            self.assertEqual(("value", 1), adapter.QueryValueEx("key", "Name"))
            self.assertEqual("child", adapter.EnumKey("key", 0))
            self.assertEqual(("name", "value", 1), adapter.EnumValue("key", 0))

    def test_core_collectors_cannot_import_winreg_directly(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        offenders = [
            str(path.relative_to(project_root))
            for path in (project_root / "core").rglob("*.py")
            if "import winreg" in path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([], offenders)

    def test_network_adapter_delegates_host_and_socket_operations(self) -> None:
        adapter = NetworkAdapter()
        fake_socket = Mock()
        fake_socket.gethostname.return_value = "host"
        fake_socket.gethostbyname.return_value = "192.0.2.1"
        fake_socket.getaddrinfo.return_value = [(2, 1, 6, "", ("192.0.2.1", 0))]
        fake_socket.inet_ntoa.return_value = "198.51.100.1"
        fake_socket.socket.return_value = "socket"
        with patch("logicytics.platform_adapters._socket", fake_socket):
            self.assertEqual("host", adapter.gethostname())
            self.assertEqual("192.0.2.1", adapter.gethostbyname("host"))
            self.assertTrue(adapter.getaddrinfo("host", None))
            self.assertEqual("198.51.100.1", adapter.inet_ntoa(b"\x00" * 4))
            self.assertEqual("socket", adapter.socket(2, 3, 4))

    def test_core_collectors_cannot_import_socket_directly(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        offenders = [
            str(path.relative_to(project_root))
            for path in (project_root / "core").rglob("*.py")
            if "import socket" in path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
