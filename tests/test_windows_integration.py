"""Bounded read-only Windows integration checks for v4 platform and output contracts."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from logicytics.artifacts import WorkspaceArtifactWriter
from logicytics.discovery import preflight
from logicytics.output_contracts import core_output_contract
from logicytics.platform_adapters import (
    process_adapter, registry_adapter, which, windows_api_adapter,
)


@unittest.skipUnless(os.name == "nt", "Windows integration tests require Windows")
class WindowsIntegrationTests(unittest.TestCase):
    def test_read_only_windows_platform_integrations_are_bounded(self) -> None:
        """Exercise live privilege, PowerShell, registry, WMI, event, disk, and network boundaries."""
        privilege = windows_api_adapter.is_administrator()
        self.assertIn(privilege, {True, False, None})

        with registry_adapter.OpenKey(
            registry_adapter.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
        ) as key:
            product_name, _ = registry_adapter.QueryValueEx(key, "ProductName")
        self.assertTrue(str(product_name).strip())

        powershell = which("powershell")
        self.assertIsNotNone(powershell)
        probes = {
            "powershell": [powershell, "-NoProfile", "-NonInteractive", "-Command", "Get-ExecutionPolicy"],
            "wmi": [powershell, "-NoProfile", "-NonInteractive", "-Command",
                    "Get-CimInstance Win32_OperatingSystem | Select-Object -First 1 Caption | ConvertTo-Json -Compress"],
            "event_log": [powershell, "-NoProfile", "-NonInteractive", "-Command",
                          "Get-WinEvent -LogName System -MaxEvents 1 | Select-Object Id | ConvertTo-Json -Compress"],
        }
        ipconfig = which("ipconfig")
        self.assertIsNotNone(ipconfig)
        probes["network"] = [ipconfig, "/all"]
        manage_bde = which("manage-bde")
        if manage_bde is not None:
            probes["bitlocker"] = [manage_bde, "-status"]
        for name, command in probes.items():
            with self.subTest(integration=name):
                result = process_adapter.run(
                    command, capture_output=True, check=False, text=True, timeout=30
                )
                self.assertIsInstance(result.returncode, int)
                self.assertLessEqual(len(result.stdout.encode("utf-8")), process_adapter.maximum_capture_bytes)
                self.assertLessEqual(len(result.stderr.encode("utf-8")), process_adapter.maximum_capture_bytes)
                self.assertTrue(result.stdout.strip() or result.stderr.strip() or result.returncode == 0)

        sysinternals = [which(name) for name in ("autorunsc", "handle", "pslist")]
        for executable in (path for path in sysinternals if path is not None):
            with self.subTest(integration="sysinternals", executable=executable):
                result = process_adapter.run(
                    [executable, "-?"], capture_output=True, check=False, text=True, timeout=20
                )
                self.assertIsInstance(result.returncode, int)

    def test_every_shipped_output_contract_publishes_on_windows(self) -> None:
        """Publish one representative artifact for every canonical core path and MIME contract."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for candidate in report.valid:
                if candidate.kind.value != "core" or candidate.metadata is None:
                    continue
                with self.subTest(collector=candidate.metadata.id):
                    contract = core_output_contract(candidate.metadata)
                    workspace = root / "work" / candidate.metadata.id.replace(".", "_")
                    artifact_root = root / "artifacts" / candidate.metadata.id.replace(".", "_")
                    workspace.mkdir(parents=True)
                    artifact_root.mkdir(parents=True)
                    writer = WorkspaceArtifactWriter(
                        candidate.metadata.id,
                        workspace,
                        artifact_root,
                        1024 * 1024,
                        len(contract.workspace_patterns),
                        allowed_relative_paths=contract.workspace_patterns,
                        allowed_media_types=contract.media_types,
                    )
                    for pattern, media_type in zip(contract.workspace_patterns, contract.media_types):
                        parts: list[str] = []
                        for part in Path(pattern).parts:
                            if part == "**":
                                parts.extend(("sample", "evidence.bin"))
                            else:
                                parts.append(part.replace("*", "sample"))
                        source = workspace.joinpath(*parts)
                        source.parent.mkdir(parents=True, exist_ok=True)
                        source.write_bytes(b"golden evidence\n")
                        artifact = writer.register_file(source, media_type=media_type)
                        self.assertIn(artifact.relative_path, {
                            f"{candidate.metadata.id.replace('.', '_')}/{source.relative_to(workspace).as_posix()}"
                        })


if __name__ == "__main__":
    unittest.main()
