"""Regression coverage for worker process security boundaries."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from logicytics.contracts import (
    Capability,
    RunRequest,
)
from logicytics.module.configuration import (
    default_config,
)
from logicytics.module.discovery import preflight
from logicytics.module.planner import build_plan
from logicytics.module.runtime import RunSupervisor
from tests.fixtures.collectors import COLLECTOR, delayed_collector_source


class WorkerSecurityTests(unittest.TestCase):
    """Worker isolation and explicit capability enforcement."""

    def test_collector_cannot_modify_peer_workspace_or_repository_files(self) -> None:
        """Direct collector writes outside its private evidence roots fail without affecting peers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            attempts = (
                "context.workspace.parent / 'z_independent' / 'compromised.txt'",
                f"Path({str(root / 'repository-compromised.txt')!r})",
            )
            for target in attempts:
                with self.subTest(target=target):
                    source = delayed_collector_source("a_attacker", 0.0).replace(
                        '        output = context.workspace / "system.txt"',
                        f"        ({target}).write_text('compromised', encoding='utf-8')\n"
                        '        output = context.workspace / "system.txt"',
                    )
                    (core_directory / "a_attacker.py").write_text(source, encoding="utf-8")
                    (core_directory / "z_independent.py").write_text(
                        delayed_collector_source("z_independent", 0.0),
                        encoding="utf-8",
                    )
                    plan = build_plan(preflight(root), RunRequest(max_workers=2, acknowledge_authorization=True))
                    outcome = RunSupervisor(root, default_config(root)).run(plan)
                    records = {record.id: record for record in outcome.manifest.collectors}
                    self.assertEqual("failed", records["core.system.a_attacker"].status)
                    self.assertIn(
                        "escapes its private workspace",
                        "\n".join(records["core.system.a_attacker"].errors),
                    )
                    self.assertEqual("succeeded", records["core.system.z_independent"].status)
                    self.assertFalse((root / "repository-compromised.txt").exists())
                    self.assertFalse(
                        (outcome.run_directory / "collectors" / "z_independent" / "compromised.txt").exists())

    def test_collector_cannot_mutate_its_process_environment(self) -> None:
        """A collector must not change process environment or working-directory policy."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace("from pathlib import Path\n", "from pathlib import Path\nimport os\n").replace(
                    '        output = context.workspace / "system.txt"',
                    '        os.environ["LOGICYTICS_COLLECTOR_MUTATION"] = "unexpected"\n        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            self.assertEqual("failed", outcome.manifest.collectors[0].status)
            self.assertIn("os.putenv", "\n".join(outcome.manifest.collectors[0].errors))

    def test_worker_rejects_undeclared_subprocess_network_and_registry_capabilities(self) -> None:
        """Privileged platform access must be declared in metadata, not merely imported."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()

            base = COLLECTOR.replace(
                "from pathlib import Path\n",
                "from pathlib import Path\nimport socket\nimport subprocess\nimport sys\nimport winreg\n",
            )

            attempts = (
                (
                    "subprocess.run([sys.executable, '-c', 'pass'], check=False)",
                    "subprocess capability",
                ),
                (
                    "socket.socket()",
                    "network capability",
                ),
                (
                    "winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Software')",
                    "registry_read capability",
                ),
            )

            for operation, message in attempts:
                with self.subTest(operation=operation):
                    collector_path.write_text(
                        base.replace(
                            '        output = context.workspace / "system.txt"',
                            f'        {operation}\n        output = context.workspace / "system.txt"',
                        ),
                        encoding="utf-8",
                    )

                    plan = build_plan(
                        preflight(root),
                        RunRequest(
                            max_workers=1,
                            acknowledge_authorization=True,
                        ),
                    )

                    outcome = RunSupervisor(
                        root,
                        default_config(root),
                    ).run(plan)

                    record = outcome.manifest.collectors[0]

                    self.assertEqual("failed", record.status)
                    self.assertIn(
                        message,
                        "\n".join(record.errors),
                    )

                    failure = record.failure
                    assert failure is not None

                    self.assertEqual("access", failure["operation"])

                    remediation = failure["remediation"]
                    if not isinstance(remediation, str):
                        self.fail(f"remediation must be str, got {type(remediation).__name__}")

                    self.assertIn("capability", remediation)

                    retry_safe = failure["retry_safe"]
                    if not isinstance(retry_safe, bool):
                        self.fail(f"retry_safe must be bool, got {type(retry_safe).__name__}")

                    self.assertFalse(retry_safe)

    def test_worker_enforces_external_browser_sensitive_and_private_key_read_capabilities(
            self,
    ) -> None:
        """External evidence reads require filesystem access plus every applicable sensitive grant."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            ordinary = root / "outside.txt"
            ordinary.write_text("ordinary evidence", encoding="utf-8")
            browser = root / "Chrome" / "User Data" / "Default" / "Cookies"
            browser.parent.mkdir(parents=True)
            browser.write_text("browser evidence", encoding="utf-8")
            private_key = root / ".ssh" / "id_ed25519"
            private_key.parent.mkdir()
            private_key.write_text("private evidence", encoding="utf-8")
            cases = (
                (ordinary, (), "filesystem_read capability"),
                (ordinary, (Capability.FILESYSTEM_READ,), None),
                (browser, (Capability.FILESYSTEM_READ,), "browser_data capability"),
                (
                    browser,
                    (Capability.FILESYSTEM_READ, Capability.BROWSER_DATA),
                    "sensitive_files capability",
                ),
                (
                    private_key,
                    (Capability.FILESYSTEM_READ, Capability.SENSITIVE_FILES),
                    "private_keys capability",
                ),
                (
                    private_key,
                    (
                        Capability.FILESYSTEM_READ,
                        Capability.SENSITIVE_FILES,
                        Capability.PRIVATE_KEYS,
                    ),
                    None,
                ),
            )
            for evidence, capabilities, denied in cases:
                with self.subTest(evidence=evidence.name, capabilities=capabilities):
                    declared = ", ".join(f"Capability.{capability.name}" for capability in capabilities)
                    if len(capabilities) == 1:
                        declared += ","
                    collector_path.write_text(
                        COLLECTOR.replace(
                            "from pathlib import Path\n",
                            "from pathlib import Path\nfrom logicytics import Capability\n",
                        )
                        .replace(
                            "            capabilities=(),",
                            f"            capabilities=({declared}),",
                        )
                        .replace(
                            '        output = context.workspace / "system.txt"',
                            f"        Path({str(evidence)!r}).read_text(encoding='utf-8')\n"
                            '        output = context.workspace / "system.txt"',
                        ),
                        encoding="utf-8",
                    )
                    plan = build_plan(
                        preflight(root),
                        RunRequest(
                            max_workers=1,
                            acknowledge_authorization=True,
                        ),
                    )
                    outcome = RunSupervisor(root, default_config(root)).run(plan)
                    record = outcome.manifest.collectors[0]
                    if denied is None:
                        self.assertEqual("succeeded", record.status, record.errors)
                    else:
                        self.assertEqual("failed", record.status)
                        self.assertIn(denied, "\n".join(record.errors))
                        self.assertIn(
                            "CAPABILITY_DECLARATION_MISMATCH",
                            "\n".join(record.errors),
                        )

    def test_worker_rejects_raw_packet_socket_without_packet_capture_capability(self) -> None:
        """General network approval must not silently authorize raw packet capture."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    "from pathlib import Path\n",
                    "from pathlib import Path\nimport socket\nfrom logicytics import Capability\n",
                )
                .replace(
                    "            capabilities=(),",
                    "            capabilities=(Capability.NETWORK,),\n            network_access=NetworkAccess.LOCAL,",
                )
                .replace(
                    '        output = context.workspace / "system.txt"',
                    "        socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)\n"
                    '        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )
            plan = build_plan(
                preflight(root),
                RunRequest(
                    max_workers=1,
                    acknowledge_authorization=True,
                    approved_capabilities=(Capability.NETWORK,),
                ),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            self.assertEqual("failed", outcome.manifest.collectors[0].status)
            self.assertIn("packet_capture capability", "\n".join(outcome.manifest.collectors[0].errors))

    def test_worker_allows_an_explicitly_declared_subprocess_by_default(self) -> None:
        """A declared subprocess capability is usable without a separate approval flag."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    "from pathlib import Path\n",
                    "from pathlib import Path\nimport subprocess\nimport sys\nfrom logicytics import Capability\n",
                )
                .replace(
                    "            capabilities=(),",
                    "            capabilities=(Capability.SUBPROCESS,),",
                )
                .replace(
                    '        output = context.workspace / "system.txt"',
                    '        subprocess.run([sys.executable, "-c", "pass"], check=True)\n        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )
            plan = build_plan(
                preflight(root),
                RunRequest(
                    max_workers=1,
                    acknowledge_authorization=True,
                ),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            self.assertEqual("succeeded", outcome.manifest.collectors[0].status)

    def test_worker_blocks_application_update_power_and_peer_collector_commands_without_stopping_peers(
            self,
    ) -> None:
        """Subprocess approval cannot escape the collector role or affect an independent worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            peer_path = core_directory / "z_independent.py"
            peer_path.write_text(delayed_collector_source("z_independent", 0.0), encoding="utf-8")
            protected_configuration = root / "logicytics.yaml"
            attempts = (
                ([sys.executable, "-m", "logicytics", "preflight"], "main_application"),
                ([sys.executable, "-m", "pip", "--version"], "repository_or_package_management"),
                (["git", "--version"], "repository_or_package_management"),
                (["shutdown", "/?"], "system_power"),
                (["cmd", "/c", "shutdown", "/?"], "system_power"),
                (
                    [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(protected_configuration)!r}).write_text('bad')",
                    ],
                    "configuration_mutation",
                ),
                ([sys.executable, str(peer_path)], "collector_launch"),
            )
            for command, category in attempts:
                with self.subTest(command=command):
                    attacker = (
                        delayed_collector_source("a_attacker", 0.0)
                        .replace(
                            "from pathlib import Path\n",
                            "from pathlib import Path\nimport subprocess\nfrom logicytics import Capability\n",
                        )
                        .replace(
                            "            capabilities=(),",
                            "            capabilities=(Capability.SUBPROCESS,),",
                        )
                        .replace(
                            '        output = context.workspace / "system.txt"',
                            f'        subprocess.run({command!r}, check=True)\n        output = context.workspace / "system.txt"',
                        )
                    )
                    (core_directory / "a_attacker.py").write_text(attacker, encoding="utf-8")
                    report = preflight(root)
                    self.assertEqual((), report.invalid)
                    plan = build_plan(
                        report,
                        RunRequest(
                            max_workers=2,
                            acknowledge_authorization=True,
                            approved_capabilities=(Capability.SUBPROCESS,),
                        ),
                    )
                    outcome = RunSupervisor(root, default_config(root)).run(plan)
                    records = {record.id: record for record in outcome.manifest.collectors}

                    self.assertEqual("failed", records["core.system.a_attacker"].status)
                    self.assertIn(
                        f"collector subprocess command is prohibited: {category}",
                        "\n".join(records["core.system.a_attacker"].errors),
                    )
                    self.assertEqual("succeeded", records["core.system.z_independent"].status)
                    self.assertEqual("partial", outcome.manifest.status.value)
                    self.assertFalse(protected_configuration.exists())

    def test_preflight_rejects_collector_imports_of_application_and_orchestration_services(
            self,
    ) -> None:
        """Collectors may import public contracts but never the main application control surface."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            attempts = (
                "import logicytics.cli",
                "from logicytics import run_collection",
                "from logicytics import packaging",
                "import logicytics\nlogicytics.run_collection",
                "import logicytics.module.configuration",
                "from logicytics.module.api import query_run",
            )
            for statement in attempts:
                with self.subTest(statement=statement):
                    collector_path.write_text(
                        COLLECTOR.replace("from pathlib import Path\n", f"from pathlib import Path\n{statement}\n"),
                        encoding="utf-8",
                    )
                    report = preflight(root)
                    self.assertEqual(1, len(report.invalid))
                    self.assertIn("forbidden application import", "\n".join(report.invalid[0].static_errors))
                    diagnostic = next(item for item in report.invalid[0].diagnostics if
                                      "forbidden application import" in item.message)
                    self.assertEqual("static.engine_boundary", diagnostic.rule)


if __name__ == "__main__":
    unittest.main()
