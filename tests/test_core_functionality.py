"""Integration-style checks for the v4 core without shipped collectors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from logicytics.artifacts import WorkspaceArtifactWriter
from logicytics.command_runner import parse_level_messages, run_command
from logicytics.configuration import default_config, load_config
from logicytics.contracts import Capability, RunRequest
from logicytics.discovery import preflight
from logicytics.errors import ArtifactError, PlanError, PreflightError
from logicytics.packaging import package_run
from logicytics.planner import build_plan
from logicytics.runtime import RunSupervisor


_COLLECTOR = '''"""Create a harmless test artifact."""

from pathlib import Path

from logicytics import CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext


class SystemInfoCollector(CoreCollector):
    """A valid test core collector."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Return test metadata."""
        return CollectorMetadata(
            id="core.system.system_info",
            name="System info",
            version="1.0.0",
            specialty=Specialty.SYSTEM,
            description="Creates a harmless text artifact for core tests.",
            author="tests",
            supported_platforms=("win32",),
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Validate the test collector."""
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Write and register a harmless artifact."""
        output = context.workspace / "system.txt"
        output.write_text("ok\\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        return CollectorResult.succeeded("test artifact created", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release test resources."""
'''


class CoreFunctionalityTests(unittest.TestCase):
    def test_command_runner_captures_output_and_parses_structured_levels(self) -> None:
        """Core command execution must avoid a shell and preserve structured output."""
        result = run_command(("python", "-c", "print('INFO: collected'); print('ordinary')"))
        self.assertEqual(0, result.returncode)
        self.assertEqual((("INFO", "collected"),), parse_level_messages(result.stdout))

    def test_configuration_schema_version_is_enforced(self) -> None:
        """Only the v4 configuration schema may be loaded for a v4 run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            config_path.write_text('{"schema_version": 3}', encoding="utf-8")
            with self.assertRaises(PlanError):
                load_config(root)
            config_path.write_text('{"schema_version": 4, "collectors": {}}', encoding="utf-8")
            self.assertEqual(4, load_config(root).schema_version)

    """Validate the foundation before real core collectors are added."""

    def test_artifacts_cannot_escape_collector_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifact_root = root / "artifacts"
            workspace.mkdir()
            artifact_root.mkdir()
            outside = root / "outside.txt"
            outside.write_text("no", encoding="utf-8")
            writer = WorkspaceArtifactWriter("core.system.test", workspace, artifact_root, 1024)
            with self.assertRaises(ArtifactError):
                writer.register_file(outside)

    def test_preflight_plan_run_and_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            plan = build_plan(
                report,
                RunRequest(profile="standard", max_workers=1, acknowledge_authorization=True),
            )
            configuration = default_config(root)
            outcome = RunSupervisor(root, configuration).run(plan)
            self.assertEqual("succeeded", outcome.manifest.status.value)
            self.assertEqual(1, len(outcome.manifest.artifact_list()))
            package_path, hash_path = package_run(outcome)
            self.assertTrue(package_path.is_file())
            self.assertTrue(hash_path.is_file())

    def test_invalid_core_blocks_a_run_but_unselected_plugin_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_path = root / "core" / "system" / "system_info.py"
            core_path.parent.mkdir(parents=True)
            core_path.write_text('"""Invalid core collector."""\n', encoding="utf-8")
            plugin_path = root / "plugins" / "broken_plugin.py"
            plugin_path.parent.mkdir()
            plugin_path.write_text('"""Invalid plugin collector."""\n', encoding="utf-8")
            report = preflight(root)
            self.assertEqual(2, len(report.invalid))
            with self.assertRaises(PreflightError):
                build_plan(report, RunRequest())

    def test_invalid_unselected_plugin_does_not_block_core_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_path = root / "core" / "system" / "system_info.py"
            core_path.parent.mkdir(parents=True)
            core_path.write_text(_COLLECTOR, encoding="utf-8")
            plugin_path = root / "plugins" / "broken_plugin.py"
            plugin_path.parent.mkdir()
            plugin_path.write_text('"""Invalid plugin collector."""\n', encoding="utf-8")
            report = preflight(root)
            plan = build_plan(report, RunRequest())
            self.assertEqual(
                ["core.system.system_info"],
                [candidate.metadata.id for candidate in plan.collectors],
            )

    def test_preflight_rejects_wrong_lifecycle_return_type(self) -> None:
        """Lifecycle annotations must match the strict collector contract exactly."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace("def collect(self, context: CollectorContext) -> CollectorResult:", "def collect(self, context: CollectorContext) -> ValidationResult:"),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("collect must return CollectorResult", report.invalid[0].static_errors)

    def test_capability_gate_requires_explicit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            secured_collector = _COLLECTOR.replace(
                "from logicytics import CollectorMetadata",
                "from logicytics import Capability, CollectorMetadata",
            ).replace(
                'supported_platforms=("win32",),',
                'supported_platforms=("win32",), capabilities=(Capability.FILESYSTEM_READ,),',
            )
            collector_path.write_text(secured_collector, encoding="utf-8")
            report = preflight(root)
            with self.assertRaises(PlanError):
                build_plan(report, RunRequest())
            plan = build_plan(
                report,
                RunRequest(approved_capabilities=(Capability.FILESYSTEM_READ,)),
            )
            self.assertEqual(1, len(plan.collectors))


if __name__ == "__main__":
    unittest.main()
