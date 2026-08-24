"""Integration-style checks for the v4 core without shipped collectors."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from logicytics.artifacts import WorkspaceArtifactWriter
from logicytics.command_runner import parse_level_messages, run_command
from logicytics.cli import _parser
from logicytics.file_listing import list_files
from logicytics.logging import deprecated, raise_logged, timed
from logicytics.sysinternals import ensure_sysinternals
from logicytics.configuration import default_config, load_config
from logicytics.contracts import Capability, CollectorMetadata, RunRequest, Specialty
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
    def test_sysinternals_archive_lifecycle_honors_ignore_and_extracts_safely(self) -> None:
        """The local bundle must honor opt-out and extract only within its target directory."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".ignore-sysinternals").write_text("", encoding="utf-8")
            self.assertEqual("ignored", ensure_sysinternals(root).status)
            (root / ".ignore-sysinternals").unlink()
            with zipfile.ZipFile(root / "SysinternalsSuite.zip", "w") as archive:
                archive.writestr("PsInfo.exe", "fixture")
            state = ensure_sysinternals(root)
            self.assertEqual("extracted", state.status)
            self.assertTrue((state.extraction_directory / "PsInfo.exe").is_file())

    def test_custom_specialty_is_plugin_only(self) -> None:
        """Plugins may extend specialties, but core metadata remains on the closed set."""
        metadata = CollectorMetadata(
            id="plugin.example", name="Example", version="4.0.0", specialty=Specialty.SYSTEM,
            description="Test metadata.", author="Test",
        ).to_dict()
        metadata["specialty"] = "evidence_graph"
        self.assertEqual("evidence_graph", CollectorMetadata.from_dict(metadata, allow_custom_specialty=True).specialty)
        with self.assertRaises(ValueError):
            CollectorMetadata.from_dict(metadata)

    def test_collector_metadata_rejects_invalid_identity_and_limits(self) -> None:
        """Collector metadata must be a complete typed declaration rather than free text."""
        common = dict(
            id="core.system.example", name="Example", version="4.0.0", specialty=Specialty.SYSTEM,
            description="Example collector.", author="tests",
        )
        with self.assertRaisesRegex(ValueError, "id has an invalid schema"):
            CollectorMetadata(**{**common, "id": "example"})
        with self.assertRaisesRegex(ValueError, "semantic versioning"):
            CollectorMetadata(**{**common, "version": "four"})
        with self.assertRaisesRegex(ValueError, "maximum_artifact_files"):
            CollectorMetadata(**{**common, "maximum_artifact_files": 0})

    def test_deprecation_decorator_logs_removal_context(self) -> None:
        """Deprecated functions must preserve behavior while reporting removal context."""
        events: list[tuple[str, str, dict[str, object]]] = []
        class Logger:
            def event(self, level: str, message: str, **fields: object) -> None:
                events.append((level, message, fields))
        @deprecated(Logger(), removal_version="5.0", reason="replacement exists")
        def old() -> str:
            return "still works"
        self.assertEqual("still works", old())
        self.assertEqual("function_deprecated", events[0][1])
        self.assertEqual("5.0", events[0][2]["removal_version"])

    def test_exception_helper_logs_before_raising(self) -> None:
        """Exception helpers must preserve the requested exception type and context."""
        events: list[tuple[str, str, dict[str, object]]] = []
        class Logger:
            def event(self, level: str, message: str, **fields: object) -> None:
                events.append((level, message, fields))
        with self.assertRaises(ValueError):
            raise_logged(Logger(), ValueError, "invalid setting", setting="workers")
        self.assertEqual("exception", events[0][0])
        self.assertEqual("ValueError", events[0][2]["exception_type"])

    def test_timed_decorator_records_function_lifecycle(self) -> None:
        """Timing instrumentation must report start and finish through EventLogger."""
        events: list[tuple[str, str, dict[str, object]]] = []
        class Logger:
            def event(self, level: str, message: str, **fields: object) -> None:
                events.append((level, message, fields))
        @timed(Logger())
        def add(left: int, right: int) -> int:
            return left + right
        self.assertEqual(3, add(1, 2))
        self.assertEqual(["function_started", "function_finished"], [event[1] for event in events])

    def test_file_listing_filters_and_normalizes_files(self) -> None:
        """Recursive file discovery must filter extensions and excluded directories."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "collectors").mkdir()
            (root / "collectors" / "valid.py").write_text("", encoding="utf-8")
            (root / "collectors" / "ignore.txt").write_text("", encoding="utf-8")
            (root / ".venv").mkdir()
            (root / ".venv" / "hidden.py").write_text("", encoding="utf-8")
            files = list_files(root, extensions=(".py",), excluded_directories=(".venv",))
            self.assertEqual((root / "collectors" / "valid.py",), files)

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

    def test_run_parser_accepts_performance_check(self) -> None:
        """The run command must expose the performance mode used by the request builder."""
        arguments = _parser().parse_args(["run", "--performance-check"])
        self.assertTrue(arguments.performance_check)

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
            writer = WorkspaceArtifactWriter("core.system.test", workspace, artifact_root, 1024, 1)
            with self.assertRaises(ArtifactError):
                writer.register_file(outside)

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
            self.assertIsNotNone(outcome.manifest.collectors[0].heartbeat_at)
            self.assertIsNotNone(outcome.manifest.collectors[0].last_progress_at)
            self.assertFalse((outcome.run_directory / "collectors" / "core_system_system_info" / "tmp").exists())
            package_path, hash_path = package_run(outcome)
            self.assertTrue(package_path.is_file())
            self.assertTrue(hash_path.is_file())
            self.assertEqual(f"{package_path.name}", hash_path.read_text(encoding="ascii").split()[1])
            with zipfile.ZipFile(package_path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("manifest.json", archive.namelist())
                self.assertIn("summary.txt", archive.namelist())
                self.assertEqual(1, len([name for name in archive.namelist() if name.startswith("artifacts/")]))

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

    def test_preflight_rejects_wrong_collection_estimate_type(self) -> None:
        """Optional estimates use the same strict typed contract as lifecycle methods."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    "    def cleanup(self, context: CollectorContext) -> None:",
                    "    def estimate(self, context: CollectorContext) -> ValidationResult:\n"
                    "        \"\"\"Return an invalid estimate type for this test.\"\"\"\n"
                    "        return ValidationResult(valid=True)\n\n"
                    "    def cleanup(self, context: CollectorContext) -> None:",
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("estimate must return CollectionEstimate", report.invalid[0].static_errors)

    def test_preflight_rejects_invalid_dependencies_contract(self) -> None:
        """Optional dependency declarations must remain typed class-level metadata."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    "    def cleanup(self, context: CollectorContext) -> None:",
                    "    def dependencies(self) -> tuple[str, ...]:\n"
                    "        \"\"\"Return an invalid instance-level dependency declaration.\"\"\"\n"
                    "        return ()\n\n"
                    "    def cleanup(self, context: CollectorContext) -> None:",
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("dependencies must be a classmethod", report.invalid[0].static_errors)

    def test_preflight_rejects_dependency_declaration_mismatch(self) -> None:
        """The planner-facing metadata and optional dependency method must agree."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    "    def cleanup(self, context: CollectorContext) -> None:",
                    "    @classmethod\n"
                    "    def dependencies(cls) -> tuple[str, ...]:\n"
                    "        \"\"\"Return a dependency absent from metadata for this test.\"\"\"\n"
                    "        return ('core.system.missing',)\n\n"
                    "    def cleanup(self, context: CollectorContext) -> None:",
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("dependencies() must match metadata.dependencies", report.invalid[0].runtime_error or "")

    def test_preflight_rejects_malformed_validate_result(self) -> None:
        """The isolated preflight probe must reject a non-ValidationResult response."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace("return ValidationResult(True)", "return CollectorResult.succeeded('invalid')"),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("validate() must return ValidationResult", report.invalid[0].runtime_error or "")

    def test_preflight_rejects_collector_print_calls(self) -> None:
        """Collectors must emit structured events rather than write directly to stdout."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace("return CollectorResult.succeeded(\"test artifact created\", (artifact,))", "print('unexpected output')\n        return CollectorResult.succeeded(\"test artifact created\", (artifact,))"),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("collectors must not print; use structured progress or logging", report.invalid[0].static_errors)

    def test_preflight_rejects_import_time_calls_in_headers_and_class_body(self) -> None:
        """Collection-like work must not run before a worker has isolated the collector."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    "class SystemInfoCollector(CoreCollector):",
                    "class SystemInfoCollector(CoreCollector, type(open('unexpected.txt'))):",
                ).replace(
                    "    @classmethod\n    def metadata",
                    "    marker = input('unexpected prompt')\n\n    @classmethod\n    def metadata",
                ).replace(
                    "def validate(self, context: CollectorContext) -> ValidationResult:",
                    "def validate(self, context: CollectorContext = open('another.txt')) -> ValidationResult:",
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            errors = report.invalid[0].static_errors
            self.assertTrue(any("forbidden import-time call: type" in error for error in errors))
            self.assertTrue(any("forbidden import-time call: input" in error for error in errors))
            self.assertTrue(any("forbidden import-time call: open" in error for error in errors))

    def test_preflight_probe_uses_a_restricted_environment(self) -> None:
        """Validation probes expose only documented preflight variables to collectors."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            source = _COLLECTOR.replace(
                "from pathlib import Path",
                "import os\nfrom pathlib import Path",
            ).replace(
                "        return ValidationResult(True)",
                "        if os.environ.get('LOGICYTICS_VALIDATION') != '1':\n"
                "            return ValidationResult(False, reasons=('validation flag missing',))\n"
                "        if os.environ.get('LOGICYTICS_TEST_SECRET') is not None:\n"
                "            return ValidationResult(False, reasons=('caller environment leaked',))\n"
                "        return ValidationResult(True)",
                1,
            )
            collector_path.write_text(source, encoding="utf-8")
            with patch.dict("os.environ", {"LOGICYTICS_TEST_SECRET": "must-not-leak"}):
                report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)

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
