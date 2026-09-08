from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logicytics.cli import CLI, main
from logicytics.contracts import (
    Capability,
    RunRequest,
)
from logicytics.module import (
    discovery,
)
from logicytics.module.configuration import (
    load_config,
)
from logicytics.module.discovery import preflight
from logicytics.module.environment import EnvironmentReport
from logicytics.module.errors import PlanError, PreflightError
from logicytics.module.planner import build_plan
from logicytics.module.runtime import RunSupervisor
from logicytics.platform_adapters import ProcessAdapter
from tests.fixtures.collectors import COLLECTOR, delayed_collector_source, plugin_collector_source


class PreflightTests(unittest.TestCase):
    """Static/runtime preflight validation and quarantine behavior."""

    def test_preflight_allows_global_ctypes_collector_infrastructure(self) -> None:
        """Collectors may use the public ctypes export without accessing the application layer."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    "from pathlib import Path\n",
                    "from pathlib import Path\nfrom logicytics import ctypes_collector\n",
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            self.assertEqual((), report.invalid)

    def test_core_preflight_requires_explicit_capability_metadata(self) -> None:
        """A core script missing its capability declaration is rejected before execution."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace("            capabilities=(),\n", ""),
                encoding="utf-8",
            )

            report = preflight(root)

            self.assertEqual((), report.valid)
            self.assertEqual(1, len(report.invalid))
            self.assertIn(
                "CAPABILITY_METADATA_MISSING",
                report.invalid[0].static_errors[0],
            )

    def test_plugin_preflight_requires_explicit_security_cost_and_output_metadata(self) -> None:
        """A plugin cannot silently inherit fields that affect consent or scheduling."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_path = root / "plugins" / "example_plugin.py"
            plugin_path.parent.mkdir(parents=True)
            source = plugin_collector_source().replace("SystemInfoCollector", "ExamplePluginCollector")
            source = source.replace("core.system.system_info", "plugin.example_plugin")
            plugin_path.write_text(
                source.replace("            network_access=NetworkAccess.NONE,\n", ""),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn(
                "plugin metadata must explicitly declare: network_access",
                report.invalid[0].static_errors,
            )

    def test_preflight_cache_requires_source_interpreter_contract_and_configuration_identity(
        self,
    ) -> None:
        """Only an exact validation context may reuse an isolated runtime probe."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(COLLECTOR, encoding="utf-8")

            initial = preflight(root, configuration_hash="configuration-a")
            self.assertEqual(1, len(initial.valid), initial.invalid)

            with patch.object(
                ProcessAdapter,
                ProcessAdapter.run.__name__,
                wraps=ProcessAdapter.run,
            ) as probe:
                cached = preflight(root, configuration_hash="configuration-a")
                self.assertEqual(1, len(cached.valid), cached.invalid)
                probe.assert_not_called()

            with patch.object(
                ProcessAdapter,
                ProcessAdapter.run.__name__,
                wraps=ProcessAdapter.run,
            ) as probe:
                invalidated = preflight(root, configuration_hash="configuration-b")
                self.assertEqual(1, len(invalidated.valid), invalidated.invalid)
                self.assertEqual(1, probe.call_count)

    def test_preflight_reports_collector_progress_before_and_after_validation(self) -> None:
        """Long-running isolated probes expose their current collector to callers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            progress: list[tuple[str, int, int, str]] = []

            report = preflight(root, progress=lambda phase, checked, total, current: progress.append((phase, checked, total, current)))

            self.assertEqual(1, len(report.valid), report.invalid)
            self.assertEqual(("checking", 0, 1, "core.system_info"), progress[0])
            self.assertEqual(("checked", 1, 1, "core.system.system_info"), progress[-1])

    def test_preflight_requires_exact_declared_artifact_media_types(self) -> None:
        """A collector cannot disguise or omit the output contract used by registration."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(
                COLLECTOR.replace(
                    'output_media_types=("text/plain",)',
                    'output_media_types=("application/json",)',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertTrue(
                any("include every" in error for error in report.invalid[0].static_errors),
                report.invalid[0].static_errors,
            )

            collector_path.write_text(
                COLLECTOR.replace('            output_media_types=("text/plain",),\n', ""),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertTrue(
                any("explicitly declare" in error for error in report.invalid[0].static_errors),
                report.invalid[0].static_errors,
            )

    def test_preflight_rejects_unregistered_collector_resource_classes(self) -> None:
        """Invalid scheduling metadata blocks shipped collectors before any run starts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            resource_class="disk_heavy",',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("resource_class", report.invalid[0].runtime_error or "")
            diagnostic = report.invalid[0].diagnostics[0]
            self.assertEqual("runtime.contract", diagnostic.rule)
            self.assertEqual(str(collector_path), diagnostic.path)
            assert diagnostic.line is not None
            self.assertGreater(diagnostic.line, 1)
            self.assertIn("resource_class", diagnostic.message)
            with self.assertRaises(PreflightError):
                build_plan(report, RunRequest())

    def test_preflight_rejects_sensitive_default_profile_collector(self) -> None:
        """A shipped collector cannot silently introduce sensitive default evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(
                COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            sensitive_data_categories=("credentials",),',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("sensitive collectors", report.invalid[0].runtime_error or "")
            with self.assertRaises(PreflightError):
                build_plan(report, RunRequest())

    def test_run_output_budget_bounds_concurrent_collector_evidence(self) -> None:
        """Concurrent isolated workers must never exceed the configured run-wide cap."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            for filename in ("a_first", "b_second"):
                source = delayed_collector_source(filename, 0.1).replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            maximum_output_bytes=4,',
                )
                (core_directory / f"{filename}.py").write_text(source, encoding="utf-8")
            config_path = root / "logicytics.yaml"
            config_path.write_text(
                '{"schema_version":4,"runtime":{"maximum_run_output_bytes":7}}',
                encoding="utf-8",
            )
            configuration = load_config(root)
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, configuration).run(plan)

            records = {record.id: record for record in outcome.manifest.collectors}
            self.assertEqual(
                "succeeded",
                records["core.system.a_first"].status,
                records["core.system.a_first"].errors,
            )
            self.assertEqual("failed", records["core.system.b_second"].status)
            self.assertTrue(any("maximum_run_output_bytes" in error for error in records["core.system.b_second"].errors))
            self.assertLessEqual(outcome.manifest.total_artifact_bytes, 7)

    def test_exhausted_run_output_budget_skips_unstarted_collectors(self) -> None:
        """A completely consumed run quota prevents later workers from launching."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            for filename in ("a_first", "b_second"):
                source = delayed_collector_source(filename, 0.0).replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            maximum_output_bytes=4,',
                )
                (core_directory / f"{filename}.py").write_text(source, encoding="utf-8")
            config_path = root / "logicytics.yaml"
            config_path.write_text(
                '{"schema_version":4,"runtime":{"maximum_run_output_bytes":4}}',
                encoding="utf-8",
            )
            configuration = load_config(root)
            plan = build_plan(preflight(root), RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, configuration).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertEqual(
                "succeeded",
                records["core.system.a_first"].status,
                records["core.system.a_first"].errors,
            )
            second = records["core.system.b_second"]
            self.assertEqual("skipped", second.status)
            self.assertIsNone(second.started_at)
            self.assertTrue(any("maximum_run_output_bytes" in error for error in second.errors))
            self.assertEqual(4, outcome.manifest.total_artifact_bytes)

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
            core_path.write_text(COLLECTOR, encoding="utf-8")

            plugin_path = root / "plugins" / "broken_plugin.py"
            plugin_path.parent.mkdir()
            plugin_path.write_text(
                '"""Invalid plugin collector."""\n',
                encoding="utf-8",
            )

            report = preflight(root)
            payload = report.to_dict()

            valid = payload["valid"]
            assert isinstance(valid, list)
            assert all(isinstance(item, dict) for item in valid)

            invalid = payload["invalid"]
            assert isinstance(invalid, list)

            quarantined = payload["quarantined"]
            assert isinstance(quarantined, list)
            assert all(isinstance(item, dict) for item in quarantined)

            self.assertEqual(
                ["core.system.system_info"],
                [item["id"] for item in valid],
            )
            self.assertEqual([], invalid)
            self.assertEqual(
                ["plugin.broken_plugin"],
                [item["id"] for item in quarantined],
            )

            quarantined_item = quarantined[0]

            diagnostics = quarantined_item["diagnostics"]
            assert isinstance(diagnostics, list)
            assert all(isinstance(item, dict) for item in diagnostics)

            diagnostic = diagnostics[0]

            self.assertEqual(str(plugin_path), diagnostic["path"])
            self.assertEqual(1, diagnostic["line"])
            self.assertEqual("static.class_name", diagnostic["rule"])

            message = diagnostic["message"]
            assert isinstance(message, str)

            self.assertIn(
                "exactly one public collector class",
                message,
            )

            selected = report.to_dict(
                selected_plugins=("plugin.broken_plugin",),
            )

            selected_quarantined = selected["quarantined"]
            assert isinstance(selected_quarantined, list)

            selected_invalid = selected["invalid"]
            assert isinstance(selected_invalid, list)
            assert all(isinstance(item, dict) for item in selected_invalid)

            self.assertEqual([], selected_quarantined)
            self.assertEqual(
                ["plugin.broken_plugin"],
                [item["id"] for item in selected_invalid],
            )

            enabled = report.to_dict(enable_plugins=True)

            enabled_invalid = enabled["invalid"]
            assert isinstance(enabled_invalid, list)
            assert all(isinstance(item, dict) for item in enabled_invalid)

            self.assertEqual(
                ["plugin.broken_plugin"],
                [item["id"] for item in enabled_invalid],
            )

            plan = build_plan(
                report,
                RunRequest(),
            )

            planned_ids: list[str] = []

            for candidate in plan.collectors:
                assert candidate.metadata is not None
                planned_ids.append(candidate.metadata.id)

            self.assertEqual(
                ["core.system.system_info"],
                planned_ids,
            )

    def test_invalid_selected_folder_or_enabled_plugin_blocks_planning(self) -> None:
        """Folder-owned invalid plugins fail closed by logical ID and explicit enablement."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_path = root / "core" / "system" / "system_info.py"
            core_path.parent.mkdir(parents=True)
            core_path.write_text(COLLECTOR, encoding="utf-8")
            plugin_path = root / "plugins" / "broken_folder" / "main.py"
            plugin_path.parent.mkdir(parents=True)
            plugin_path.write_text('"""Invalid folder-owned plugin collector."""\n', encoding="utf-8")
            report = preflight(root)

            self.assertEqual("plugin.broken_folder", report.invalid[0].selection_id)
            build_plan(report, RunRequest())
            with self.assertRaises(PreflightError):
                build_plan(report, RunRequest(include=("plugin.broken_folder",)))
            with self.assertRaises(PreflightError):
                build_plan(report, RunRequest(enable_plugins=True))

    def test_preflight_cli_reports_quarantine_and_blocks_selected_invalid_plugins(self) -> None:
        """The public preflight report exposes actionable diagnostics and fail-closed selection."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_path = root / "core" / "system" / "system_info.py"
            core_path.parent.mkdir(parents=True)
            core_path.write_text(COLLECTOR, encoding="utf-8")
            plugin_path = root / "plugins" / "broken_plugin.py"
            plugin_path.parent.mkdir()
            plugin_path.write_text('"""Invalid plugin collector."""\n', encoding="utf-8")
            (root / "logicytics.yaml").write_text(
                "schema_version: 4\nmaintenance:\n  sysinternals_enabled: false\n",
                encoding="utf-8",
            )

            for arguments, expected_exit in (
                (["preflight"], 0),
                (["preflight", "--include", "plugin.broken_plugin"], 2),
                (["preflight", "--plugins"], 2),
            ):
                with self.subTest(arguments=arguments):
                    output = io.StringIO()
                    with (
                        patch.object(CLI, "project_root", return_value=root),
                        patch(
                            "sys.stdout",
                            output,
                        ),
                        patch(
                            "sys.stderr",
                            output,
                        ),
                    ):
                        exit_code = main(arguments)
                    rendered = output.getvalue()
                    self.assertEqual(expected_exit, exit_code)
                    if arguments == ["preflight"]:
                        self.assertIn("plugin.broken_plugin", rendered)
                        self.assertIn("Quarantined extensions: 1", rendered)

    def test_preflight_rejects_wrong_lifecycle_return_type(self) -> None:
        """Lifecycle annotations must match the strict collector contract exactly."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    "def collect(self, context: CollectorContext) -> CollectorResult:",
                    "def collect(self, context: CollectorContext) -> ValidationResult:",
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("collect must return CollectorResult", report.invalid[0].static_errors)
            diagnostic = next(item for item in report.invalid[0].diagnostics if item.message == "collect must return CollectorResult")
            source_lines = collector_path.read_text(encoding="utf-8").splitlines()
            expected_line = next(
                index
                for index, line in enumerate(source_lines, start=1)
                if "def collect(self, context: CollectorContext) -> ValidationResult:" in line
            )
            self.assertEqual(expected_line, diagnostic.line)
            self.assertEqual("static.return_annotation", diagnostic.rule)
            self.assertEqual(str(collector_path), diagnostic.path)

    def test_preflight_rejects_wrong_collection_estimate_type(self) -> None:
        """Optional estimates use the same strict typed contract as lifecycle methods."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    "    def cleanup(self, context: CollectorContext) -> None:",
                    "    def estimate(self, context: CollectorContext) -> ValidationResult:\n"
                    '        """Return an invalid estimate type for this test."""\n'
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
                COLLECTOR.replace(
                    "    def cleanup(self, context: CollectorContext) -> None:",
                    "    def dependencies(self) -> tuple[str, ...]:\n"
                    '        """Return an invalid instance-level dependency declaration."""\n'
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
                COLLECTOR.replace(
                    "    def cleanup(self, context: CollectorContext) -> None:",
                    "    @classmethod\n"
                    "    def dependencies(cls) -> tuple[str, ...]:\n"
                    '        """Return a dependency absent from metadata for this test."""\n'
                    "        return ('core.system.missing',)\n\n"
                    "    def cleanup(self, context: CollectorContext) -> None:",
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn(
                "dependencies() must match metadata.dependencies",
                report.invalid[0].runtime_error or "",
            )

    def test_preflight_rejects_malformed_validate_result(self) -> None:
        """The isolated preflight probe must reject a non-ValidationResult response."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace("return ValidationResult(True)", "return CollectorResult.succeeded('invalid')"),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("validate() must return ValidationResult", report.invalid[0].runtime_error or "")

    def test_preflight_rejects_validation_filesystem_process_network_and_environment_side_effects(
        self,
    ) -> None:
        """No validation probe may mutate files, launch processes, open sockets, or alter its environment."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            outside = root / "outside.marker"
            source = COLLECTOR.replace(
                "from pathlib import Path\n",
                "from pathlib import Path\nimport os\nimport socket\nimport subprocess\n",
            )
            attempts = (
                (f"Path({str(outside)!r}).write_text('unexpected', encoding='utf-8')", "open"),
                ("(context.workspace / 'unexpected').mkdir()", "os.mkdir"),
                ("subprocess.run(['python', '-c', 'pass'], check=False)", "subprocess.Popen"),
                ("socket.socket()", "socket.__new__"),
                ("os.environ['LOGICYTICS_PROBE_MUTATION'] = 'unexpected'", "os.putenv"),
            )
            for statement, event in attempts:
                with self.subTest(event=event):
                    collector_path.write_text(
                        source.replace(
                            "        return ValidationResult(True)",
                            f"        {statement}\n        return ValidationResult(True)",
                        ),
                        encoding="utf-8",
                    )
                    report = preflight(root)
                    self.assertEqual(1, len(report.invalid))
                    self.assertIn(
                        "validate() must not perform side effects",
                        report.invalid[0].runtime_error or "",
                    )
                    self.assertIn(event, report.invalid[0].runtime_error or "")
            self.assertFalse(outside.exists())

    def test_preflight_rejects_validation_artifact_registration_and_exceptions(self) -> None:
        """Artifact registration and ordinary validation exceptions block collection."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            attempts = (
                (
                    "context.artifacts.register_file(context.workspace / 'evidence.txt')",
                    "must not register artifacts",
                ),
                (
                    "raise RuntimeError('unexpected validation failure')",
                    "unexpected validation failure",
                ),
            )
            for statement, message in attempts:
                with self.subTest(message=message):
                    collector_path.write_text(
                        COLLECTOR.replace(
                            "        return ValidationResult(True)",
                            f"        {statement}\n        return ValidationResult(True)",
                        ),
                        encoding="utf-8",
                    )
                    report = preflight(root)
                    self.assertEqual(1, len(report.invalid))
                    self.assertIn(message, report.invalid[0].runtime_error or "")

    def test_preflight_rejects_a_hanging_validation_worker(self) -> None:
        """A timed-out validation subprocess must quarantine its collector before launch."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            with patch.object(
                discovery.process_adapter,
                discovery.process_adapter.run.__name__,
                side_effect=subprocess.TimeoutExpired(
                    ["validation-worker"],
                    timeout=10,
                ),
            ):
                report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("validation worker failed", report.invalid[0].runtime_error or "")
            self.assertIn("timed out", report.invalid[0].runtime_error or "")

    def test_preflight_rejects_collector_print_calls(self) -> None:
        """Collectors must emit structured events rather than write directly to stdout."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    'return CollectorResult.succeeded("test artifact created", (artifact,))',
                    "print('unexpected output')\n        return CollectorResult.succeeded(\"test artifact created\", (artifact,))",
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn(
                "collectors must not print; use structured progress or logging",
                report.invalid[0].static_errors,
            )

    def test_preflight_rejects_import_time_calls_in_headers_and_class_body(self) -> None:
        """Collection-like work must not run before a worker has isolated the collector."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    "class SystemInfoCollector(CoreCollector):",
                    "class SystemInfoCollector(CoreCollector, type(open('unexpected.txt'))):",
                )
                .replace(
                    "    @classmethod\n    def metadata",
                    "    marker = input('unexpected prompt')\n\n    @classmethod\n    def metadata",
                )
                .replace(
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
            source = COLLECTOR.replace(
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

    def test_unselected_privileged_plugin_never_requests_host_elevation(self) -> None:
        """An opt-in privileged plugin must not affect an ordinary core-only plan."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_path = root / "plugins" / "admin_plugin.py"
            plugin_path.parent.mkdir(parents=True)
            source = plugin_collector_source()
            source = source.replace("SystemInfoCollector", "AdminPluginCollector")
            source = source.replace("core.system.system_info", "plugin.admin_plugin")
            source = (
                source.replace(
                    "from logicytics import CollectorMetadata",
                    "from logicytics import Capability, CollectorMetadata",
                )
                .replace(
                    "            capabilities=(),",
                    "            capabilities=(Capability.ELEVATED_PRIVILEGES,),",
                )
                .replace(
                    "            privilege_level=PrivilegeLevel.STANDARD,",
                    "            privilege_level=PrivilegeLevel.ELEVATED,",
                )
            )
            plugin_path.write_text(source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual((), report.invalid)
            with patch("logicytics.module.planner.inspect_environment") as inspect:
                plan = build_plan(report, RunRequest())
            self.assertEqual((), plan.collectors)
            inspect.assert_not_called()
            enabled = RunRequest(
                enable_plugins=True,
                approved_capabilities=(Capability.ELEVATED_PRIVILEGES,),
            )
            with (
                patch(
                    "logicytics.module.planner.inspect_environment",
                    return_value=EnvironmentReport(False, True, None),
                ),
                self.assertRaisesRegex(PlanError, "administrator account"),
            ):
                build_plan(report, enabled)


if __name__ == "__main__":
    unittest.main()
