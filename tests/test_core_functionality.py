"""Integration-style checks for the v4 core without shipped collectors."""

from __future__ import annotations

import hashlib
import ctypes
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from logicytics import packaging
from logicytics.artifacts import WorkspaceArtifactWriter
from logicytics.command_runner import parse_level_messages, run_command
from logicytics.cli import _parser, _request
from logicytics.file_listing import list_files
from logicytics.logging import FileEventLogger, deprecated, raise_logged, timed
from logicytics.sysinternals import ensure_sysinternals
from logicytics.configuration import default_config, load_config
from logicytics.contracts import Capability, CollectorMetadata, ResourceClass, RunRequest, Specialty
from logicytics.discovery import preflight
from logicytics.environment import EnvironmentReport
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


def _delayed_collector_source(
        filename: str,
        delay: float,
        *,
        parallel_safe: bool = True,
        dependencies: tuple[str, ...] = (),
        resource_class: ResourceClass = ResourceClass.GENERAL,
        fail: bool = False,
) -> str:
    """Create a valid fixture collector with observable scheduling duration."""
    class_name = "".join(part.title() for part in filename.split("_"))
    source = _COLLECTOR.replace("SystemInfoCollector", f"{class_name}Collector")
    source = source.replace("core.system.system_info", f"core.system.{filename}")
    source = source.replace("from pathlib import Path\n", "from pathlib import Path\nfrom time import sleep\n")
    source = source.replace(
        "from logicytics import CollectorMetadata,",
        "from logicytics import CollectorMetadata, ResourceClass,",
    )
    source = source.replace(
        '            supported_platforms=("win32",),',
        '            supported_platforms=("win32",),\n'
        f'            dependencies={dependencies!r},\n'
        f'            resource_class=ResourceClass.{resource_class.name},\n'
        f'            parallel_safe={parallel_safe!r},',
    )
    source = source.replace(
        "    def validate(self, context: CollectorContext) -> ValidationResult:\n",
        "    @classmethod\n"
        "    def dependencies(cls) -> tuple[str, ...]:\n"
        "        \"\"\"Return fixture dependencies for scheduler tests.\"\"\"\n"
        f"        return {dependencies!r}\n\n"
        "    def validate(self, context: CollectorContext) -> ValidationResult:\n",
    )
    source = source.replace(
        '        output = context.workspace / "system.txt"',
        f'        sleep({delay})\n        output = context.workspace / "system.txt"',
    )
    if fail:
        source = source.replace(
            f'        sleep({delay})\n        output = context.workspace / "system.txt"',
            f'        sleep({delay})\n        raise RuntimeError("dependency fixture failed")',
        )
    return source


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
        with self.assertRaisesRegex(ValueError, "maximum_memory_bytes"):
            CollectorMetadata(**{**common, "maximum_memory_bytes": True})
        with self.assertRaisesRegex(ValueError, "maximum_artifact_bytes"):
            CollectorMetadata(**{**common, "maximum_artifact_bytes": True})
        with self.assertRaisesRegex(ValueError, "maximum_artifact_bytes"):
            CollectorMetadata(**{**common, "maximum_output_bytes": 8, "maximum_artifact_bytes": 9})
        with self.assertRaisesRegex(ValueError, "resource_class"):
            CollectorMetadata(**{**common, "resource_class": "disk_heavy"})
        with self.assertRaisesRegex(ValueError, "ResourceClass"):
            CollectorMetadata.from_dict({**CollectorMetadata(**common).to_dict(), "resource_class": "unknown"})
        with self.assertRaisesRegex(ValueError, "sensitive collectors"):
            CollectorMetadata(**{**common, "sensitive_data_categories": ("credentials",)})
        for retries in (-1, 4, True):
            with self.subTest(maximum_retries=retries):
                with self.assertRaisesRegex(ValueError, "maximum_retries"):
                    CollectorMetadata(**{**common, "maximum_retries": retries})
        for retry_delay in (-1, 31, True, float("inf")):
            with self.subTest(retry_delay_seconds=retry_delay):
                with self.assertRaisesRegex(ValueError, "retry_delay_seconds"):
                    CollectorMetadata(**{**common, "retry_delay_seconds": retry_delay})
        metadata = CollectorMetadata(**{**common, "maximum_output_bytes": 8})
        self.assertEqual(8, metadata.maximum_artifact_bytes)
        self.assertIs(ResourceClass.GENERAL, metadata.resource_class)
        disk_metadata = CollectorMetadata(**{**common, "resource_class": ResourceClass.DISK_HEAVY})
        self.assertEqual("disk_heavy", disk_metadata.to_dict()["resource_class"])
        self.assertIs(ResourceClass.DISK_HEAVY, CollectorMetadata.from_dict(disk_metadata.to_dict()).resource_class)

    def test_sensitive_collectors_require_explicit_profile_or_include_opt_in(self) -> None:
        """Default collection excludes sensitive evidence unless explicitly selected."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            (core_directory / "a_standard.py").write_text(
                _delayed_collector_source("a_standard", 0.0),
                encoding="utf-8",
            )
            sensitive_id = "core.system.z_sensitive"
            source = _delayed_collector_source("z_sensitive", 0.0).replace(
                "from logicytics import CollectorMetadata",
                "from logicytics import Capability, CollectorMetadata",
            ).replace(
                '            supported_platforms=("win32",),',
                '            supported_platforms=("win32",),\n'
                '            capabilities=(Capability.SENSITIVE_FILES,),\n'
                '            sensitive_data_categories=("credentials",),\n'
                '            default_profiles=("deep",),',
            )
            (core_directory / "z_sensitive.py").write_text(source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual((), report.invalid)

            standard = build_plan(report, RunRequest())
            self.assertEqual(["core.system.a_standard"], [item.metadata.id for item in standard.collectors])
            with self.assertRaisesRegex(PlanError, "unapproved capabilities"):
                build_plan(report, RunRequest(include=(sensitive_id,)))
            opted_in = build_plan(
                report,
                RunRequest(
                    include=(sensitive_id,),
                    exclude=("core.system.a_standard",),
                    approved_capabilities=(Capability.SENSITIVE_FILES,),
                ),
            )
            self.assertEqual([sensitive_id], [item.metadata.id for item in opted_in.collectors])
            deep = build_plan(
                report,
                RunRequest(profile="deep", approved_capabilities=(Capability.SENSITIVE_FILES,)),
            )
            self.assertEqual([sensitive_id], [item.metadata.id for item in deep.collectors])

    def test_preflight_rejects_unregistered_collector_resource_classes(self) -> None:
        """Invalid scheduling metadata blocks shipped collectors before any run starts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            resource_class="disk_heavy",',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("resource_class", report.invalid[0].runtime_error)
            with self.assertRaises(PreflightError):
                build_plan(report, RunRequest())

    def test_preflight_rejects_sensitive_default_profile_collector(self) -> None:
        """A shipped collector cannot silently introduce sensitive default evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(
                _COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n'
                    '            sensitive_data_categories=("credentials",),',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("sensitive collectors", report.invalid[0].runtime_error)
            with self.assertRaises(PreflightError):
                build_plan(report, RunRequest())

    def test_authorization_error_summarizes_categories_and_sensitive_outputs(self) -> None:
        """Collection consent must explain requested evidence before creating a workspace."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(
                _COLLECTOR.replace(
                    "from logicytics import CollectorMetadata",
                    "from logicytics import Capability, CollectorMetadata",
                ).replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n'
                    '            capabilities=(Capability.SENSITIVE_FILES,),\n'
                    '            sensitive_data_categories=("credentials", "personal_documents"),\n'
                    '            default_profiles=("deep",),',
                ),
                encoding="utf-8",
            )
            configuration = default_config(root)
            plan = build_plan(
                preflight(root),
                RunRequest(profile="deep", approved_capabilities=(Capability.SENSITIVE_FILES,)),
            )
            with self.assertRaises(PermissionError) as rejected:
                RunSupervisor(root, configuration).run(plan)
            message = str(rejected.exception)
            self.assertIn("--acknowledge-authorization", message)
            self.assertIn("selected categories: system", message)
            self.assertIn("sensitive outputs: credentials, personal_documents", message)
            self.assertFalse(configuration.runtime.output_root.exists())

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

    def test_structured_event_logger_redacts_secret_fields_and_inline_credentials(self) -> None:
        """Structured diagnostics must preserve useful fields without exposing secrets."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            logger = FileEventLogger(path, run_id="test-run", collector_id="core.system.private_keys")
            logger.event(
                "INFO",
                "Authorization: Bearer bearer-value password=message-value",
                password="field-value",
                api_key="api-value",
                ordinary="safe",
            )
            contents = path.read_text(encoding="utf-8")
            for secret in ("bearer-value", "message-value", "field-value", "api-value"):
                self.assertNotIn(secret, contents)
            event = json.loads(contents)
            self.assertEqual("info", event["level"])
            self.assertEqual("core.system.private_keys", event["collector_id"])
            self.assertEqual("[REDACTED]", event["fields"]["password"])
            self.assertEqual("[REDACTED]", event["fields"]["api_key"])
            self.assertEqual("safe", event["fields"]["ordinary"])

    def test_structured_event_logger_serializes_concurrent_jsonl_events(self) -> None:
        """Concurrent diagnostics stay complete, independently parseable, and redacted."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            logger = FileEventLogger(path, run_id="concurrent-run", collector_id="core.system.test")

            def write_event(index: int) -> None:
                logger.event("info", "collector_progress", sequence=index, password=f"secret-{index}")

            with ThreadPoolExecutor(max_workers=12) as executor:
                list(executor.map(write_event, range(120)))
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(120, len(events))
            self.assertEqual(set(range(120)), {event["fields"]["sequence"] for event in events})
            self.assertTrue(all(event["fields"]["password"] == "[REDACTED]" for event in events))

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
            config_path.write_text('{"schema_version":4,"runtime":{"package_completed_runs":"yes"}}', encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "package_completed_runs"):
                load_config(root)

    def test_configuration_manifest_redacts_nested_secrets_without_mutating_worker_settings(self) -> None:
        """Manifest snapshots hide credentials while collectors retain configured access."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_id = "core.system.private_keys"
            settings = {
                "password": "password-value",
                "nested": {"refresh_token": "token-value", "cookie": "cookie-value"},
                "private_key": "private-value",
                "ordinary": "safe",
            }
            (root / "logicytics.json").write_text(
                json.dumps({"schema_version": 4, "collectors": {collector_id: settings}}),
                encoding="utf-8",
            )
            configuration = load_config(root)
            self.assertEqual(settings, configuration.settings_for(collector_id))
            snapshot = configuration.to_manifest_dict()
            manifest_settings = snapshot["collector_settings"][collector_id]
            self.assertEqual("[REDACTED]", manifest_settings["password"])
            self.assertEqual("[REDACTED]", manifest_settings["nested"]["refresh_token"])
            self.assertEqual("[REDACTED]", manifest_settings["nested"]["cookie"])
            self.assertEqual("[REDACTED]", manifest_settings["private_key"])
            self.assertEqual("safe", manifest_settings["ordinary"])
            for secret in ("password-value", "token-value", "cookie-value", "private-value"):
                self.assertNotIn(secret, json.dumps(snapshot))
            self.assertEqual(settings, configuration.settings_for(collector_id))

    def test_configuration_rejects_boolean_workers_and_invalid_output_roots(self) -> None:
        """Runtime worker limits and output locations must retain strict JSON types."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            invalid_configurations = (
                ('{"schema_version":true}', "schema_version"),
                ('{"schema_version":4,"runtime":{"default_max_workers":true}}', "worker limits"),
                ('{"schema_version":4,"runtime":{"maximum_workers":true}}', "worker limits"),
                ('{"schema_version":4,"runtime":{"maximum_run_output_bytes":true}}', "maximum_run_output_bytes"),
                ('{"schema_version":4,"runtime":{"maximum_run_output_bytes":0}}', "maximum_run_output_bytes"),
                ('{"schema_version":4,"runtime":{"output_root":false}}', "output_root"),
                ('{"schema_version":4,"runtime":{"output_root":"   "}}', "output_root"),
            )
            for payload, message in invalid_configurations:
                with self.subTest(payload=payload):
                    config_path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)

    def test_run_request_rejects_invalid_selection_and_execution_policy(self) -> None:
        """Run requests must be immutable, typed declarations before a plan exists."""
        collector_id = "core.system.system_info"
        invalid_requests = (
            ({"profile": "Standard"}, "profile"),
            ({"include": [collector_id]}, "include"),
            ({"include": ("../system",)}, "include"),
            ({"include": (collector_id, collector_id)}, "include"),
            ({"include": (collector_id,), "exclude": (collector_id,)}, "overlap"),
            ({"enable_plugins": 1}, "enable_plugins"),
            ({"acknowledge_authorization": 1}, "acknowledge_authorization"),
            ({"performance_check": 1}, "performance_check"),
            ({"performance_check": True, "max_workers": 2}, "performance_check"),
            ({"max_workers": True}, "max_workers"),
            ({"max_workers": 65}, "max_workers"),
            ({"approved_capabilities": ("filesystem_read",)}, "approved_capabilities"),
            (
                {"approved_capabilities": (Capability.FILESYSTEM_READ, Capability.FILESYSTEM_READ)},
                "approved_capabilities",
            ),
        )
        for options, message in invalid_requests:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    RunRequest(**options)

    def test_configured_worker_limit_blocks_oversized_run_before_workspace_creation(self) -> None:
        """The configured maximum worker count bounds a run before any collector starts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            config_path = root / "logicytics.json"
            config_path.write_text(
                '{"schema_version":4,"runtime":{"default_max_workers":1,"maximum_workers":2}}',
                encoding="utf-8",
            )
            configuration = load_config(root)
            self.assertEqual(1, configuration.runtime.default_max_workers)
            self.assertEqual(2, configuration.runtime.maximum_workers)
            plan = build_plan(preflight(root), RunRequest(max_workers=3, acknowledge_authorization=True))
            with self.assertRaisesRegex(ValueError, "maximum_workers"):
                RunSupervisor(root, configuration).run(plan)
            self.assertFalse(configuration.runtime.output_root.exists())

    def test_configuration_validates_bounded_network_and_packet_settings(self) -> None:
        """Collector-specific settings fail early rather than being silently coerced at runtime."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            config_path.write_text(
                '{"schema_version":4,"collectors":{"core.network.bandwidth_sample":'
                '{"sample_count":11}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PlanError, "sample_count"):
                load_config(root)
            config_path.write_text(
                '{"schema_version":4,"collectors":{"core.packet.packet_capture":'
                '{"packet_count":10,"timeout_seconds":5,"retry_window_seconds":2,"interface":"127.0.0.1"}}}',
                encoding="utf-8",
            )
            settings = load_config(root).settings_for("core.packet.packet_capture")
            self.assertEqual(10, settings["packet_count"])
            self.assertEqual(2, settings["retry_window_seconds"])

    def test_run_parser_accepts_performance_check(self) -> None:
        """The run command must expose the performance mode used by the request builder."""
        arguments = _parser().parse_args(["run", "--performance-check"])
        self.assertTrue(arguments.performance_check)
        request = _request(arguments, default_workers=4)
        self.assertTrue(request.performance_check)
        self.assertEqual(1, request.max_workers)

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

    def test_run_output_budget_bounds_concurrent_collector_evidence(self) -> None:
        """Concurrent isolated workers must never exceed the configured run-wide cap."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            for filename in ("a_first", "b_second"):
                source = _delayed_collector_source(filename, 0.1).replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            maximum_output_bytes=4,',
                )
                (core_directory / f"{filename}.py").write_text(source, encoding="utf-8")
            config_path = root / "logicytics.json"
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
            self.assertEqual("succeeded", records["core.system.a_first"].status, records["core.system.a_first"].errors)
            self.assertEqual("failed", records["core.system.b_second"].status)
            self.assertTrue(
                any("maximum_run_output_bytes" in error for error in records["core.system.b_second"].errors)
            )
            self.assertLessEqual(outcome.manifest.total_artifact_bytes, 7)

    def test_exhausted_run_output_budget_skips_unstarted_collectors(self) -> None:
        """A completely consumed run quota prevents later workers from launching."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            for filename in ("a_first", "b_second"):
                source = _delayed_collector_source(filename, 0.0).replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            maximum_output_bytes=4,',
                )
                (core_directory / f"{filename}.py").write_text(source, encoding="utf-8")
            config_path = root / "logicytics.json"
            config_path.write_text(
                '{"schema_version":4,"runtime":{"maximum_run_output_bytes":4}}',
                encoding="utf-8",
            )
            configuration = load_config(root)
            plan = build_plan(preflight(root), RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, configuration).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertEqual("succeeded", records["core.system.a_first"].status, records["core.system.a_first"].errors)
            second = records["core.system.b_second"]
            self.assertEqual("skipped", second.status)
            self.assertIsNone(second.started_at)
            self.assertTrue(any("maximum_run_output_bytes" in error for error in second.errors))
            self.assertEqual(4, outcome.manifest.total_artifact_bytes)

    def test_isolated_worker_enforces_declared_individual_artifact_limit(self) -> None:
        """Metadata file limits must survive preflight and reach the isolated worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
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
            artifact = writer.register_file(source, transformations=("normalized",))
            self.assertEqual("evidence_graph", artifact.source_category)
            datetime.fromisoformat(artifact.collected_at)
            self.assertEqual(
                ("normalized", "copied into run artifact store"),
                artifact.transformations,
            )

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
            self.assertEqual("run", outcome.manifest.action)
            self.assertEqual("4.0", outcome.manifest.engine_version)
            self.assertIn("user", outcome.manifest.host)
            self.assertIn("is_administrator", outcome.manifest.host)
            self.assertIsNotNone(outcome.manifest.collectors[0].heartbeat_at)
            self.assertIsNotNone(outcome.manifest.collectors[0].last_progress_at)
            self.assertFalse((outcome.run_directory / "collectors" / "core_system_system_info" / "tmp").exists())
            self.assertIsNotNone(outcome.manifest.package)
            self.assertTrue(Path(outcome.manifest.package["path"]).is_file())
            self.assertTrue(Path(outcome.manifest.package["sha256_path"]).is_file())
            package_path, hash_path = package_run(outcome)
            self.assertTrue(package_path.is_file())
            self.assertTrue(hash_path.is_file())
            package_digest, sidecar_name = hash_path.read_text(encoding="ascii").split()
            self.assertEqual(package_path.name, sidecar_name)
            self.assertEqual(hashlib.sha256(package_path.read_bytes()).hexdigest(), package_digest)
            self.assertEqual(package_digest, outcome.manifest.package["sha256"])
            with zipfile.ZipFile(package_path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("manifest.json", archive.namelist())
                self.assertIn("summary.txt", archive.namelist())
                self.assertNotIn("logs/performance.json", archive.namelist())
                self.assertEqual(1, len([name for name in archive.namelist() if name.startswith("artifacts/")]))
                artifact = outcome.manifest.artifact_list()[0]
                archived_bytes = archive.read(f"artifacts/{artifact.relative_path}")
                self.assertEqual(artifact.size_bytes, len(archived_bytes))
                self.assertEqual(artifact.sha256, hashlib.sha256(archived_bytes).hexdigest())
                self.assertEqual("system", artifact.source_category)
                datetime.fromisoformat(artifact.collected_at)
                self.assertEqual(("copied into run artifact store",), artifact.transformations)
                packaged_manifest = json.loads(archive.read("manifest.json"))
                packaged_artifact = packaged_manifest["collectors"][0]["artifacts"][0]
                self.assertEqual(artifact.source_category, packaged_artifact["source_category"])
                self.assertEqual(artifact.collected_at, packaged_artifact["collected_at"])
                self.assertEqual(list(artifact.transformations), packaged_artifact["transformations"])
                record = outcome.manifest.collectors[0]
                summary = archive.read("summary.txt").decode("utf-8")
                self.assertIn(f"Status: {record.status}", summary)
                self.assertIn(f"Started: {record.started_at}", summary)
                self.assertIn(f"Finished: {record.finished_at}", summary)

    def test_sensitive_artifact_bytes_are_preserved_while_packaged_diagnostics_are_redacted(self) -> None:
        """Evidence retains intentional secrets, but no packaged diagnostics disclose them."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '        output = context.workspace / "system.txt"',
                    '        password = context.settings["password"]\n'
                    '        token = context.settings["nested"]["access_token"]\n'
                    '        output = context.workspace / "system.txt"',
                ).replace(
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")',
                    '        output.write_text(f"password={password} token={token}", encoding="utf-8")\n'
                    '        context.logger.event("info", f"password={password}", access_token=token)\n'
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")',
                ).replace(
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    '        return CollectorResult.succeeded(f"password={password}", (artifact,))',
                ),
                encoding="utf-8",
            )
            (root / "logicytics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "collectors": {
                            "core.system.system_info": {
                                "password": "evidence-password",
                                "nested": {"access_token": "evidence-token"},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, load_config(root)).run(plan)
            self.assertEqual("succeeded", outcome.manifest.status.value)
            self.assertEqual("password=[REDACTED]", outcome.manifest.collectors[0].summary)
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                artifact = outcome.manifest.artifact_list()[0]
                self.assertEqual(
                    "password=evidence-password token=evidence-token",
                    archive.read(f"artifacts/{artifact.relative_path}").decode("utf-8"),
                )
                diagnostics = "\n".join(
                    archive.read(name).decode("utf-8")
                    for name in archive.namelist()
                    if not name.startswith("artifacts/")
                )
                self.assertNotIn("evidence-password", diagnostics)
                self.assertNotIn("evidence-token", diagnostics)
                self.assertIn("[REDACTED]", diagnostics)

    def test_worker_exception_secrets_are_redacted_from_manifest_and_package(self) -> None:
        """Collector crashes remain actionable without leaking inline credentials."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '        output = context.workspace / "system.txt"',
                    '        raise RuntimeError("password=crash-password token=crash-token")',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            self.assertEqual("failed", outcome.manifest.collectors[0].status)
            self.assertIn("RuntimeError", "\n".join(outcome.manifest.collectors[0].errors))
            failure = outcome.manifest.collectors[0].failure
            self.assertIsNotNone(failure)
            self.assertNotIn("crash-password", failure["platform_error"])
            self.assertNotIn("crash-token", failure["platform_error"])
            self.assertIn("[REDACTED]", failure["platform_error"])
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                packaged_record = json.loads(archive.read("manifest.json"))["collectors"][0]
                self.assertEqual(failure, packaged_record["failure"])
                diagnostics = "\n".join(
                    archive.read(name).decode("utf-8")
                    for name in archive.namelist()
                    if not name.startswith("artifacts/")
                )
                self.assertNotIn("crash-password", diagnostics)
                self.assertNotIn("crash-token", diagnostics)
                self.assertIn("[REDACTED]", diagnostics)

    def test_performance_report_is_finalized_before_automatic_packaging(self) -> None:
        """Performance-mode timing evidence must be present in the automatic ZIP."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            plan = build_plan(
                preflight(root),
                RunRequest(
                    max_workers=1,
                    acknowledge_authorization=True,
                    performance_check=True,
                ),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            performance_path = outcome.run_directory / "logs" / "performance.json"
            self.assertTrue(performance_path.is_file())
            report = json.loads(performance_path.read_text(encoding="utf-8"))
            self.assertEqual(outcome.manifest.run_id, report["run_id"])
            self.assertEqual("core.system.system_info", report["collectors"][0]["id"])
            self.assertIsNotNone(report["collectors"][0]["duration_seconds"])
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                packaged_report = json.loads(archive.read("logs/performance.json"))
            self.assertEqual(report, packaged_report)
            performance_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "performance report"):
                package_run(outcome)

    def test_collector_resource_progress_persists_in_manifest_summary_and_performance_report(self) -> None:
        """Incremental progress fields remain visible live and in every packaged diagnostic."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace("from pathlib import Path\n", "from pathlib import Path\nfrom time import sleep\n").replace(
                    '        output = context.workspace / "system.txt"',
                    '        context.report_progress("scan_started", scanned_files=3)\n'
                    '        sleep(0.2)\n'
                    '        context.report_progress(\n'
                    '            "scan_finished", scanned_files=9, copied_files=4, bytes_written=25,\n'
                    '            observation_count=6, event_count=11,\n'
                    '        )\n'
                    '        sleep(0.2)\n'
                    '        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )
            plan = build_plan(
                preflight(root),
                RunRequest(max_workers=1, acknowledge_authorization=True, performance_check=True),
            )
            snapshots: list[dict[str, object]] = []
            from logicytics.manifest import write_manifest as original_write_manifest

            def capture_manifest(path: Path, manifest: object) -> None:
                snapshots.append(json.loads(json.dumps(manifest.to_dict())))
                original_write_manifest(path, manifest)

            with patch("logicytics.runtime.write_manifest", side_effect=capture_manifest):
                outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]
            expected = {
                "files_scanned": 9,
                "files_copied": 4,
                "bytes_written": 25,
                "packets_observed": 6,
                "events_processed": 11,
            }
            for field, value in expected.items():
                self.assertEqual(value, record.progress[field])
            self.assertGreater(record.progress["elapsed_seconds"], 0)
            self.assertTrue(
                any(
                    snapshot["collectors"][0]["status"] == "running"
                    and snapshot["collectors"][0]["progress"]["files_scanned"] >= 3
                    for snapshot in snapshots
                )
            )
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                packaged_record = json.loads(archive.read("manifest.json"))["collectors"][0]
                report = json.loads(archive.read("logs/performance.json"))["collectors"][0]
                summary = archive.read("summary.txt").decode("utf-8")
            self.assertEqual(record.progress, packaged_record["progress"])
            self.assertEqual(record.progress, report["progress"])
            self.assertEqual(record.peak_memory_bytes, report["peak_memory_bytes"])
            self.assertIn("Files scanned: 9", summary)
            self.assertIn("Files copied: 4", summary)
            self.assertIn("Bytes written: 25", summary)
            self.assertIn("Packets observed: 6", summary)
            self.assertIn("Events processed: 11", summary)

    def test_package_excludes_unregistered_files_and_rejects_tampered_artifacts(self) -> None:
        """Only manifest artifacts may enter a package, and their final bytes must match."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            report = preflight(root)
            plan = build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            unregistered = outcome.run_directory / "artifacts" / "unregistered.txt"
            unregistered.write_text("must not be packaged\n", encoding="utf-8")

            package_path, _ = package_run(outcome)
            with zipfile.ZipFile(package_path) as archive:
                self.assertNotIn("artifacts/unregistered.txt", archive.namelist())

            original_write = packaging._stream_archive_member

            def tampering_write(
                    archive: zipfile.ZipFile,
                    filename: Path,
                    arcname: str,
            ) -> None:
                if arcname.startswith("artifacts/"):
                    archive.writestr(arcname, b"tampered artifact bytes")
                    return
                original_write(archive, filename, arcname)

            with patch.object(packaging, "_stream_archive_member", new=tampering_write):
                with self.assertRaisesRegex(ValueError, "packaged artifact verification failed"):
                    package_run(outcome)
            self.assertFalse(package_path.with_suffix(".zip.tmp").exists())

    def test_package_excludes_unregistered_collector_and_engine_jsonl_files(self) -> None:
        """Collector-created JSONL files cannot bypass the registered artifact contract."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            collector_workspace = outcome.run_directory / "collectors" / "core_system_system_info"
            (collector_workspace / "unregistered.jsonl").write_text("secret workspace data", encoding="utf-8")
            (outcome.run_directory / "logs" / "unregistered.jsonl").write_text(
                "secret engine data",
                encoding="utf-8",
            )

            package_path, _ = package_run(outcome)
            with zipfile.ZipFile(package_path) as archive:
                self.assertIn("logs/engine.jsonl", archive.namelist())
                self.assertIn("collectors/core_system_system_info/events.jsonl", archive.namelist())
                self.assertNotIn("logs/unregistered.jsonl", archive.namelist())
                self.assertNotIn("collectors/core_system_system_info/unregistered.jsonl", archive.namelist())

    def test_package_sidecar_failure_restores_existing_verified_package(self) -> None:
        """A failed hash publication cannot replace or corrupt a previously verified ZIP."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            package_path = Path(outcome.manifest.package["path"])
            hash_path = Path(outcome.manifest.package["sha256_path"])
            original_package = package_path.read_bytes()
            original_hash = hash_path.read_bytes()
            original_replace = os.replace

            def fail_sidecar(source: str | Path, destination: str | Path) -> None:
                if Path(destination) == hash_path and Path(source).suffix == ".tmp":
                    raise OSError("simulated sidecar publication failure")
                original_replace(source, destination)

            with patch("logicytics.packaging.os.replace", side_effect=fail_sidecar):
                with self.assertRaisesRegex(OSError, "sidecar publication failure"):
                    package_run(outcome)
            self.assertEqual(original_package, package_path.read_bytes())
            self.assertEqual(original_hash, hash_path.read_bytes())
            self.assertFalse(package_path.with_suffix(".zip.tmp").exists())
            self.assertFalse(package_path.with_suffix(".zip.backup").exists())
            self.assertFalse(hash_path.with_suffix(".sha256.tmp").exists())
            self.assertFalse(hash_path.with_suffix(".sha256.backup").exists())

    def test_package_rejects_manifest_artifact_path_escape_and_forged_collector_ownership(self) -> None:
        """A modified manifest cannot package outside files or another collector's evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            artifact = outcome.manifest.collectors[0].artifacts[0]
            original_path = artifact["relative_path"]
            artifact["relative_path"] = "../outside.jsonl"
            with self.assertRaisesRegex(ValueError, "escapes its collector-owned store"):
                package_run(outcome)
            artifact["relative_path"] = original_path
            artifact["collector_id"] = "core.system.another"
            with self.assertRaisesRegex(ValueError, "collector ownership"):
                package_run(outcome)

    def test_package_rejects_collector_event_log_symlinks_outside_the_run(self) -> None:
        """A collector event-channel symlink must not smuggle external files into a ZIP."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            outside = root / "outside.jsonl"
            outside.write_text("outside credentials", encoding="utf-8")
            events = outcome.run_directory / "collectors" / "core_system_system_info" / "events.jsonl"
            events.unlink()
            events.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "diagnostic log escapes"):
                package_run(outcome)

    def test_package_rejects_manifest_artifact_symlinks_outside_collector_storage(self) -> None:
        """Matching bytes cannot make an externally redirected artifact safe to package."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            artifact = outcome.manifest.artifact_list()[0]
            stored = outcome.run_directory / "artifacts" / artifact.relative_path
            outside = root / "outside.txt"
            outside.write_bytes(stored.read_bytes())
            stored.unlink()
            stored.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "escapes its collector-owned store"):
                package_run(outcome)

    def test_package_streams_large_evidence_and_hashes_with_bounded_reads(self) -> None:
        """Large artifacts must enter ZIP/hash verification without whole-file reads."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")',
                    '        output.write_bytes(b"x" * (2 * 1024 * 1024 + 17))\n'
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            artifact = outcome.manifest.artifact_list()[0]
            source = outcome.run_directory / "artifacts" / artifact.relative_path
            expected_package_path = Path(outcome.manifest.package["path"]).with_suffix(".zip.tmp")
            original_open = Path.open
            read_sizes: list[tuple[Path, int]] = []

            class BoundedReader:
                def __init__(self, stream: object, path: Path) -> None:
                    self.stream = stream
                    self.path = path

                def __enter__(self) -> "BoundedReader":
                    self.stream.__enter__()
                    return self

                def __exit__(self, *arguments: object) -> object:
                    return self.stream.__exit__(*arguments)

                def read(self, size: int = -1) -> bytes:
                    if not 0 < size <= 1024 * 1024:
                        raise AssertionError(f"unbounded evidence read requested: {size}")
                    read_sizes.append((self.path, size))
                    return self.stream.read(size)

            def guarded_open(path: Path, *arguments: object, **options: object) -> object:
                stream = original_open(path, *arguments, **options)
                if path in {source, expected_package_path} and arguments and arguments[0] == "rb":
                    return BoundedReader(stream, path)
                return stream

            with patch.object(Path, "open", new=guarded_open):
                package_path, hash_path = package_run(outcome)
            self.assertGreaterEqual(len(read_sizes), 6)
            self.assertEqual({source, expected_package_path}, {path for path, _ in read_sizes})
            self.assertEqual(artifact.size_bytes, 2 * 1024 * 1024 + 17)
            self.assertTrue(package_path.is_file())
            self.assertTrue(hash_path.is_file())

    def test_parallel_completion_preserves_deterministic_manifest_order(self) -> None:
        """Collector completion order must not reorder the preflighted execution plan."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()

            (core_directory / "a_slow.py").write_text(
                _delayed_collector_source("a_slow", 0.6),
                encoding="utf-8",
            )
            (core_directory / "z_fast.py").write_text(
                _delayed_collector_source("z_fast", 0.0),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(
                report,
                RunRequest(max_workers=2, acknowledge_authorization=True),
            )
            planned_ids = [candidate.metadata.id for candidate in plan.collectors]
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertEqual(planned_ids, [record.id for record in outcome.manifest.collectors])
            self.assertLess(records["core.system.z_fast"].finished_at, records["core.system.a_slow"].finished_at)

    def test_parallel_unsafe_collector_runs_without_worker_overlap(self) -> None:
        """A collector declaring parallel_safe=False must run alone between bounded workers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            fixtures = (
                ("a_parallel", 0.3, True),
                ("m_serial", 0.2, False),
                ("z_parallel", 0.0, True),
            )
            for filename, delay, parallel_safe in fixtures:
                (core_directory / f"{filename}.py").write_text(
                    _delayed_collector_source(filename, delay, parallel_safe=parallel_safe),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(
                report,
                RunRequest(max_workers=3, acknowledge_authorization=True),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            first = records["core.system.a_parallel"]
            serial = records["core.system.m_serial"]
            last = records["core.system.z_parallel"]
            self.assertLessEqual(first.finished_at, serial.started_at)
            self.assertLessEqual(serial.finished_at, last.started_at)

    def test_conflicting_resource_classes_run_without_worker_overlap(self) -> None:
        """Disk, network, and registry resource conflicts each serialize their owners."""
        for resource_class in (
                ResourceClass.DISK_HEAVY,
                ResourceClass.NETWORK_HEAVY,
                ResourceClass.REGISTRY_SENSITIVE,
        ):
            with self.subTest(resource_class=resource_class), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                core_directory = root / "core" / "system"
                core_directory.mkdir(parents=True)
                (root / "plugins").mkdir()
                for filename in ("a_first", "z_second"):
                    (core_directory / f"{filename}.py").write_text(
                        _delayed_collector_source(filename, 0.15, resource_class=resource_class),
                        encoding="utf-8",
                    )
                report = preflight(root)
                self.assertEqual((), report.invalid)
                plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
                outcome = RunSupervisor(root, default_config(root)).run(plan)
                records = {record.id: record for record in outcome.manifest.collectors}
                first = records["core.system.a_first"]
                second = records["core.system.z_second"]

                self.assertEqual("succeeded", first.status, first.errors)
                self.assertEqual("succeeded", second.status, second.errors)
                self.assertLessEqual(first.finished_at, second.started_at)

    def test_independent_resource_classes_preserve_bounded_parallelism(self) -> None:
        """Different noninteractive resource owners can overlap in isolated workers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            fixtures = (
                ("a_disk", ResourceClass.DISK_HEAVY),
                ("z_network", ResourceClass.NETWORK_HEAVY),
            )
            for filename, resource_class in fixtures:
                (core_directory / f"{filename}.py").write_text(
                    _delayed_collector_source(filename, 0.35, resource_class=resource_class),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertEqual("succeeded", records["core.system.a_disk"].status)
            self.assertEqual("succeeded", records["core.system.z_network"].status)
            self.assertLess(records["core.system.z_network"].started_at, records["core.system.a_disk"].finished_at)

    def test_interactive_resource_class_runs_without_any_worker_overlap(self) -> None:
        """Interactive collectors require an exclusive deterministic scheduler window."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            fixtures = (
                ("a_disk", ResourceClass.DISK_HEAVY),
                ("m_interactive", ResourceClass.INTERACTIVE),
                ("z_general", ResourceClass.GENERAL),
            )
            for filename, resource_class in fixtures:
                (core_directory / f"{filename}.py").write_text(
                    _delayed_collector_source(filename, 0.15, resource_class=resource_class),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=3, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertLessEqual(
                records["core.system.a_disk"].finished_at,
                records["core.system.m_interactive"].started_at,
            )
            self.assertLessEqual(
                records["core.system.m_interactive"].finished_at,
                records["core.system.z_general"].started_at,
            )

    def test_dependencies_finish_before_dependents_start(self) -> None:
        """Topological order must become an execution barrier under bounded parallelism."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            dependency_id = "core.system.a_dependency"
            (core_directory / "a_dependency.py").write_text(
                _delayed_collector_source("a_dependency", 0.3),
                encoding="utf-8",
            )
            (core_directory / "b_dependent.py").write_text(
                _delayed_collector_source("b_dependent", 0.0, dependencies=(dependency_id,)),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            dependency = records[dependency_id]
            dependent = records["core.system.b_dependent"]
            self.assertEqual("succeeded", dependent.status, dependent.errors)
            self.assertIsNotNone(dependency.finished_at)
            self.assertIsNotNone(dependent.started_at)
            self.assertLessEqual(dependency.finished_at, dependent.started_at)

    def test_failed_dependency_skips_dependent_without_launching_it(self) -> None:
        """A failed prerequisite must contain failure and prevent dependent execution."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            dependency_id = "core.system.a_dependency"
            (core_directory / "a_dependency.py").write_text(
                _delayed_collector_source("a_dependency", 0.0, fail=True),
                encoding="utf-8",
            )
            (core_directory / "b_dependent.py").write_text(
                _delayed_collector_source("b_dependent", 0.0, dependencies=(dependency_id,)),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertEqual("failed", records[dependency_id].status)
            dependent = records["core.system.b_dependent"]
            self.assertEqual("skipped", dependent.status)
            self.assertIsNone(dependent.started_at)
            self.assertTrue(any(dependency_id in error for error in dependent.errors))

    def test_failed_collector_is_manifested_and_never_reported_as_success(self) -> None:
        """An isolated collector crash must produce a durable failed run outcome."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    'output = context.workspace / "system.txt"\n'
                    '        output.write_text("ok\\n", encoding="utf-8")\n'
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")\n'
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    'raise RuntimeError("intentional test collector failure")',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            plan = build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            self.assertEqual("failed", outcome.manifest.status.value)
            record = outcome.manifest.collectors[0]
            self.assertEqual("failed", record.status)
            self.assertTrue(any("RuntimeError" in error for error in record.errors))
            self.assertIsNotNone(outcome.manifest.package)
            package_path = Path(outcome.manifest.package["path"])
            self.assertTrue(package_path.is_file())
            with zipfile.ZipFile(package_path) as archive:
                summary = archive.read("summary.txt").decode("utf-8")
            self.assertIn("Status: failed", summary)
            self.assertIn("Reasons:", summary)
            self.assertIn("RuntimeError", summary)
            self.assertIn(f"Started: {record.started_at}", summary)
            self.assertIn(f"Finished: {record.finished_at}", summary)

    def test_transient_collector_failure_retries_in_a_new_isolated_worker(self) -> None:
        """An explicitly retryable collector may recover before any evidence is registered."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n'
                    '            maximum_retries=1,\n'
                    '            retry_delay_seconds=0.01,',
                ).replace(
                    '        output = context.workspace / "system.txt"',
                    '        marker = context.workspace / "attempt.marker"\n'
                    '        if not marker.exists():\n'
                    '            marker.write_text("first attempt failed", encoding="utf-8")\n'
                    '            raise RuntimeError("temporary collector failure")\n'
                    '        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)

            record = outcome.manifest.collectors[0]
            self.assertEqual("succeeded", record.status, record.errors)
            self.assertIsNone(record.failure)
            self.assertEqual(2, record.attempt_count)
            self.assertEqual(1, len(record.retry_history))
            self.assertEqual("collect", record.retry_history[0]["failure"]["operation"])
            self.assertTrue(record.retry_history[0]["failure"]["retry_safe"])
            self.assertTrue(
                any("temporary collector failure" in error for error in record.retry_history[0]["errors"])
            )
            self.assertIsInstance(record.worker_pid, int)
            self.assertIsInstance(record.retry_history[0]["worker_pid"], int)
            self.assertNotEqual(record.worker_pid, record.retry_history[0]["worker_pid"])
            self.assertEqual("failed", record.retry_history[0]["termination_reason"])
            self.assertEqual("completed", record.termination_reason)
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                packaged_record = json.loads(archive.read("manifest.json"))["collectors"][0]
                summary = archive.read("summary.txt").decode("utf-8")
            self.assertEqual(2, packaged_record["attempt_count"])
            self.assertEqual(1, len(packaged_record["retry_history"]))
            self.assertIsNone(packaged_record["failure"])
            self.assertTrue(packaged_record["retry_history"][0]["failure"]["retry_safe"])
            self.assertIn("Attempts: 2", summary)

    def test_retry_policy_stops_after_declared_attempt_limit(self) -> None:
        """A repeatedly failing collector may not exceed its explicitly declared retry cap."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            maximum_retries=1,',
                ).replace(
                    '        output = context.workspace / "system.txt"',
                    '        raise RuntimeError("persistent collector failure")',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]

            self.assertEqual("failed", record.status)
            self.assertEqual(2, record.attempt_count)
            self.assertEqual(1, len(record.retry_history))
            self.assertTrue(any("persistent collector failure" in error for error in record.errors))

    def test_failed_collector_with_registered_evidence_is_never_retried(self) -> None:
        """Evidence-producing failures must not be repeated or overwrite forensic state."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            maximum_retries=2,',
                ).replace(
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    '        raise RuntimeError("failure after evidence registration")',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]

            self.assertEqual("failed", record.status)
            self.assertEqual(1, record.attempt_count)
            self.assertEqual([], record.retry_history)
            self.assertTrue((outcome.run_directory / "artifacts" / "core_system_system_info" / "system.txt").is_file())
            self.assertEqual(1, len(record.artifacts))
            self.assertEqual(1, len(outcome.manifest.artifact_list()))
            self.assertEqual("core.system.system_info", record.failure["collector_id"])
            self.assertEqual("collect", record.failure["operation"])
            self.assertIn("failure after evidence registration", record.failure["platform_error"])
            self.assertFalse(record.failure["retry_safe"])
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                self.assertIn("artifacts/core_system_system_info/system.txt", archive.namelist())
                packaged = json.loads(archive.read("manifest.json"))["collectors"][0]
                summary = archive.read("summary.txt").decode("utf-8")
            self.assertEqual(record.failure, packaged["failure"])
            self.assertIn("Failed operation: collect", summary)
            self.assertIn("Retry safe: false", summary)
            self.assertIn("Remediation:", summary)

    def test_crashed_collector_runs_cleanup_and_packages_partial_evidence_and_isolation_results(self) -> None:
        """A failed process finalizes once, preserves evidence, and cannot stop its neighbor."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            failing_source = _delayed_collector_source("a_failed", 0.0).replace(
                '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                '        raise RuntimeError("collection failed after evidence registration")',
            ).replace(
                '        """Release test resources."""',
                '        """Release test resources."""\n'
                '        (context.workspace / "cleanup.marker").write_text("cleaned", encoding="utf-8")\n'
                '        (context.temporary_directory / "scratch.txt").write_text("scratch", encoding="utf-8")',
            )
            (core_directory / "a_failed.py").write_text(failing_source, encoding="utf-8")
            (core_directory / "z_independent.py").write_text(
                _delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}
            failed = records["core.system.a_failed"]
            succeeded = records["core.system.z_independent"]
            workspace = outcome.run_directory / "collectors" / "core_system_a_failed"

            self.assertEqual("partial", outcome.manifest.status.value)
            self.assertEqual("failed", failed.status)
            self.assertEqual("succeeded", succeeded.status)
            self.assertEqual("cleaned", (workspace / "cleanup.marker").read_text(encoding="utf-8"))
            self.assertFalse((workspace / "tmp").exists())
            self.assertEqual(1, len(failed.artifacts))
            self.assertIn("RuntimeError", "\n".join(failed.errors))
            self.assertEqual("process", failed.isolation_mode)
            self.assertIsInstance(failed.worker_pid, int)
            self.assertEqual("failed", failed.termination_reason)
            self.assertEqual("completed", succeeded.termination_reason)
            self.assertNotEqual(failed.worker_pid, succeeded.worker_pid)
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                packaged = {item["id"]: item for item in json.loads(archive.read("manifest.json"))["collectors"]}
                summary = archive.read("summary.txt").decode("utf-8")
                self.assertIn("artifacts/core_system_a_failed/system.txt", archive.namelist())
            self.assertEqual("failed", packaged["core.system.a_failed"]["status"])
            self.assertEqual("process", packaged["core.system.a_failed"]["isolation_mode"])
            self.assertEqual(failed.worker_pid, packaged["core.system.a_failed"]["worker_pid"])
            self.assertEqual("succeeded", packaged["core.system.z_independent"]["status"])
            self.assertIn("Isolation: process", summary)
            self.assertIn("Termination: failed", summary)
            self.assertIn("collection failed after evidence registration", summary)

    def test_cleanup_failure_preserves_original_collector_crash_and_independent_results(self) -> None:
        """Cleanup errors must be reported without replacing the original failure."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            failing_source = _delayed_collector_source("a_failed", 0.0).replace(
                '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                '        raise RuntimeError("original collection failure")',
            ).replace(
                '        """Release test resources."""',
                '        """Release test resources."""\n'
                '        raise ValueError("secondary cleanup failure")',
            )
            (core_directory / "a_failed.py").write_text(failing_source, encoding="utf-8")
            (core_directory / "z_independent.py").write_text(
                _delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}
            errors = "\n".join(records["core.system.a_failed"].errors)

            self.assertEqual("failed", records["core.system.a_failed"].status)
            self.assertIn("RuntimeError: original collection failure", errors)
            self.assertIn("collector cleanup failed: ValueError: secondary cleanup failure", errors)
            self.assertEqual("succeeded", records["core.system.z_independent"].status)
            self.assertEqual("partial", outcome.manifest.status.value)

    def test_cleanup_failure_after_success_preserves_registered_evidence(self) -> None:
        """A failed finalizer turns success into failure without discarding registered bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '        """Release test resources."""',
                    '        """Release test resources."""\n'
                    '        raise RuntimeError("finalizer failed after collecting evidence")',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]

            self.assertEqual("failed", record.status)
            self.assertEqual("collector cleanup failed", record.summary)
            self.assertEqual(1, len(record.artifacts))
            self.assertIn("finalizer failed after collecting evidence", "\n".join(record.errors))
            self.assertEqual("cleanup", record.failure["operation"])
            self.assertIn("cleanup", record.failure["remediation"])
            self.assertFalse(record.failure["retry_safe"])
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                self.assertIn("artifacts/core_system_system_info/system.txt", archive.namelist())

    def test_collector_timeout_preserves_independent_worker_results(self) -> None:
        """A timed-out worker is terminated without cancelling unrelated collection."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            timeout_source = _delayed_collector_source("a_timeout", 2.0).replace(
                '            supported_platforms=("win32",),',
                '            supported_platforms=("win32",),\n            timeout_seconds=1,',
            )
            (core_directory / "a_timeout.py").write_text(timeout_source, encoding="utf-8")
            (core_directory / "z_independent.py").write_text(
                _delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertEqual("failed", records["core.system.a_timeout"].status)
            self.assertTrue(any("timeout" in error for error in records["core.system.a_timeout"].errors))
            self.assertEqual("timeout_exceeded", records["core.system.a_timeout"].termination_reason)
            self.assertTrue(records["core.system.a_timeout"].failure["retry_safe"])
            self.assertIn("timeout", records["core.system.a_timeout"].failure["remediation"])
            self.assertEqual("succeeded", records["core.system.z_independent"].status)
            self.assertIsNone(records["core.system.z_independent"].failure)
            self.assertEqual("partial", outcome.manifest.status.value)

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
                    source = _delayed_collector_source("a_attacker", 0.0).replace(
                        '        output = context.workspace / "system.txt"',
                        f"        ({target}).write_text('compromised', encoding='utf-8')\n"
                        '        output = context.workspace / "system.txt"',
                    )
                    (core_directory / "a_attacker.py").write_text(source, encoding="utf-8")
                    (core_directory / "z_independent.py").write_text(
                        _delayed_collector_source("z_independent", 0.0),
                        encoding="utf-8",
                    )
                    plan = build_plan(preflight(root), RunRequest(max_workers=2, acknowledge_authorization=True))
                    outcome = RunSupervisor(root, default_config(root)).run(plan)
                    records = {record.id: record for record in outcome.manifest.collectors}
                    self.assertEqual("failed", records["core.system.a_attacker"].status)
                    self.assertIn("escapes its private workspace", "\n".join(records["core.system.a_attacker"].errors))
                    self.assertEqual("succeeded", records["core.system.z_independent"].status)
                    self.assertFalse((root / "repository-compromised.txt").exists())
                    self.assertFalse(
                        (outcome.run_directory / "collectors" / "z_independent" / "compromised.txt").exists()
                    )

    def test_collector_cannot_mutate_its_process_environment(self) -> None:
        """A collector must not change process environment or working-directory policy."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace("from pathlib import Path\n", "from pathlib import Path\nimport os\n").replace(
                    '        output = context.workspace / "system.txt"',
                    '        os.environ["LOGICYTICS_COLLECTOR_MUTATION"] = "unexpected"\n'
                    '        output = context.workspace / "system.txt"',
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
            base = _COLLECTOR.replace(
                "from pathlib import Path\n",
                "from pathlib import Path\nimport socket\nimport subprocess\nimport sys\nimport winreg\n",
            )
            attempts = (
                ("subprocess.run([sys.executable, '-c', 'pass'], check=False)", "subprocess capability"),
                ("socket.socket()", "network capability"),
                ("winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Software')", "registry_read capability"),
            )
            for operation, message in attempts:
                with self.subTest(operation=operation):
                    collector_path.write_text(
                        base.replace(
                            '        output = context.workspace / "system.txt"',
                            f"        {operation}\n        output = context.workspace / \"system.txt\"",
                        ),
                        encoding="utf-8",
                    )
                    plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
                    outcome = RunSupervisor(root, default_config(root)).run(plan)
                    self.assertEqual("failed", outcome.manifest.collectors[0].status)
                    self.assertIn(message, "\n".join(outcome.manifest.collectors[0].errors))
                    failure = outcome.manifest.collectors[0].failure
                    self.assertEqual("access", failure["operation"])
                    self.assertIn("capability", failure["remediation"])
                    self.assertFalse(failure["retry_safe"])

    def test_worker_enforces_external_browser_sensitive_and_private_key_read_capabilities(self) -> None:
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
                (browser, (Capability.FILESYSTEM_READ, Capability.BROWSER_DATA), "sensitive_files capability"),
                (
                    private_key,
                    (Capability.FILESYSTEM_READ, Capability.SENSITIVE_FILES),
                    "private_keys capability",
                ),
                (
                    private_key,
                    (Capability.FILESYSTEM_READ, Capability.SENSITIVE_FILES, Capability.PRIVATE_KEYS),
                    None,
                ),
            )
            for evidence, capabilities, denied in cases:
                with self.subTest(evidence=evidence.name, capabilities=capabilities):
                    declared = ", ".join(f"Capability.{capability.name}" for capability in capabilities)
                    if len(capabilities) == 1:
                        declared += ","
                    collector_path.write_text(
                        _COLLECTOR.replace(
                            "from pathlib import Path\n",
                            "from pathlib import Path\nfrom logicytics import Capability\n",
                        ).replace(
                            '            supported_platforms=("win32",),',
                            f'            supported_platforms=("win32",),\n'
                            f'            capabilities=({declared}),',
                        ).replace(
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
                            approved_capabilities=capabilities,
                        ),
                    )
                    outcome = RunSupervisor(root, default_config(root)).run(plan)
                    record = outcome.manifest.collectors[0]
                    if denied is None:
                        self.assertEqual("succeeded", record.status, record.errors)
                    else:
                        self.assertEqual("failed", record.status)
                        self.assertIn(denied, "\n".join(record.errors))

    def test_worker_rejects_raw_packet_socket_without_packet_capture_capability(self) -> None:
        """General network approval must not silently authorize raw packet capture."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    "from pathlib import Path\n",
                    "from pathlib import Path\nimport socket\nfrom logicytics import Capability\n",
                ).replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            capabilities=(Capability.NETWORK,),',
                ).replace(
                    '        output = context.workspace / "system.txt"',
                    '        socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)\n'
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

    def test_worker_allows_an_explicitly_declared_and_approved_subprocess(self) -> None:
        """A declared subprocess capability remains usable after explicit request approval."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    "from pathlib import Path\n",
                    "from pathlib import Path\nimport subprocess\nimport sys\nfrom logicytics import Capability\n",
                ).replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            capabilities=(Capability.SUBPROCESS,),',
                ).replace(
                    '        output = context.workspace / "system.txt"',
                    '        subprocess.run([sys.executable, "-c", "pass"], check=True)\n'
                    '        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )
            plan = build_plan(
                preflight(root),
                RunRequest(
                    max_workers=1,
                    acknowledge_authorization=True,
                    approved_capabilities=(Capability.SUBPROCESS,),
                ),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            self.assertEqual("succeeded", outcome.manifest.collectors[0].status)

    @unittest.skipUnless(os.name == "nt", "Windows collector subprocess-tree containment")
    def test_collector_timeout_terminates_spawned_subprocesses_without_stopping_peers(self) -> None:
        """A timed-out worker cannot leave its child alive or terminate independent collectors."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            source = _delayed_collector_source("a_tree", 0.0).replace(
                "from pathlib import Path\n",
                "from pathlib import Path\nimport subprocess\nimport sys\nfrom logicytics import Capability\n",
            ).replace(
                '            supported_platforms=("win32",),',
                '            supported_platforms=("win32",),\n'
                '            capabilities=(Capability.SUBPROCESS,),\n            timeout_seconds=2,',
            ).replace(
                '        output = context.workspace / "system.txt"',
                '        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])\n'
                '        (context.workspace / "child.pid").write_text(str(child.pid), encoding="ascii")\n'
                '        sleep(5)\n'
                '        output = context.workspace / "system.txt"',
            )
            (core_directory / "a_tree.py").write_text(source, encoding="utf-8")
            (core_directory / "z_independent.py").write_text(
                _delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )
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
            child_pid = int(
                (outcome.run_directory / "collectors" / "core_system_a_tree" / "child.pid").read_text(
                    encoding="ascii"
                )
            )
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, child_pid)
            if handle:
                try:
                    exit_code = ctypes.c_ulong()
                    self.assertTrue(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)))
                    self.assertNotEqual(259, exit_code.value, "collector subprocess survived worker termination")
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)

            self.assertEqual("failed", records["core.system.a_tree"].status)
            self.assertEqual("timeout_exceeded", records["core.system.a_tree"].termination_reason)
            self.assertEqual("succeeded", records["core.system.z_independent"].status)

    @unittest.skipUnless(os.name == "nt", "Windows working-set enforcement test")
    def test_collector_memory_limit_preserves_independent_worker_results(self) -> None:
        """A memory-limit violation terminates only the offending isolated worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            limited_source = _delayed_collector_source("a_limited", 2.0).replace(
                '            supported_platforms=("win32",),',
                '            supported_platforms=("win32",),\n            maximum_memory_bytes=1,',
            )
            (core_directory / "a_limited.py").write_text(limited_source, encoding="utf-8")
            (core_directory / "z_independent.py").write_text(
                _delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            limited = records["core.system.a_limited"]
            self.assertEqual("failed", limited.status)
            self.assertGreater(limited.peak_memory_bytes, 1)
            self.assertTrue(any("maximum_memory_bytes=1" in error for error in limited.errors))
            self.assertEqual("memory_limit_exceeded", limited.termination_reason)
            self.assertEqual("collect", limited.failure["operation"])
            self.assertIn("memory limit", limited.failure["remediation"])
            self.assertTrue(limited.failure["retry_safe"])
            self.assertEqual("succeeded", records["core.system.z_independent"].status)
            self.assertEqual("partial", outcome.manifest.status.value)

    def test_cancelled_run_writes_a_recoverable_package_and_manifest(self) -> None:
        """Keyboard cancellation must affect only this run and preserve its partial report."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            report = preflight(root)
            plan = build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            supervisor = RunSupervisor(root, default_config(root))
            with patch.object(supervisor, "_supervise", side_effect=KeyboardInterrupt):
                outcome = supervisor.run(plan)
            self.assertEqual("cancelled", outcome.manifest.status.value)
            record = outcome.manifest.collectors[0]
            self.assertEqual("cancelled", record.status)
            self.assertEqual("run cancelled by user", record.summary)
            self.assertTrue((outcome.run_directory / ".cancelled").is_file())
            self.assertIsNotNone(outcome.manifest.package)
            package_path = Path(outcome.manifest.package["path"])
            self.assertTrue(package_path.is_file())
            with zipfile.ZipFile(package_path) as archive:
                summary = archive.read("summary.txt").decode("utf-8")
            self.assertIn("Status: cancelled", summary)
            self.assertIn("Summary: run cancelled by user", summary)
            self.assertIn("Started: not started", summary)
            self.assertIn(f"Finished: {record.finished_at}", summary)

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

    def test_preflight_rejects_validation_filesystem_process_network_and_environment_side_effects(self) -> None:
        """No validation probe may mutate files, launch processes, open sockets, or alter its environment."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            outside = root / "outside.marker"
            source = _COLLECTOR.replace(
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
                    self.assertIn("validate() must not perform side effects", report.invalid[0].runtime_error or "")
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
                ("context.artifacts.register_file(context.workspace / 'evidence.txt')", "must not register artifacts"),
                ("raise RuntimeError('unexpected validation failure')", "unexpected validation failure"),
            )
            for statement, message in attempts:
                with self.subTest(message=message):
                    collector_path.write_text(
                        _COLLECTOR.replace(
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
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            with patch(
                    "logicytics.discovery.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(["validation-worker"], timeout=10),
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

    def test_elevated_collectors_require_approval_and_administrator_privileges(self) -> None:
        """Privilege-sensitive collection must fail closed before launching a worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            secured_collector = _COLLECTOR.replace(
                "from logicytics import CollectorMetadata",
                "from logicytics import Capability, CollectorMetadata",
            ).replace(
                'supported_platforms=("win32",),',
                'supported_platforms=("win32",), capabilities=(Capability.ELEVATED_PRIVILEGES,),',
            )
            collector_path.write_text(secured_collector, encoding="utf-8")
            report = preflight(root)
            with self.assertRaisesRegex(PlanError, "unapproved capabilities"):
                build_plan(report, RunRequest())
            approved = RunRequest(approved_capabilities=(Capability.ELEVATED_PRIVILEGES,))
            for administrator_state in (False, None):
                with self.subTest(administrator_state=administrator_state):
                    environment = EnvironmentReport(administrator_state, True, None)
                    with patch("logicytics.planner.inspect_environment", return_value=environment):
                        with self.assertRaisesRegex(PlanError, "administrator account"):
                            build_plan(report, approved)
            with patch(
                    "logicytics.planner.inspect_environment",
                    return_value=EnvironmentReport(True, True, None),
            ):
                plan = build_plan(report, approved)
            self.assertEqual(1, len(plan.collectors))

    def test_unselected_privileged_plugin_never_requests_host_elevation(self) -> None:
        """An opt-in privileged plugin must not affect an ordinary core-only plan."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_path = root / "plugins" / "admin_plugin.py"
            plugin_path.parent.mkdir(parents=True)
            source = _COLLECTOR.replace("CoreCollector", "PluginCollector")
            source = source.replace("SystemInfoCollector", "AdminPluginCollector")
            source = source.replace("core.system.system_info", "plugin.admin_plugin")
            source = source.replace(
                "from logicytics import CollectorMetadata",
                "from logicytics import Capability, CollectorMetadata",
            ).replace(
                'supported_platforms=("win32",),',
                'supported_platforms=("win32",), capabilities=(Capability.ELEVATED_PRIVILEGES,),',
            )
            plugin_path.write_text(source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual((), report.invalid)
            with patch("logicytics.planner.inspect_environment") as inspect:
                plan = build_plan(report, RunRequest())
            self.assertEqual((), plan.collectors)
            inspect.assert_not_called()
            enabled = RunRequest(
                enable_plugins=True,
                approved_capabilities=(Capability.ELEVATED_PRIVILEGES,),
            )
            with patch(
                    "logicytics.planner.inspect_environment",
                    return_value=EnvironmentReport(False, True, None),
            ):
                with self.assertRaisesRegex(PlanError, "administrator account"):
                    build_plan(report, enabled)


if __name__ == "__main__":
    unittest.main()
