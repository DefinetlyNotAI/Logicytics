"""Integration-style checks for the v4 core without shipped collectors."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from logicytics import (
    load_configuration,
    open_artifact,
    plan_run,
    query_run,
    read_artifact,
    run_collection,
)
from logicytics import packaging
from logicytics.artifacts import WorkspaceArtifactWriter
from logicytics.cli import launch_action_window, parser, request, main
from logicytics.command_runner import parse_level_messages, run_command
from logicytics.configuration import (
    LoggingSettings,
    MaintenanceSettings,
    default_config,
    load_config,
)
from logicytics.contracts import (
    Artifact,
    Capability,
    CollectorMetadata,
    CollectorResult,
    CollectorStatus,
    EvidenceKind,
    OutputPolicy,
    PostRunAction,
    ResourceClass,
    RunRequest,
    RunStatus,
    Specialty,
)
from logicytics.discovery import PreflightReport, preflight
from logicytics.environment import EnvironmentReport
from logicytics.errors import ArtifactError, PlanError, PreflightError
from logicytics.file_listing import list_files
from logicytics.interaction import load_history, match_flag, usage_statistics
from logicytics.logging import (
    ApplicationLogger,
    FileEventLogger,
    deprecated,
    get_application_logger,
    get_event_logger,
    raise_logged,
    timed,
)
from logicytics.maintenance import (
    build_manifest,
    compare_files,
    compare_versions,
    fetch_remote_manifest,
    project_files,
    write_local_manifest,
)
from logicytics.manifest import write_manifest
from logicytics.modes import EXECUTION_MODES, LEGACY_MODE_ALIASES, mode_matrix
from logicytics.output_layout import ensure_output_layout
from logicytics.packaging import package_run
from logicytics.planner import BUILTIN_PROFILES, build_plan
from logicytics.runtime import RunSupervisor
from logicytics.sysinternals import ensure_sysinternals

_COLLECTOR = '''"""Create a harmless test artifact."""

from pathlib import Path

from logicytics import CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics import EstimatedCost, NetworkAccess, PrivilegeLevel
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
            output_media_types=("text/plain",),
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


def _plugin_collector_source() -> str:
    """Convert the core fixture into a plugin with every security field explicit."""
    return _COLLECTOR.replace("CoreCollector", "PluginCollector").replace(
        '            author="tests",',
        '            author="tests",\n'
        '            capabilities=(),\n'
        '            privilege_level=PrivilegeLevel.STANDARD,\n'
        '            sensitive_data_categories=(),\n'
        '            network_access=NetworkAccess.NONE,\n'
        '            estimated_cost=EstimatedCost.LOW,\n'
        '            timeout_seconds=60,\n'
        '            maximum_output_bytes=100 * 1024 * 1024,\n'
        '            minimum_contract_version="4.0",',
    )


def _mod_metadata(name: str, *, filesystem_write: bool = False) -> dict[str, object]:
    """Return a complete sidecar declaration for a harmless legacy script fixture."""
    return {
        "id": f"mod.{name}",
        "name": f"{name} mod",
        "version": "1.0.0",
        "specialty": "integration",
        "description": "Harmless isolated legacy script fixture.",
        "author": "tests",
        "supported_platforms": [sys.platform],
        "capabilities": [
            "subprocess",
            *(["filesystem_write"] if filesystem_write else []),
        ],
        "privilege_level": "standard",
        "sensitive_data_categories": [],
        "network_access": "none",
        "estimated_cost": "low",
        "timeout_seconds": 15,
        "maximum_output_bytes": 1024 * 1024,
        "maximum_artifact_files": 10,
        "output_media_types": ["text/plain"],
        "minimum_contract_version": "4.0",
        "default_profiles": ["standard"],
    }


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
    def test_application_logging_levels_colors_retention_and_dispatch_are_bounded(self) -> None:
        """The application sink is typed, redacted, reusable, colored, and size bounded."""

        class TerminalBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "output" / "logs" / "Logicytics.log"
            path.parent.mkdir(parents=True)
            path.write_text("previous secret\n", encoding="utf-8")
            old = path.parent / "Logicytics-old.log"
            old.write_text("expired\n", encoding="utf-8")
            os.utime(old, (0, 0))
            console = TerminalBuffer()
            settings = LoggingSettings(
                level="DEBUG",
                console_enabled=True,
                color_enabled=True,
                maximum_bytes=1024,
                delete_previous=True,
                retention_days=1,
            )
            logger = ApplicationLogger(path, settings, console=console)
            for level in (
                    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "INTERNAL", "EXCEPTION"
            ):
                logger.event(level, "typed event", password="hidden", sequence=1)
            logger.dispatch(("WARNING: parsed warning", "plain batch row"))
            logger.raw("raw access_token=hidden")
            logger.separator()
            for index in range(40):
                logger.event("INFO", "bounded row " + str(index) + " " + "x" * 80)

            contents = path.read_text(encoding="utf-8")
            self.assertNotIn("previous secret", contents)
            self.assertNotIn("hidden", contents)
            self.assertLessEqual(path.stat().st_size, settings.maximum_bytes)
            self.assertFalse(old.exists())
            self.assertIn("\033[", console.getvalue())
            self.assertIn("[EXCEPTION]", console.getvalue())
            with self.assertRaisesRegex(ValueError, "unsupported log level"):
                logger.event("TRACE", "unsupported")
            with self.assertRaisesRegex(ValueError, "raw log end"):
                logger.raw("bad", end="\r\n")

            shared = get_application_logger(path, LoggingSettings(console_enabled=False))
            self.assertIs(
                shared,
                get_application_logger(path, LoggingSettings(console_enabled=False)),
            )
            event_path = root / "output" / "data" / "run-test" / "logs" / "engine.jsonl"
            engine = get_event_logger(event_path, run_id="run-test")
            self.assertIs(engine, get_event_logger(event_path, run_id="run-test"))
            collector = get_event_logger(
                event_path.parent / "collector.jsonl",
                run_id="run-test",
                collector_id="core.system.system_info",
            )
            self.assertIs(
                collector,
                get_event_logger(
                    event_path.parent / "collector.jsonl",
                    run_id="run-test",
                    collector_id="core.system.system_info",
                ),
            )
            self.assertIsNot(engine, collector)

    def test_output_layout_and_logging_configuration_are_complete_and_validated(self) -> None:
        """Every global output directory and logging policy value has one typed source."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = ensure_output_layout(root / "output" / "data")
            for directory in (
                    layout.data,
                    layout.logs,
                    layout.debug_logs,
                    layout.performance_logs,
                    layout.packages,
                    layout.hashes,
            ):
                self.assertTrue(directory.is_dir())
            config_path = root / "logicytics.json"
            config_path.write_text(
                json.dumps({
                    "schema_version": 4,
                    "logging": {
                        "level": "debug",
                        "console_enabled": False,
                        "color_enabled": False,
                        "file_enabled": True,
                        "maximum_bytes": 2048,
                        "delete_previous": True,
                        "retention_days": 7,
                    },
                }),
                encoding="utf-8",
            )
            configuration = load_config(root, config_path)
            self.assertEqual("DEBUG", configuration.logging.level)
            self.assertEqual(2048, configuration.logging.maximum_bytes)
            self.assertEqual(7, configuration.logging.retention_days)

            for invalid_logging in (
                    {"level": "TRACE"},
                    {"maximum_bytes": True},
                    {"retention_days": -1},
                    {"console_enabled": 1},
                    {"unknown": True},
            ):
                config_path.write_text(
                    json.dumps({"schema_version": 4, "logging": invalid_logging}),
                    encoding="utf-8",
                )
                with self.subTest(logging=invalid_logging), self.assertRaises(PlanError):
                    load_config(root, config_path)

    def test_maintenance_configuration_requires_pinned_https_and_python_order(
            self,
    ) -> None:
        """Integrity endpoints and Python policy fail closed during configuration loading."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "logicytics.json"
            valid = {
                "schema_version": 4,
                "maintenance": {
                    "remote_manifest_url": "https://example.invalid/project.manifest.json",
                    "remote_manifest_sha256": "a" * 64,
                    "local_manifest_path": "integrity/project.json",
                    "minimum_python": "3.11",
                    "recommended_python": "3.12",
                },
            }
            config.write_text(json.dumps(valid), encoding="utf-8")
            loaded = load_config(root, config)
            self.assertEqual(
                Path("integrity/project.json"), loaded.maintenance.local_manifest_path
            )
            self.assertEqual("3.12", loaded.maintenance.recommended_python)

            for mutation in (
                    {"remote_manifest_url": "http://example.invalid/project.json"},
                    {"remote_manifest_sha256": None},
                    {"local_manifest_path": "../escape.json"},
                    {"minimum_python": "3.12", "recommended_python": "3.11"},
            ):
                invalid = json.loads(json.dumps(valid))
                invalid["maintenance"].update(mutation)
                config.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(PlanError):
                    load_config(root, config)

    def test_authenticated_remote_manifest_is_strict_data_only_configuration(
            self,
    ) -> None:
        """Pinned HTTPS bytes are accepted, while unpinned or execution-bearing data is rejected."""
        payload = json.dumps(
            {
                "schema_version": 1,
                "version": "4.1.0-snapshot.2",
                "files": {"logicytics/cli.py": "a" * 64},
            }
        ).encode("utf-8")
        settings = MaintenanceSettings(
            remote_manifest_url="https://example.invalid/project.manifest.json",
            remote_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = payload
        with patch("logicytics.maintenance.urllib.request.urlopen", return_value=response):
            manifest = fetch_remote_manifest(settings)
        self.assertIsNotNone(manifest)
        self.assertEqual("4.1.0-snapshot.2", manifest.version)

        tampered = MaintenanceSettings(
            remote_manifest_url=settings.remote_manifest_url,
            remote_manifest_sha256="0" * 64,
        )
        with patch("logicytics.maintenance.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "does not match"):
                fetch_remote_manifest(tampered)

        execution_payload = json.dumps(
            {
                "schema_version": 1,
                "version": "4.1.0",
                "files": {},
                "collectors": ["remote.code"],
            }
        ).encode("utf-8")
        execution_response = MagicMock()
        execution_response.__enter__.return_value.read.return_value = execution_payload
        execution_settings = MaintenanceSettings(
            remote_manifest_url=settings.remote_manifest_url,
            remote_manifest_sha256=hashlib.sha256(execution_payload).hexdigest(),
        )
        with patch(
                "logicytics.maintenance.urllib.request.urlopen",
                return_value=execution_response,
        ):
            with self.assertRaisesRegex(ValueError, "only schema_version"):
                fetch_remote_manifest(execution_settings)

    def test_integrity_manifest_comparison_and_snapshot_version_ordering(self) -> None:
        """Developer integrity reports file states and compare snapshots semantically."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = MaintenanceSettings()
            (root / "same.txt").write_text("same", encoding="utf-8")
            (root / "changed.txt").write_text("before", encoding="utf-8")
            (root / "removed.txt").write_text("remove", encoding="utf-8")
            manifest = build_manifest(root, settings, "4.1.0-snapshot.2")
            write_local_manifest(root, settings, manifest)
            (root / "changed.txt").write_text("after", encoding="utf-8")
            (root / "removed.txt").unlink()
            (root / "added.txt").write_text("add", encoding="utf-8")
            comparison = compare_files(root, settings, manifest)
            self.assertEqual(["removed.txt"], comparison["missing"])
            self.assertEqual(["changed.txt"], comparison["modified"])
            self.assertEqual(["added.txt"], comparison["extra"])
            self.assertEqual(["same.txt"], comparison["unchanged"])
            self.assertEqual("behind", compare_versions("4.1.0-snapshot.2", "4.1.0"))
            self.assertEqual(
                "behind", compare_versions("4.1.0-snapshot.2", "4.1.0-snapshot.10")
            )
            cache = root / ".pytest_cache" / "ignored.txt"
            cache.parent.mkdir()
            cache.write_text("ignored", encoding="utf-8")
            tool = root / "tools" / "Sysinternals" / "ignored.exe"
            tool.parent.mkdir(parents=True)
            tool.write_bytes(b"ignored")
            names = {path.relative_to(root).as_posix() for path in project_files(root, settings)}
            self.assertNotIn(".pytest_cache/ignored.txt", names)
            self.assertNotIn("tools/Sysinternals/ignored.exe", names)

    def test_dev_writes_explicit_manifest_and_debug_persists_diagnostics(self) -> None:
        """Side actions remain separate from collection and write only their dedicated artifacts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text(
                '[project]\nname = "fixture"\nversion = "4.0.0"\n',
                encoding="utf-8",
            )
            with patch("logicytics.cli._project_root", return_value=root), patch(
                    "sys.stdout", new_callable=io.StringIO
            ):
                self.assertEqual(0, main(["dev", "--write-manifest", "--next-version", "4.1.0"]))
            manifest_path = root / "project.manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("4.1.0", manifest_payload["version"])

            with patch("logicytics.cli._project_root", return_value=root), patch(
                    "sys.stdout", new_callable=io.StringIO
            ):
                self.assertEqual(0, main(["debug"]))
            debug_path = root / "output" / "logs" / "debug" / "debug.json"
            debug = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertEqual("4.0.0", debug["maintenance"]["local_version"])
            self.assertIn(
                debug["maintenance"]["python_support"]["status"],
                {"recommended", "supported", "incompatible"},
            )
            self.assertEqual("missing", debug["sysinternals"]["status"])

    def test_dev_updates_legacy_ini_manifest_after_explicit_confirmation(self) -> None:
        """Legacy dev mode preserves comments and updates only its version and file list."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text(
                '[project]\nname = "fixture"\nversion = "4.0.0"\n',
                encoding="utf-8",
            )
            collector = root / "core" / "system" / "example.py"
            collector.parent.mkdir(parents=True)
            collector.write_text('"""Example collector module."""\n', encoding="utf-8")
            code = root / "CODE"
            code.mkdir()
            legacy = code / "config.ini"
            legacy.write_text(
                "# preserve this comment\n"
                "[System Settings]\n"
                "version = 3.6.0\n"
                'files = "old.py"\n\n'
                "[Unrelated]\n"
                "value = untouched\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("logicytics.cli._project_root", return_value=root), patch(
                    "sys.stdout", output
            ):
                self.assertEqual(
                    0,
                    main(["dev", "--write-manifest", "--next-version", "4.1.0"]),
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(str(legacy), payload["manifest_written"])
            self.assertEqual([], payload["checks"]["misplaced_python"])
            updated = legacy.read_text(encoding="utf-8")
            self.assertIn("# preserve this comment", updated)
            self.assertIn("version = 4.1.0", updated)
            self.assertIn("CODE/config.ini", updated)
            self.assertIn("core/system/example.py", updated)
            self.assertIn("value = untouched", updated)
            self.assertNotIn("old.py", updated)
            self.assertFalse((root / "project.manifest.json").exists())

    def test_mods_require_sidecars_and_run_as_isolated_registered_artifacts(self) -> None:
        """Legacy scripts enter the pipeline only through typed metadata and worker isolation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mods = root / "MODS"
            mods.mkdir()
            script = mods / "example.py"
            script.write_text(
                "from pathlib import Path\n"
                "Path('report.txt').write_text('mod evidence\\n', encoding='utf-8')\n"
                "print('INFO: fixture completed')\n",
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("requires metadata sidecar", report.invalid[0].static_errors[0])
            with self.assertRaises(PreflightError):
                build_plan(report, RunRequest(enable_mods=True))

            script.with_suffix(".py.mod.json").write_text(
                json.dumps(_mod_metadata("example")),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            self.assertEqual((), build_plan(report, RunRequest()).collectors)
            plan = build_plan(
                report,
                RunRequest(
                    enable_mods=True,
                    approved_capabilities=(Capability.SUBPROCESS,),
                    acknowledge_authorization=True,
                    max_workers=1,
                ),
            )
            self.assertEqual(["mod.example"], [item.metadata.id for item in plan.collectors])
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]
            self.assertEqual("succeeded", record.status, record.errors)
            self.assertEqual(2, len(record.artifacts))
            report_artifact = next(item for item in record.artifacts if item["name"] == "report.txt")
            artifact_path = outcome.run_directory / "artifacts" / str(report_artifact["relative_path"])
            self.assertEqual("mod evidence\n", artifact_path.read_text(encoding="utf-8"))
            mods_package = Path(str(outcome.manifest.package["mods_path"]))
            mods_hash = Path(str(outcome.manifest.package["mods_sha256_path"]))
            self.assertTrue(mods_package.name.startswith("mods-run-"))
            self.assertTrue(mods_hash.is_file())
            with zipfile.ZipFile(mods_package) as archive:
                names = archive.namelist()
                self.assertIn("metadata/mods.json", names)
                self.assertTrue(any(name.endswith("/report.txt") for name in names))
                self.assertIsNone(archive.testzip())

    def test_python_mod_cannot_mutate_project_configuration_without_write_approval(self) -> None:
        """The Python MOD bootstrap blocks host writes while an independent MOD still succeeds."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration_path = root / "logicytics.json"
            configuration_path.write_text('{"schema_version":4}\n', encoding="utf-8")
            mods = root / "MODS"
            mods.mkdir()
            scripts = {
                "good.py": "from pathlib import Path\nPath('report.txt').write_text('ok', encoding='utf-8')\n",
                "malicious.py": (
                    "from pathlib import Path\n"
                    f"Path({str(configuration_path)!r}).write_text('replaced', encoding='utf-8')\n"
                ),
            }
            for filename, source in scripts.items():
                script = mods / filename
                script.write_text(source, encoding="utf-8")
                script.with_suffix(".py.mod.json").write_text(
                    json.dumps(_mod_metadata(script.stem)),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual(2, len(report.valid), report.invalid)
            plan = build_plan(
                report,
                RunRequest(
                    enable_mods=True,
                    approved_capabilities=(Capability.SUBPROCESS,),
                    acknowledge_authorization=True,
                    max_workers=2,
                ),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}
            self.assertEqual("succeeded", records["mod.good"].status, records["mod.good"].errors)
            self.assertEqual("failed", records["mod.malicious"].status)
            self.assertTrue(
                any("private workspace" in error for error in records["mod.malicious"].errors),
                records["mod.malicious"].errors,
            )
            self.assertEqual('{"schema_version":4}\n', configuration_path.read_text(encoding="utf-8"))

    def test_native_mod_requires_explicit_filesystem_write_declaration(self) -> None:
        """Native child processes are quarantined unless their unconfined write risk is declared."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mods = root / "MODS"
            mods.mkdir()
            script = mods / "native.bat"
            script.write_text("@echo off\n", encoding="utf-8")
            script.with_suffix(".bat.mod.json").write_text(
                json.dumps(_mod_metadata("native")),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("filesystem_write", report.invalid[0].runtime_error)

            script.with_suffix(".bat.mod.json").write_text(
                json.dumps(_mod_metadata("native", filesystem_write=True)),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            with self.assertRaisesRegex(PlanError, "filesystem_write"):
                build_plan(
                    report,
                    RunRequest(
                        enable_mods=True,
                        approved_capabilities=(Capability.SUBPROCESS,),
                    ),
                )

    def test_nopy_and_modded_modes_select_declared_mod_types_without_helpers(self) -> None:
        """Compatibility modes include all MODS or only non-Python MODS deterministically."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mods = root / "MODS"
            mods.mkdir()
            for name, extension in (("python_mod", ".py"), ("batch_mod", ".bat")):
                script = mods / f"{name}{extension}"
                script.write_text("pass\n" if extension == ".py" else "@echo off\n", encoding="utf-8")
                script.with_suffix(extension + ".mod.json").write_text(
                    json.dumps(_mod_metadata(name, filesystem_write=extension != ".py")),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual(2, len(report.valid), report.invalid)
            modded = build_plan(report, RunRequest(
                enable_mods=True,
                approved_capabilities=(Capability.SUBPROCESS, Capability.FILESYSTEM_WRITE),
            ))
            nopy = build_plan(report, RunRequest(
                enable_mods=True,
                non_python_only=True,
                approved_capabilities=(Capability.SUBPROCESS, Capability.FILESYSTEM_WRITE),
            ))
            self.assertEqual(["mod.batch_mod", "mod.python_mod"], [item.metadata.id for item in modded.collectors])
            self.assertEqual(["mod.batch_mod"], [item.metadata.id for item in nopy.collectors])
            parser = parser()
            self.assertTrue(request(parser.parse_args(["run", "--modded"]), 2).enable_mods)
            nopy_request = request(parser.parse_args(["run", "--nopy"]), 2)
            self.assertTrue(nopy_request.enable_mods)
            self.assertTrue(nopy_request.non_python_only)

    @unittest.skipUnless(os.name == "nt", "legacy script adapters require Windows")
    def test_non_python_mod_adapters_execute_powershell_batch_and_executable_files(self) -> None:
        """Every documented non-Python MODS type executes through its explicit adapter."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mods = root / "MODS"
            mods.mkdir()
            scripts = {
                "powershell_mod.ps1": "Set-Content -LiteralPath report.txt -Value 'powershell evidence'\n",
                "batch_mod.bat": "@echo off\r\necho batch evidence>report.txt\r\n",
            }
            for filename, contents in scripts.items():
                script = mods / filename
                script.write_text(contents, encoding="utf-8")
                script.with_suffix(script.suffix + ".mod.json").write_text(
                    json.dumps(_mod_metadata(script.stem, filesystem_write=True)),
                    encoding="utf-8",
                )
            executable = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "whoami.exe"
            self.assertTrue(executable.is_file())
            copied_executable = mods / "identity_mod.exe"
            shutil.copy2(executable, copied_executable)
            copied_executable.with_suffix(".exe.mod.json").write_text(
                json.dumps(_mod_metadata("identity_mod", filesystem_write=True)),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(3, len(report.valid), report.invalid)
            plan = build_plan(
                report,
                RunRequest(
                    enable_mods=True,
                    non_python_only=True,
                    approved_capabilities=(Capability.SUBPROCESS, Capability.FILESYSTEM_WRITE),
                    acknowledge_authorization=True,
                    max_workers=1,
                ),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            self.assertEqual(
                ["succeeded", "succeeded", "succeeded"],
                [record.status for record in outcome.manifest.collectors],
                [record.errors for record in outcome.manifest.collectors],
            )

    def test_semantic_flag_matching_history_usage_and_graph_are_local_and_opt_in(self) -> None:
        """Natural-language actions use configured matching and persist only with consent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()
            with patch("logicytics.cli._project_root", return_value=root), patch("sys.stdout", output):
                self.assertEqual(0, main(["--match", "run a quick basic collection"]))
            payload = json.loads(output.getvalue())
            self.assertEqual("minimal", payload["matched_flag"])
            self.assertFalse(payload["history_persisted"])
            self.assertFalse((root / "output" / "data" / "interaction_history.json.gz").exists())

            (root / "logicytics.json").write_text(
                json.dumps({
                    "schema_version": 4,
                    "interaction": {
                        "history_enabled": True,
                        "similarity_threshold": 0.5,
                        "model_name": "stdlib-test-model",
                        "model_debug": True,
                    },
                }),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("logicytics.cli._project_root", return_value=root), patch("sys.stdout", output):
                self.assertEqual(0, main(["--match", "an exhaustive slow scan"]))
            payload = json.loads(output.getvalue())
            self.assertEqual("depth", payload["matched_flag"])
            self.assertEqual("stdlib-test-model", payload["model_debug"]["model_name"])
            history_path = root / "output" / "data" / "interaction_history.json.gz"
            history = load_history(history_path)
            self.assertEqual(1, len(history))
            self.assertIn("timestamp", history[0])
            self.assertIn("device_name", history[0])

            output = io.StringIO()
            with patch("logicytics.cli._project_root", return_value=root), patch("sys.stdout", output):
                self.assertEqual(0, main(["--usage"]))
            usage = json.loads(output.getvalue())
            self.assertEqual(1, usage["total_interactions"])
            self.assertEqual(1, usage["per_flag_frequency"]["depth"])
            graph_path = Path(usage["graph_path"])
            self.assertTrue(graph_path.exists())
            self.assertIn("<svg", graph_path.read_text(encoding="utf-8"))

    def test_semantic_matching_can_fall_back_to_prior_inputs(self) -> None:
        """A weak direct match may reuse a stronger local historical association."""
        history = [{"input": "collect the strange moon report", "matched_flag": "debug"}]
        result = match_flag(
            "collect strange moon report",
            threshold=0.99,
            model_name="stdlib-sequence-matcher",
            history=history,
        )
        self.assertEqual("debug", result.matched_flag)
        self.assertEqual("history", result.source)
        statistics = usage_statistics([{**history[0], "accuracy": 0.9, "device_name": "host"}])
        self.assertEqual("collect the strange moon report", statistics["common_input"])

    def test_interaction_configuration_rejects_unsafe_or_ambiguous_values(self) -> None:
        """Matching and persistence policy must be typed before any history is opened."""
        invalid_sections = (
            {"similarity_threshold": 1.1},
            {"history_enabled": "yes"},
            {"model_debug": 1},
            {"model_name": "bad\nname"},
            {"unknown": True},
        )
        for section in invalid_sections:
            with self.subTest(section=section), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "logicytics.json").write_text(
                    json.dumps({"schema_version": 4, "interaction": section}),
                    encoding="utf-8",
                )
                with self.assertRaises(PlanError):
                    load_config(root)

    def test_plugin_preflight_requires_explicit_security_cost_and_output_metadata(self) -> None:
        """A plugin cannot silently inherit fields that affect consent or scheduling."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_path = root / "plugins" / "example_plugin.py"
            plugin_path.parent.mkdir(parents=True)
            source = _plugin_collector_source().replace("SystemInfoCollector", "ExamplePluginCollector")
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

    def test_post_run_actions_are_typed_exclusive_and_require_verified_packaging(self) -> None:
        """Power actions remain explicit and cannot run before durable package publication."""
        arguments = parser().parse_args([
            "run", "--shutdown", "--performance-check", "--acknowledge-authorization",
        ])
        request = request(arguments, default_workers=4)
        self.assertEqual(PostRunAction.SHUTDOWN, request.post_run_action)
        self.assertEqual(OutputPolicy.PACKAGE, request.output_policy)
        self.assertEqual(1, request.max_workers)
        with self.assertRaises(SystemExit):
            parser().parse_args(["run", "--reboot", "--shutdown"])
        with self.assertRaisesRegex(ValueError, "require packaged output"):
            request(parser().parse_args(["run", "--reboot", "--no-package"]), default_workers=1)

        with tempfile.TemporaryDirectory() as temporary:
            logger = FileEventLogger(Path(temporary) / "events.jsonl", run_id="run-" + "a" * 32)
            manifest = type("Manifest", (), {
                "status": RunStatus.SUCCEEDED,
                "package": {"path": "evidence.zip", "sha256": "a" * 64},
            })()
            completed = subprocess.CompletedProcess(["shutdown"], 0, "", "")
            with patch("logicytics.runtime.process_adapter.run", return_value=completed) as command:
                RunSupervisor._execute_post_run_action(PostRunAction.REBOOT, manifest, logger)
                self.assertEqual("shutdown", command.call_args.args[0][0])
                self.assertEqual("/r", command.call_args.args[0][1])
                self.assertEqual("60", command.call_args.args[0][3])

    def test_preflight_cache_requires_source_interpreter_contract_and_configuration_identity(self) -> None:
        """Only an exact validation context may reuse an isolated runtime probe."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(_COLLECTOR, encoding="utf-8")

            initial = preflight(root, configuration_hash="configuration-a")
            self.assertEqual(1, len(initial.valid), initial.invalid)
            with patch("logicytics.discovery.process_adapter.run", wraps=subprocess.run) as probe:
                cached = preflight(root, configuration_hash="configuration-a")
                self.assertEqual(1, len(cached.valid), cached.invalid)
                probe.assert_not_called()

            with patch("logicytics.discovery.process_adapter.run", wraps=subprocess.run) as probe:
                invalidated = preflight(root, configuration_hash="configuration-b")
                self.assertEqual(1, len(invalidated.valid), invalidated.invalid)
                self.assertEqual(1, probe.call_count)

    def test_public_package_import_does_not_start_runtime_or_create_files(self) -> None:
        """Importing contracts exposes lazy application services without host or disk side effects."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_root = Path(__file__).resolve().parents[1]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(project_root)
            probe = (
                "import pathlib, sys; "
                "before = tuple(pathlib.Path.cwd().iterdir()); "
                "import logicytics; "
                "assert 'logicytics.runtime' not in sys.modules; "
                "assert {'load_configuration', 'plan_run', 'run_collection', 'query_run', 'read_artifact'}"
                ".issubset(logicytics.__all__); "
                "assert tuple(pathlib.Path.cwd().iterdir()) == before"
            )
            result = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual([], list(root.iterdir()))

    def test_public_planning_is_read_only_bounded_and_keeps_plugins_opt_in(self) -> None:
        """The application planner validates configuration and extensions without starting work."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            plugin_path = root / "plugins" / "example_plugin.py"
            plugin_path.parent.mkdir()
            plugin_path.write_text(
                _plugin_collector_source()
                .replace("SystemInfoCollector", "ExamplePluginCollector")
                .replace("core.system.system_info", "plugin.example_plugin"),
                encoding="utf-8",
            )
            (root / "settings.json").write_text(
                '{"schema_version":4,"runtime":{"default_max_workers":1,"maximum_workers":2}}',
                encoding="utf-8",
            )
            configuration = load_configuration(root, "settings.json")
            standard = plan_run(root, RunRequest(max_workers=1), configuration=configuration)
            selected = plan_run(
                root,
                RunRequest(max_workers=1, include=("plugin.example_plugin",)),
                configuration=configuration,
            )

            self.assertEqual(["core.system.system_info"], [item.metadata.id for item in standard.collectors])
            self.assertEqual(
                ["core.system.system_info", "plugin.example_plugin"],
                [item.metadata.id for item in selected.collectors],
            )
            with self.assertRaisesRegex(PlanError, "maximum_workers"):
                plan_run(root, RunRequest(max_workers=3), configuration=configuration)
            with self.assertRaisesRegex(PlanError, "either configuration or config_path"):
                plan_run(root, RunRequest(max_workers=1), configuration=configuration, config_path="settings.json")
            plugin_path.write_text('"""Invalid optional plugin."""\n', encoding="utf-8")
            quarantined = plan_run(root, RunRequest(max_workers=1), configuration=configuration)
            self.assertEqual(["core.system.system_info"], [item.metadata.id for item in quarantined.collectors])
            with self.assertRaises(PreflightError):
                plan_run(
                    root,
                    RunRequest(max_workers=1, include=("plugin.example_plugin",)),
                    configuration=configuration,
                )
            collector_path.write_text('"""Invalid shipped core collector."""\n', encoding="utf-8")
            with self.assertRaises(PreflightError):
                plan_run(root, RunRequest(max_workers=1), configuration=configuration)
            self.assertFalse(configuration.runtime.output_root.exists())

    def test_public_api_runs_queries_and_reads_verified_registered_evidence(self) -> None:
        """The complete public lifecycle retains isolation and immutable manifest-backed status."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            configuration = load_configuration(root)
            request = RunRequest(max_workers=1, acknowledge_authorization=True)
            outcome = run_collection(root, request, configuration=configuration)
            snapshot = query_run(root, outcome.manifest.run_id, configuration=configuration)

            self.assertEqual(RunStatus.SUCCEEDED, snapshot.status, outcome.manifest.package)
            self.assertEqual(outcome.run_directory, snapshot.run_directory)
            self.assertEqual(outcome.manifest_path, snapshot.manifest_path)
            self.assertEqual(1,
                             json.loads(outcome.manifest_path.read_text(encoding="utf-8"))["manifest_schema_version"])
            self.assertEqual(["core.system.system_info"], [item.collector_id for item in snapshot.collectors])
            self.assertEqual(["succeeded"], [item.status for item in snapshot.collectors])
            collector = snapshot.collectors[0]
            self.assertIsNotNone(collector.started_at)
            self.assertIsNotNone(collector.finished_at)
            self.assertIsNotNone(collector.duration_seconds)
            self.assertGreaterEqual(collector.duration_seconds, 0)
            self.assertEqual("test artifact created", collector.summary)
            self.assertEqual((), collector.errors)
            self.assertIsNone(collector.failure)
            self.assertEqual(1, len(snapshot.artifacts))
            self.assertEqual(
                b"ok" + os.linesep.encode("ascii"),
                read_artifact(root, snapshot.run_id, snapshot.artifacts[0].id, configuration=configuration),
            )
            with self.assertRaises(FrozenInstanceError):
                snapshot.status = RunStatus.FAILED
            with self.assertRaises(FrozenInstanceError):
                collector.status = "failed"

    def test_public_run_snapshot_and_cli_expose_verified_collector_failure_and_duration(self) -> None:
        """Callers can inspect redacted lifecycle timing and actionable failure details without parsing manifests."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '        output = context.workspace / "system.txt"',
                    '        raise RuntimeError("fixture lifecycle failure")\n'
                    '        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )
            outcome = run_collection(root, RunRequest(max_workers=1, acknowledge_authorization=True))
            snapshot = query_run(root, outcome.manifest.run_id)
            collector = snapshot.collectors[0]

            self.assertEqual("failed", collector.status)
            self.assertIsNotNone(collector.duration_seconds)
            self.assertGreaterEqual(collector.duration_seconds, 0)
            self.assertIn("worker crashed", collector.summary)
            self.assertTrue(any("fixture lifecycle failure" in error for error in collector.errors))
            self.assertEqual("core.system.system_info", collector.failure.collector_id)
            self.assertEqual("collect", collector.failure.operation)
            self.assertIn("fixture lifecycle failure", collector.failure.platform_error)
            self.assertTrue(collector.failure.remediation)
            self.assertTrue(collector.failure.retry_safe)

            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            output = io.StringIO()
            with patch("logicytics.cli._project_root", return_value=root), patch(
                    "sys.stdout", output
            ), patch("builtins.input", return_value="") as final_prompt:
                exit_code = main(
                    ["run", "--default", "--interactive", "--acknowledge-authorization"]
                )
            self.assertEqual(0, exit_code, output.getvalue())
            final_prompt.assert_called_once_with("Press Enter to exit...")
            self.assertIn("Collectors:", output.getvalue())
            self.assertIn("core.system.system_info status=succeeded duration_seconds=", output.getvalue())
            self.assertIn("summary=test artifact created", output.getvalue())

    def test_public_run_queries_reject_traversal_forged_identity_and_manifest_links(self) -> None:
        """Persisted status lookup cannot leave its configured run or trust a forged manifest."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            outcome = run_collection(root, RunRequest(max_workers=1, acknowledge_authorization=True))
            original = outcome.manifest_path.read_text(encoding="utf-8")

            for invalid in ("../outside", "run-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "run-short"):
                with self.subTest(run_id=invalid):
                    with self.assertRaisesRegex(PlanError, "canonical run identifier"):
                        query_run(root, invalid)
            payload = json.loads(original)
            payload["run_id"] = "run-" + "0" * 32
            outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "identity"):
                query_run(root, outcome.manifest.run_id)
            invalid_manifests = (
                ("manifest_schema_version", 2, "unsupported run manifest schema_version"),
                ("manifest_schema_version", True, "unsupported run manifest schema_version"),
                ("status", [], "unsupported run status"),
                ("resolved_plan", ["core.system.forged"], "resolved plan"),
                ("total_artifact_bytes", 999, "total_artifact_bytes"),
                ("requested_at", "not-a-timestamp", "requested_at"),
                ("finished_at", None, "finished_at"),
            )
            for field, value, message in invalid_manifests:
                with self.subTest(field=field):
                    payload = json.loads(original)
                    payload[field] = value
                    outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        query_run(root, outcome.manifest.run_id)
            payload = json.loads(original)
            payload.pop("manifest_schema_version")
            outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "unsupported run manifest schema_version"):
                query_run(root, outcome.manifest.run_id)
            invalid_collector_fields = (
                ("started_at", "not-a-timestamp", "collector started_at"),
                ("finished_at", None, "require finished_at"),
                ("duration_seconds", -1, "duration_seconds"),
                ("summary", "", "summary"),
                ("errors", [""], "errors"),
                (
                    "failure",
                    {
                        "collector_id": "core.system.system_info",
                        "operation": "collect",
                        "platform_error": "forged",
                        "remediation": "forged",
                        "retry_safe": True,
                    },
                    "must match failed status",
                ),
            )
            for field, value, message in invalid_collector_fields:
                with self.subTest(collector_field=field):
                    payload = json.loads(original)
                    payload["collectors"][0][field] = value
                    outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        query_run(root, outcome.manifest.run_id)
            payload = json.loads(original)
            payload["collectors"][0]["id"] = "../outside"
            payload["resolved_plan"] = ["../outside"]
            outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "collector record"):
                query_run(root, outcome.manifest.run_id)
            outcome.manifest_path.write_text(
                '{"run_id":"' + outcome.manifest.run_id + '","run_id":"duplicate"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PlanError, "duplicate run manifest key"):
                query_run(root, outcome.manifest.run_id)
            outcome.manifest_path.write_text(original, encoding="utf-8")
            outside = root / "outside-manifest.json"
            outside.write_text(original, encoding="utf-8")
            outcome.manifest_path.unlink()
            outcome.manifest_path.symlink_to(outside)
            with self.assertRaisesRegex(PlanError, "escapes its run-owned directory"):
                query_run(root, outcome.manifest.run_id)

    def test_public_artifact_reads_reject_forgery_tampering_escapes_and_unbounded_access(self) -> None:
        """Only exact, bounded, manifest-cataloged bytes from the producing collector may be read."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            outcome = run_collection(root, RunRequest(max_workers=1, acknowledge_authorization=True))
            artifact = outcome.manifest.artifact_list()[0]
            stored = outcome.run_directory / "artifacts" / artifact.relative_path
            original = stored.read_bytes()
            manifest_source = outcome.manifest_path.read_text(encoding="utf-8")

            with patch("logicytics.api.os.startfile") as startfile:
                self.assertEqual(
                    stored,
                    open_artifact(root, outcome.manifest.run_id, artifact.id),
                )
                startfile.assert_called_once_with(stored)

            with self.assertRaisesRegex(ArtifactError, "canonical"):
                read_artifact(root, outcome.manifest.run_id, "../outside")
            with self.assertRaisesRegex(ArtifactError, "not registered"):
                read_artifact(root, outcome.manifest.run_id, "artifact." + "0" * 32)
            for invalid_limit in (True, 0, 67_108_865):
                with self.subTest(maximum_bytes=invalid_limit):
                    with self.assertRaisesRegex(ArtifactError, "maximum_bytes"):
                        read_artifact(root, outcome.manifest.run_id, artifact.id, maximum_bytes=invalid_limit)
            with self.assertRaisesRegex(ArtifactError, "bounded read limit"):
                read_artifact(root, outcome.manifest.run_id, artifact.id, maximum_bytes=1)
            stored.write_bytes(b"bad")
            with self.assertRaisesRegex(ArtifactError, "SHA-256 verification"):
                read_artifact(root, outcome.manifest.run_id, artifact.id)
            with self.assertRaisesRegex(ArtifactError, "size"):
                open_artifact(root, outcome.manifest.run_id, artifact.id)
            stored.write_bytes(original)
            payload = json.loads(manifest_source)
            payload["artifact_catalog"][0]["producer_status"] = "failed"
            outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "artifact catalog"):
                read_artifact(root, outcome.manifest.run_id, artifact.id)
            outcome.manifest_path.write_text(manifest_source, encoding="utf-8")
            outside = root / "outside-evidence.txt"
            outside.write_bytes(original)
            stored.unlink()
            stored.symlink_to(outside)
            with self.assertRaisesRegex(ArtifactError, "collector-owned store"):
                read_artifact(root, outcome.manifest.run_id, artifact.id)
            stored.unlink()
            owner = stored.parent
            owner.rmdir()
            redirected_owner = root / "redirected-owner"
            redirected_owner.mkdir()
            (redirected_owner / stored.name).write_bytes(original)
            owner.symlink_to(redirected_owner, target_is_directory=True)
            with self.assertRaisesRegex(ArtifactError, "collector-owned store"):
                read_artifact(root, outcome.manifest.run_id, artifact.id)

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

    def test_preflight_requires_exact_declared_artifact_media_types(self) -> None:
        """A collector cannot disguise or omit the output contract used by registration."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(
                _COLLECTOR.replace(
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
                _COLLECTOR.replace('            output_media_types=("text/plain",),\n', ""),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertTrue(
                any("explicitly declare" in error for error in report.invalid[0].static_errors),
                report.invalid[0].static_errors,
            )

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

    def test_collector_results_expose_all_strict_typed_terminal_states(self) -> None:
        """Every terminal collector outcome has an explicit, validated result constructor."""
        outcomes = (
            (CollectorResult.succeeded("complete"), CollectorStatus.SUCCEEDED),
            (CollectorResult.partial("incomplete", errors=("one source unavailable",)), CollectorStatus.PARTIAL),
            (CollectorResult.skipped("prerequisite unavailable"), CollectorStatus.SKIPPED),
            (CollectorResult.cancelled("operator cancelled"), CollectorStatus.CANCELLED),
            (CollectorResult.failed("collection failed", errors=("access denied",)), CollectorStatus.FAILED),
        )
        for result, status in outcomes:
            with self.subTest(status=status):
                self.assertIs(status, result.status)
        invalid = (
            ({"status": "succeeded", "summary": "complete"}, "status"),
            ({"status": CollectorStatus.SUCCEEDED, "summary": ""}, "summary"),
            ({"status": CollectorStatus.SUCCEEDED, "summary": "complete", "artifacts": []}, "artifacts"),
            ({"status": CollectorStatus.FAILED, "summary": "failed", "errors": ["failure"]}, "errors"),
            ({"status": CollectorStatus.FAILED, "summary": "failed", "errors": ("",)}, "errors"),
            ({"status": CollectorStatus.PARTIAL, "summary": "partial", "metrics": {"count": True}}, "metrics"),
            (
                {"status": CollectorStatus.PARTIAL, "summary": "partial", "metrics": {"count": float("inf")}},
                "metrics",
            ),
        )
        for options, message in invalid:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    CollectorResult(**options)

    def test_artifact_contract_rejects_malformed_catalog_metadata(self) -> None:
        """Evidence records validate identity, MIME type, provenance, timestamps, and status."""
        valid = {
            "id": "artifact." + "a" * 32,
            "relative_path": "core_system_example/report.json",
            "sha256": "b" * 64,
            "size_bytes": 2,
            "media_type": "application/json",
            "collector_id": "core.system.example",
            "source_category": "system",
            "collected_at": "2026-01-01T00:00:00+00:00",
            "transformations": ("normalized",),
            "evidence_kind": EvidenceKind.DERIVED,
            "name": "report.json",
            "status": "registered",
        }
        invalid = (
            ({"id": "artifact.invalid"}, "id"),
            ({"sha256": "bad"}, "sha256"),
            ({"size_bytes": True}, "size_bytes"),
            ({"size_bytes": -1}, "size_bytes"),
            ({"media_type": "not a mime type"}, "media_type"),
            ({"collector_id": "another"}, "collector_id"),
            ({"source_category": "System"}, "source_category"),
            ({"collected_at": "2026-01-01T00:00:00"}, "timezone"),
            ({"transformations": ["normalized"]}, "transformations"),
            ({"evidence_kind": "derived"}, "evidence_kind"),
            ({"name": "../report.json"}, "name"),
            ({"status": "failed"}, "status"),
        )
        self.assertEqual("report.json", Artifact(**valid).name)
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    Artifact(**{**valid, **changes})

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

    def test_builtin_profiles_select_declared_members_and_reject_unknown_names(self) -> None:
        """Only documented profiles resolve collector-declared membership before any run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            memberships = {
                "a_essential": ("minimal", "standard", "deep", "offline"),
                "b_standard": ("standard", "deep"),
                "z_deep": ("deep",),
            }
            for filename, profiles in memberships.items():
                source = _delayed_collector_source(filename, 0.0).replace(
                    '            supported_platforms=("win32",),',
                    f'            supported_platforms=("win32",),\n            default_profiles={profiles!r},',
                )
                (core_directory / f"{filename}.py").write_text(source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual((), report.invalid)
            expected = {
                "minimal": ["core.system.a_essential"],
                "standard": ["core.system.a_essential", "core.system.b_standard"],
                "deep": ["core.system.a_essential", "core.system.b_standard", "core.system.z_deep"],
                "offline": ["core.system.a_essential"],
            }
            self.assertEqual(set(expected), set(BUILTIN_PROFILES))
            for profile, collector_ids in expected.items():
                with self.subTest(profile=profile):
                    plan = build_plan(report, RunRequest(profile=profile))
                    self.assertEqual(collector_ids, [candidate.metadata.id for candidate in plan.collectors])
            with self.assertRaisesRegex(PlanError, "unknown collection profile"):
                build_plan(report, RunRequest(profile="invented"))
            with patch("sys.stderr", new_callable=io.StringIO) as errors:
                with self.assertRaises(SystemExit):
                    parser().parse_args(["plan", "--profile", "invented"])
            self.assertIn("invalid choice", errors.getvalue())

    def test_offline_profile_rejects_network_collectors_even_with_explicit_approval(self) -> None:
        """Offline collection remains local-only regardless of include and capability overrides."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_id = "core.system.network_source"
            source = _delayed_collector_source("network_source", 0.0).replace(
                "from logicytics import CollectorMetadata",
                "from logicytics import Capability, CollectorMetadata",
            ).replace(
                '            supported_platforms=("win32",),',
                '            supported_platforms=("win32",),\n'
                '            capabilities=(Capability.NETWORK,),\n'
                '            network_access=NetworkAccess.LOCAL,\n'
                '            default_profiles=("standard",),',
            )
            (core_directory / "network_source.py").write_text(source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual((), report.invalid)
            standard = build_plan(report, RunRequest(approved_capabilities=(Capability.NETWORK,)))
            self.assertEqual([collector_id], [candidate.metadata.id for candidate in standard.collectors])
            self.assertEqual((), build_plan(report, RunRequest(profile="offline")).collectors)
            with self.assertRaisesRegex(PlanError, "offline profile prohibits network-capable"):
                build_plan(
                    report,
                    RunRequest(profile="offline", include=(collector_id,), approved_capabilities=(Capability.NETWORK,)),
                )

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
            diagnostic = report.invalid[0].diagnostics[0]
            self.assertEqual("runtime.contract", diagnostic.rule)
            self.assertEqual(str(collector_path), diagnostic.path)
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
            @staticmethod
            def event(level: str, message: str, **fields: object) -> None:
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
            @staticmethod
            def event(level: str, message: str, **fields: object) -> None:
                events.append((level, message, fields))

        with self.assertRaises(ValueError):
            raise_logged(Logger(), ValueError, "invalid setting", setting="workers")
        self.assertEqual("exception", events[0][0])
        self.assertEqual("ValueError", events[0][2]["exception_type"])

    def test_timed_decorator_records_function_lifecycle(self) -> None:
        """Timing instrumentation must report start and finish through EventLogger."""
        events: list[tuple[str, str, dict[str, object]]] = []

        class Logger:
            @staticmethod
            def event(level: str, message: str, **fields: object) -> None:
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
        """Current v4 and explicitly migrated v3 schemas are accepted; other versions are not."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            for unsupported in (0, 1, 2, 5):
                with self.subTest(schema_version=unsupported):
                    config_path.write_text(json.dumps({"schema_version": unsupported}), encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, "unsupported configuration schema_version"):
                        load_config(root)
            config_path.write_text('{"schema_version": 3}', encoding="utf-8")
            migrated = load_config(root)
            self.assertEqual(4, migrated.schema_version)
            self.assertEqual(3, migrated.migrated_from_schema)
            config_path.write_text('{"schema_version": 4, "collectors": {}}', encoding="utf-8")
            current = load_config(root)
            self.assertEqual(4, current.schema_version)
            self.assertIsNone(current.migrated_from_schema)
            config_path.write_text('{"schema_version":4,"runtime":{"package_completed_runs":"yes"}}', encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "package_completed_runs"):
                load_config(root)

    def test_legacy_configuration_migrates_runtime_and_collector_aliases_without_writing(self) -> None:
        """A supported v3 configuration migrates once in memory and preserves its source bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            settings = {"core.packet.packet_capture": {"packet_count": 7, "timeout_seconds": 5}}
            original = json.dumps({
                "schema_version": 3,
                "workers": 3,
                "max_workers": 6,
                "output_root": "ACCESS/RUNS",
                "collector_settings": settings,
            }, indent=2)
            config_path.write_text(original, encoding="utf-8")
            configuration = load_config(root)

            self.assertEqual(4, configuration.schema_version)
            self.assertEqual(3, configuration.migrated_from_schema)
            self.assertEqual(3, configuration.runtime.default_max_workers)
            self.assertEqual(6, configuration.runtime.maximum_workers)
            self.assertEqual(root / "output" / "data", configuration.runtime.output_root)
            self.assertEqual(settings["core.packet.packet_capture"],
                             configuration.settings_for("core.packet.packet_capture"))
            self.assertEqual(original, config_path.read_text(encoding="utf-8"))

    def test_historical_code_config_ini_migrates_into_typed_v4_settings(self) -> None:
        """The original CODE/config.ini is a bounded read-only fallback when JSON is absent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "CODE"
            code.mkdir()
            config_path = code / "config.ini"
            original = """[Settings]
log_using_debug = true
delete_old_logs = true
max_workers = 6
save_preferences = false

[Flag Settings]
accuracy_min = 30.0
model_to_use = all-MiniLM-L6-v2
model_debug = true

[DumpMemory Settings]
file_size_limit = 8
file_size_safety = 1.5

[NetWorkPsutil Settings]
sample_count = 5
interval = 1.5

[PacketSniffer Settings]
interface = WiFi
packet_count = 5000
timeout = 10
max_retry_time = 30
"""
            config_path.write_text(original, encoding="utf-8")

            configuration = load_config(root)

            self.assertEqual(3, configuration.migrated_from_schema)
            self.assertEqual(6, configuration.runtime.default_max_workers)
            self.assertEqual(6, configuration.runtime.maximum_workers)
            self.assertEqual("DEBUG", configuration.logging.level)
            self.assertTrue(configuration.logging.delete_previous)
            self.assertFalse(configuration.interaction.history_enabled)
            self.assertEqual(0.3, configuration.interaction.similarity_threshold)
            self.assertEqual("all-MiniLM-L6-v2", configuration.interaction.model_name)
            self.assertTrue(configuration.interaction.model_debug)
            self.assertEqual(
                {
                    "output_limit_bytes": 8 * 1024 * 1024,
                    "disk_safety_margin_bytes": 4 * 1024 * 1024,
                    "dump_directory": "memory_maps",
                },
                configuration.settings_for("core.process.memory_map"),
            )
            self.assertEqual(
                {"sample_count": 5, "interval_seconds": 1.5},
                configuration.settings_for("core.network.bandwidth_sample"),
            )
            self.assertEqual(5000, configuration.settings_for("core.packet.packet_capture")["packet_count"])
            self.assertEqual(original, config_path.read_text(encoding="utf-8"))

            (root / "logicytics.json").write_text('{"schema_version":4}', encoding="utf-8")
            self.assertIsNone(load_config(root).migrated_from_schema)
            self.assertFalse((root / "output").exists())
            self.assertEqual(3, configuration.to_manifest_dict()["migrated_from_schema"])

    def test_legacy_configuration_rejects_ambiguous_unsafe_and_plugin_enabling_migrations(self) -> None:
        """Migration cannot override settings, weaken validation, or silently enable extensions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            invalid = (
                ({"workers": 2, "worker_count": 3}, "conflicting settings"),
                ({"workers": 2, "runtime": {"default_max_workers": 2}}, "conflicting settings"),
                ({"runtime": {"workers": 2, "worker_count": 3}}, "conflicting settings"),
                ({"collectors": {}, "collector_settings": {}}, "conflicting collectors"),
                ({"enable_plugins": True}, "unsupported root"),
                ({"workers": True}, "worker limits"),
                ({"collector_settings": {"core.process.memory_map": {"dump_directory": "../outside"}}},
                 "collector-workspace"),
                ({"collector_settings": {"core.packet.packet_capture": {"packet_count": 0}}}, "packet_count"),
            )
            for legacy, message in invalid:
                with self.subTest(legacy=legacy):
                    config_path.write_text(json.dumps({"schema_version": 3, **legacy}), encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)
                    self.assertFalse((root / "output").exists())

    def test_migrated_configuration_run_preserves_manifest_provenance_and_legacy_evidence(self) -> None:
        """Migrated collection stays isolated, packages normally, and never moves old evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            legacy = root / "ACCESS" / "RUNS" / "previous.txt"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("keep legacy evidence", encoding="utf-8")
            config_path = root / "logicytics.json"
            original = json.dumps({"schema_version": 3, "worker_count": 1, "output_root": "ACCESS/RUNS"})
            config_path.write_text(original, encoding="utf-8")
            configuration = load_config(root)
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, configuration).run(plan)

            self.assertEqual(root / "output" / "data", outcome.run_directory.parent)
            self.assertEqual(3, outcome.manifest.configuration["migrated_from_schema"])
            self.assertEqual("keep legacy evidence", legacy.read_text(encoding="utf-8"))
            self.assertEqual(original, config_path.read_text(encoding="utf-8"))
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                packaged = json.loads(archive.read("metadata/manifest.json"))
            self.assertEqual(4, packaged["configuration"]["schema_version"])
            self.assertEqual(3, packaged["configuration"]["migrated_from_schema"])

    def test_configuration_defaults_to_one_canonical_output_data_root(self) -> None:
        """Defaults and loaded settings share output/data while explicit roots remain supported."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "output" / "data"
            self.assertEqual(expected, default_config(root).runtime.output_root)
            self.assertEqual(expected, load_config(root).runtime.output_root)
            config_path = root / "logicytics.json"
            config_path.write_text('{"schema_version":4,"runtime":{}}', encoding="utf-8")
            self.assertEqual(expected, load_config(root).runtime.output_root)
            config_path.write_text(
                '{"schema_version":4,"runtime":{"output_root":"custom/evidence"}}',
                encoding="utf-8",
            )
            self.assertEqual(root / "custom" / "evidence", load_config(root).runtime.output_root)
            self.assertFalse(expected.exists())

    def test_configuration_manifest_redacts_nested_secrets_without_mutating_worker_settings(self) -> None:
        """Manifest snapshots hide credentials while collectors retain configured access."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_id = "plugin.private_keys"
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

    def test_configuration_rejects_unknown_duplicate_and_non_finite_values(self) -> None:
        """Ambiguous keys, typos, malformed IDs, and nonstandard JSON fail before planning."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            invalid = (
                ('{"schema_version":4,"unexpected":true}', "unsupported root"),
                ('{"schema_version":4,"runtime":{"worker_typo":2}}', "unsupported settings"),
                ('{"schema_version":4,"schema_version":4}', "duplicate configuration key"),
                ('{"schema_version":4,"runtime":{"maximum_workers":NaN}}', "non-finite"),
                ('{"schema_version":4,"collectors":{"plugin.custom":{"value":1e999}}}', "non-finite"),
                ('{"schema_version":4,"collectors":{"../escape":{}}}', "invalid collector ID"),
                ('{"schema_version":4,"collectors":{"core.system.example":{"invalid-name":1}}}',
                 "invalid setting name"),
                ('{"schema_version":4,"collectors":{"core.packet.packet_capture":{"packet_typo":1}}}',
                 "unsupported settings"),
            )
            for payload, message in invalid:
                with self.subTest(payload=payload):
                    config_path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)
                    self.assertFalse((root / "output").exists())
            config_path.write_text(
                '{"schema_version":4,"collectors":{"plugin.custom":{"extension_setting":"allowed"}}}',
                encoding="utf-8",
            )
            self.assertEqual("allowed", load_config(root).settings_for("plugin.custom")["extension_setting"])

    def test_configuration_validates_filesystem_and_sensitive_inventory_bounds(self) -> None:
        """Traversal and sensitive inventory limits cannot silently coerce or expand in workers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            invalid = (
                ("core.filesystem.system_drive_tree", {"max_entries": True}, "max_entries"),
                ("core.filesystem.system_drive_tree", {"max_depth": 33}, "max_depth"),
                ("core.filesystem.system_drive_listing", {"workers": 2}, "unsupported settings"),
                ("core.filesystem.system_drive_listing", {"max_entries": "100"}, "max_entries"),
                ("core.filesystem.sensitive_file_inventory", {"max_directories": 50_001}, "max_directories"),
                ("core.filesystem.sensitive_file_inventory", {"max_matches": 0}, "max_matches"),
                ("core.filesystem.sensitive_file_inventory", {"root": "relative/path"}, "absolute"),
            )
            for collector_id, settings, message in invalid:
                with self.subTest(collector_id=collector_id, settings=settings):
                    config_path.write_text(
                        json.dumps({"schema_version": 4, "collectors": {collector_id: settings}}),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)
                    self.assertFalse((root / "output").exists())
            valid = {
                "core.filesystem.system_drive_tree": {"max_entries": 100, "max_depth": 3},
                "core.filesystem.system_drive_listing": {"max_entries": 200, "max_depth": 4},
                "core.filesystem.sensitive_file_inventory": {
                    "root": str(root), "max_directories": 100, "max_matches": 10,
                },
            }
            config_path.write_text(json.dumps({"schema_version": 4, "collectors": valid}), encoding="utf-8")
            configuration = load_config(root)
            for collector_id, settings in valid.items():
                self.assertEqual(settings, configuration.settings_for(collector_id))

    def test_configuration_validates_metadata_only_memory_map_limits_and_workspace_paths(self) -> None:
        """The existing metadata-only memory mapper rejects unsafe values before worker launch."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            collector_id = "core.process.memory_map"
            invalid = (
                ({"max_regions": True}, "max_regions"),
                ({"max_regions": 100_001}, "max_regions"),
                ({"output_limit_bytes": 1_023}, "output_limit_bytes"),
                ({"output_limit_bytes": 64 * 1024 * 1024 + 1}, "output_limit_bytes"),
                ({"disk_safety_margin_bytes": -1}, "disk_safety_margin_bytes"),
                ({"disk_safety_margin_bytes": "100"}, "disk_safety_margin_bytes"),
                ({"dump_directory": "../outside"}, "collector-workspace"),
                ({"dump_directory": str(root)}, "collector-workspace"),
                ({"dump_directory": "   "}, "non-empty"),
            )
            for settings, message in invalid:
                with self.subTest(settings=settings):
                    config_path.write_text(
                        json.dumps({"schema_version": 4, "collectors": {collector_id: settings}}),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)
                    self.assertFalse((root / "output").exists())
            settings = {
                "max_regions": 500,
                "output_limit_bytes": 4096,
                "disk_safety_margin_bytes": 0,
                "dump_directory": "bounded/maps",
            }
            config_path.write_text(
                json.dumps({"schema_version": 4, "collectors": {collector_id: settings}}),
                encoding="utf-8",
            )
            self.assertEqual(settings, load_config(root).settings_for(collector_id))

    def test_configuration_profile_collector_and_request_boundaries_are_isolated(self) -> None:
        """Product, profile, collector, and invocation settings remain separate contracts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_id = "core.process.memory_map"
            collector_settings = {
                "output_limit_bytes": 8192,
                "disk_safety_margin_bytes": 4096,
                "dump_directory": "maps",
            }
            (root / "logicytics.json").write_text(
                json.dumps({
                    "schema_version": 4,
                    "runtime": {"default_max_workers": 3, "maximum_workers": 8},
                    "collectors": {collector_id: collector_settings},
                }),
                encoding="utf-8",
            )
            configuration = load_config(root)
            request = RunRequest(
                profile="deep",
                include=(collector_id,),
                max_workers=1,
                acknowledge_authorization=True,
            )

            self.assertEqual(3, configuration.runtime.default_max_workers)
            self.assertEqual(collector_settings, configuration.settings_for(collector_id))
            self.assertEqual("deep", request.profile)
            self.assertEqual((collector_id,), request.include)
            self.assertEqual(1, request.max_workers)
            self.assertNotIn("profile", configuration.to_manifest_dict()["runtime"])
            self.assertNotIn("max_workers", configuration.to_manifest_dict()["runtime"])

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
            ({"rerun_from": "../outside", "include": (collector_id,)}, "rerun_from"),
            ({"rerun_from": "run-" + "a" * 32}, "explicit included"),
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
        arguments = parser().parse_args(["run", "--performance-check"])
        self.assertTrue(arguments.performance_check)
        request = request(arguments, default_workers=4)
        self.assertTrue(request.performance_check)
        self.assertEqual(1, request.max_workers)

    def test_collector_command_runs_only_the_selected_id_and_declared_dependencies(self) -> None:
        """Direct execution never pulls unrelated profile members into its supervised run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core = root / "core" / "system"
            core.mkdir(parents=True)
            (root / "plugins").mkdir()
            (core / "first.py").write_text(
                _delayed_collector_source("first", 0),
                encoding="utf-8",
            )
            (core / "second.py").write_text(
                _delayed_collector_source("second", 0),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("logicytics.cli._project_root", return_value=root), patch(
                    "sys.stdout", output
            ):
                self.assertEqual(
                    0,
                    main([
                        "collector",
                        "core.system.first",
                        "--acknowledge-authorization",
                    ]),
                )
            manifests = list((root / "output" / "data").glob("run-*/manifest.json"))
            self.assertEqual(1, len(manifests))
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(
                ["core.system.first"],
                [record["id"] for record in manifest["collectors"]],
            )
            self.assertIn("Status: succeeded", output.getvalue())

            dependent = core / "dependent.py"
            dependent.write_text(
                _delayed_collector_source(
                    "dependent",
                    0,
                    dependencies=("core.system.second",),
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            plan = build_plan(
                report,
                RunRequest(
                    include=("core.system.dependent",),
                    selection_only=True,
                    acknowledge_authorization=True,
                    max_workers=1,
                ),
            )
            self.assertEqual(
                ["core.system.second", "core.system.dependent"],
                [candidate.metadata.id for candidate in plan.collectors],
            )

    def test_typed_mode_registry_maps_every_user_mode_and_legacy_alias(self) -> None:
        """One immutable matrix owns profile, scheduling, MODS, and performance behavior."""
        parser = parser()
        expected = {
            "standard": ("standard", 1, False, False, False),
            "balanced": ("standard", 4, False, False, False),
            "quick": ("minimal", 4, False, False, False),
            "thorough": ("deep", 4, False, False, False),
            "offline": ("offline", 4, False, False, False),
            "extensions": ("standard", 4, True, False, False),
            "non-python": ("standard", 4, True, True, False),
            "performance": ("standard", 1, False, False, True),
        }
        self.assertEqual(set(expected), set(EXECUTION_MODES))
        self.assertEqual(set(expected), {item["name"] for item in mode_matrix()["modes"]})
        for name, contract in expected.items():
            with self.subTest(mode=name):
                request = request(parser.parse_args(["run", "--mode", name]), 4)
                self.assertEqual(
                    contract,
                    (
                        request.profile,
                        request.max_workers,
                        request.enable_mods,
                        request.non_python_only,
                        request.performance_check,
                    ),
                )

        alias_flags = {
            "default_mode": "--default",
            "threaded": "--threaded",
            "minimal": "--minimal",
            "depth": "--depth",
            "modded": "--modded",
            "nopy": "--nopy",
            "performance_check": "--performance-check",
        }
        for field, mode_name in LEGACY_MODE_ALIASES.items():
            with self.subTest(alias=alias_flags[field]):
                legacy = request(parser.parse_args(["run", alias_flags[field]]), 4)
                named = request(parser.parse_args(["run", "--mode", mode_name]), 4)
                self.assertEqual(named, legacy)

        with self.assertRaisesRegex(ValueError, "--profile"):
            request(parser.parse_args(["run", "--mode", "quick", "--profile", "deep"]), 4)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--mode", "quick", "--minimal"])

    def test_modes_action_prints_the_complete_machine_readable_matrix(self) -> None:
        """Users and release checks can inspect the same authoritative mode registry."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()
            with patch("logicytics.cli._project_root", return_value=root), patch(
                    "sys.stdout", output
            ):
                self.assertEqual(0, main(["--modes"]))
            payload = json.loads(output.getvalue())
            self.assertEqual(1, payload["schema_version"])
            self.assertEqual(list(EXECUTION_MODES), [item["name"] for item in payload["modes"]])
            self.assertTrue(all("legacy_aliases" in item for item in payload["modes"]))
            self.assertEqual([], payload["collectors"])

    def test_cli_without_action_prints_help_and_modes_are_parser_exclusive(self) -> None:
        """An empty invocation is useful while contradictory legacy actions fail immediately."""
        output = io.StringIO()
        with patch("sys.stdout", output):
            self.assertEqual(0, main([]))
        rendered = output.getvalue()
        self.assertIn("Logicytics v4 run-oriented evidence framework", rendered)
        self.assertIn("preflight", rendered)
        self.assertIn("run", rendered)
        with patch("sys.stderr", new_callable=io.StringIO) as errors:
            with self.assertRaises(SystemExit):
                parser().parse_args(["run", "--default", "--performance-check"])
        self.assertIn("not allowed with argument", errors.getvalue())

    def test_update_can_explicitly_launch_an_allowlisted_action_in_a_new_window(self) -> None:
        """The paired update options launch exactly one shell-free visible Windows action."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            git = subprocess.CompletedProcess(["git", "--version"], 0, "git version 2.0\n", "")
            output = io.StringIO()
            with patch("logicytics.cli._project_root", return_value=root), patch(
                    "logicytics.cli.process_adapter.run", return_value=git
            ), patch("logicytics.cli.launch_action_window", return_value=321) as launch, patch(
                "sys.stdout", output
            ):
                self.assertEqual(
                    0,
                    main(["update", "--launch-action", "debug", "--new-window"]),
                )
            payload = json.loads(output.getvalue())
            self.assertEqual("debug", payload["launched_action"])
            self.assertEqual(321, payload["launched_process_id"])
            launch.assert_called_once_with(root, "debug")

            with patch("logicytics.cli._project_root", return_value=root), patch(
                    "sys.stdout", new_callable=io.StringIO
            ) as invalid_output:
                self.assertEqual(2, main(["update", "--new-window"]))
            self.assertIn("must be provided together", invalid_output.getvalue())

    def test_new_window_launcher_uses_current_interpreter_without_a_shell(self) -> None:
        """Visible maintenance windows preserve argument boundaries and repository cwd."""
        root = Path("C:/repo").resolve()
        process = MagicMock(pid=42)
        with patch("logicytics.cli.sys.platform", "win32"), patch(
                "logicytics.cli.process_adapter.popen", return_value=process
        ) as popen:
            self.assertEqual(42, launch_action_window(root, "preflight"))
        popen.assert_called_once_with(
            [sys.executable, "-m", "logicytics", "preflight"],
            cwd=root,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010),
            close_fds=True,
        )
        with patch("logicytics.cli.sys.platform", "linux"), self.assertRaisesRegex(
                OSError, "only on Windows"
        ):
            launch_action_window(root, "debug")

    def test_run_parser_exposes_explicit_sequential_and_bounded_parallel_modes(self) -> None:
        """Execution policy is selectable directly instead of relying on compatibility modes."""
        parser = parser()
        sequential = request(parser.parse_args(["run", "--sequential"]), default_workers=4)
        parallel = request(parser.parse_args(["run", "--parallel"]), default_workers=4)
        bounded_parallel = request(
            parser.parse_args(["run", "--parallel", "--workers", "3"]),
            default_workers=4,
        )

        self.assertEqual(1, sequential.max_workers)
        self.assertEqual(4, parallel.max_workers)
        self.assertEqual(3, bounded_parallel.max_workers)
        self.assertEqual("standard", sequential.profile)
        self.assertEqual("standard", parallel.profile)

    def test_run_parser_rejects_conflicting_explicit_execution_modes(self) -> None:
        """Contradictory worker policies fail before creating a collection plan."""
        parser = parser()
        conflicts = (
            (["run", "--sequential", "--workers", "2"], 4, "sequential execution"),
            (["run", "--sequential", "--threaded"], 4, "legacy --threaded"),
            (["run", "--parallel", "--performance-check"], 4, "performance/default"),
            (["run", "--parallel", "--default"], 4, "performance/default"),
            (["run", "--parallel", "--workers", "1"], 4, "at least two"),
            (["run", "--parallel"], 1, "at least two"),
        )
        for arguments, default_workers, error in conflicts:
            with self.subTest(arguments=arguments, default_workers=default_workers):
                with self.assertRaisesRegex(ValueError, error):
                    request(parser.parse_args(arguments), default_workers=default_workers)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--sequential", "--parallel"])

    def test_rerun_request_rejects_unfinalized_unknown_and_unselected_original_work(self) -> None:
        """Reruns require a finalized authentic-shaped manifest and explicit original collector IDs."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            collector_id = "core.system.system_info"
            valid_manifest = {
                "manifest_schema_version": 1,
                "run_id": "run-" + "a" * 32,
                "status": "succeeded",
                "resolved_plan": [collector_id],
            }
            manifest_path.write_text(json.dumps(valid_manifest), encoding="utf-8")
            parser = parser()

            with self.assertRaisesRegex(ValueError, "explicit --include"):
                request(parser.parse_args(["run", "--rerun-from", str(manifest_path)]), 2)
            with self.assertRaisesRegex(ValueError, "not present in the original"):
                request(
                    parser.parse_args(
                        ["run", "--rerun-from", str(manifest_path), "--include", "core.system.other"]
                    ),
                    2,
                )
            manifest_path.write_text(json.dumps({**valid_manifest, "status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finalized"):
                request(
                    parser.parse_args(["run", "--rerun-from", str(manifest_path), "--include", collector_id]),
                    2,
                )
            for schema_version in (None, True, 2):
                with self.subTest(manifest_schema_version=schema_version):
                    invalid_manifest = {**valid_manifest, "manifest_schema_version": schema_version}
                    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
                        request(
                            parser.parse_args(
                                ["run", "--rerun-from", str(manifest_path), "--include", collector_id]
                            ),
                            2,
                        )
            manifest_path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be loaded"):
                request(
                    parser.parse_args(["run", "--rerun-from", str(manifest_path), "--include", collector_id]),
                    2,
                )

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
            persisted_states: list[str] = []

            def capture_manifest_state(path, manifest):
                persisted_states.append(manifest.status.value)
                write_manifest(path, manifest)

            with patch("logicytics.runtime.write_manifest", side_effect=capture_manifest_state):
                outcome = RunSupervisor(root, configuration).run(plan)
            self.assertEqual("planned", persisted_states[0])
            self.assertEqual("running", persisted_states[1])
            self.assertEqual("succeeded", persisted_states[-1])
            self.assertEqual("succeeded", outcome.manifest.status.value)
            self.assertEqual(("core.system.system_info",), outcome.manifest.resolved_plan)
            self.assertEqual(plan.fingerprint, outcome.manifest.plan_fingerprint)
            self.assertEqual(64, len(plan.fingerprint))
            self.assertFalse(outcome.manifest.cancellation_requested)
            self.assertEqual([], outcome.manifest.errors)
            self.assertEqual([], outcome.manifest.skipped_collectors)
            self.assertEqual(configuration.runtime.output_root, outcome.run_directory.parent)
            self.assertEqual(root / "output" / "data", outcome.run_directory.parent)
            self.assertTrue(outcome.run_directory.is_absolute())
            self.assertTrue((outcome.run_directory / "artifacts" / "core_system_system_info").is_dir())
            self.assertFalse((root / "system.txt").exists())
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
            self.assertEqual(outcome.run_directory / "packages", Path(outcome.manifest.package["path"]).parent)
            self.assertEqual(outcome.run_directory / "hashes", Path(outcome.manifest.package["sha256_path"]).parent)
            self.assertTrue((outcome.run_directory / "logs" / "engine.jsonl").is_file())
            self.assertFalse((root / "ACCESS").exists())
            self.assertFalse((root / "output" / "RUNS").exists())
            self.assertFalse((root / "output" / "PACKAGES").exists())
            package_path, hash_path = package_run(outcome)
            self.assertTrue(package_path.is_file())
            self.assertTrue(hash_path.is_file())
            requested_at = datetime.fromisoformat(outcome.manifest.requested_at)
            timestamp = requested_at.strftime("%Y%m%dT%H%M%S.%fZ")
            self.assertEqual(f"run-{timestamp}-{outcome.manifest.run_id}.zip", package_path.name)
            self.assertEqual(f"{package_path.name}.sha256", hash_path.name)
            package_digest, sidecar_name = hash_path.read_text(encoding="ascii").split()
            self.assertEqual(package_path.name, sidecar_name)
            self.assertEqual(hashlib.sha256(package_path.read_bytes()).hexdigest(), package_digest)
            self.assertEqual(package_digest, outcome.manifest.package["sha256"])
            with zipfile.ZipFile(package_path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("metadata/manifest.json", archive.namelist())
                self.assertIn("reports/summary.txt", archive.namelist())
                self.assertIn("hashes/artifacts.sha256", archive.namelist())
                self.assertNotIn("logs/performance.json", archive.namelist())
                self.assertEqual(1, len([name for name in archive.namelist() if name.startswith("evidence/")]))
                artifact = outcome.manifest.artifact_list()[0]
                archived_bytes = archive.read(
                    f"evidence/{artifact.evidence_kind.value}/{artifact.relative_path}"
                )
                self.assertEqual(artifact.size_bytes, len(archived_bytes))
                self.assertEqual(artifact.sha256, hashlib.sha256(archived_bytes).hexdigest())
                self.assertEqual("system", artifact.source_category)
                datetime.fromisoformat(artifact.collected_at)
                self.assertEqual(("copied into run artifact store",), artifact.transformations)
                packaged_manifest = json.loads(archive.read("metadata/manifest.json"))
                self.assertEqual(["core.system.system_info"], packaged_manifest["resolved_plan"])
                self.assertEqual(plan.fingerprint, packaged_manifest["plan_fingerprint"])
                self.assertFalse(packaged_manifest["cancellation_requested"])
                self.assertEqual([], packaged_manifest["errors"])
                self.assertEqual([], packaged_manifest["skipped_collectors"])
                packaged_artifact = packaged_manifest["collectors"][0]["artifacts"][0]
                catalog_artifact = packaged_manifest["artifact_catalog"][0]
                self.assertEqual(artifact.source_category, packaged_artifact["source_category"])
                self.assertEqual(artifact.collected_at, packaged_artifact["collected_at"])
                self.assertEqual(list(artifact.transformations), packaged_artifact["transformations"])
                self.assertEqual(artifact.id, catalog_artifact["id"])
                self.assertEqual("system.txt", catalog_artifact["name"])
                self.assertEqual("text/plain", catalog_artifact["media_type"])
                self.assertEqual("registered", catalog_artifact["status"])
                self.assertEqual("succeeded", catalog_artifact["producer_status"])
                self.assertEqual("core.system.system_info", catalog_artifact["collector_id"])
                record = outcome.manifest.collectors[0]
                summary = archive.read("reports/summary.txt").decode("utf-8")
                self.assertIn(f"Status: {record.status}", summary)
                self.assertIn("Cancellation requested: false", summary)
                self.assertIn("Resolved collectors: 1", summary)
                self.assertIn(f"Plan fingerprint: {plan.fingerprint}", summary)
                self.assertIn(f"Started: {record.started_at}", summary)
                self.assertIn(f"Finished: {record.finished_at}", summary)

            repeated = RunSupervisor(root, configuration).run(plan)
            self.assertNotEqual(outcome.manifest.run_id, repeated.manifest.run_id)
            self.assertNotEqual(outcome.run_directory, repeated.run_directory)

    def test_package_separates_typed_evidence_reports_logs_hashes_and_metadata(self) -> None:
        """The versioned package layout is manifest-led and has no ambiguous root entries."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            source = _COLLECTOR.replace(
                "CollectorResult, CoreCollector, Specialty",
                "CollectorResult, CoreCollector, EvidenceKind, Specialty",
            ).replace(
                'output_media_types=("text/plain",)',
                'output_media_types=("text/plain", "application/octet-stream")',
            ).replace(
                '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                '        raw = context.workspace / "source.bin"\n'
                '        raw.write_bytes(b"raw evidence")\n'
                '        raw_artifact = context.artifacts.register_file(\n'
                '            raw, evidence_kind=EvidenceKind.RAW,\n'
                '        )\n'
                '        return CollectorResult.succeeded(\n'
                '            "typed artifacts created", (artifact, raw_artifact),\n'
                '        )',
            )
            collector_path.write_text(source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            outcome = RunSupervisor(root, default_config(root)).run(
                build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            )
            self.assertEqual("succeeded", outcome.manifest.status.value, outcome.manifest.package)

            package_path = Path(outcome.manifest.package["path"])
            with zipfile.ZipFile(package_path) as archive:
                names = archive.namelist()
                self.assertEqual(
                    {"evidence", "hashes", "logs", "metadata", "reports"},
                    {name.split("/", 1)[0] for name in names},
                )
                self.assertNotIn("manifest.json", names)
                self.assertNotIn("summary.txt", names)
                self.assertFalse(any(name.startswith("artifacts/") for name in names))
                artifacts = outcome.manifest.artifact_list()
                expected_catalog = "".join(
                    f"{artifact.sha256}  evidence/{artifact.evidence_kind.value}/{artifact.relative_path}\n"
                    for artifact in sorted(
                        artifacts,
                        key=lambda item: f"evidence/{item.evidence_kind.value}/{item.relative_path}",
                    )
                )
                self.assertEqual(
                    expected_catalog,
                    archive.read("hashes/artifacts.sha256").decode("ascii"),
                )
                for artifact in artifacts:
                    archive_name = f"evidence/{artifact.evidence_kind.value}/{artifact.relative_path}"
                    self.assertEqual(artifact.sha256, hashlib.sha256(archive.read(archive_name)).hexdigest())
                packaged = json.loads(archive.read("metadata/manifest.json"))
            self.assertEqual("1.0", packaged["package_layout_version"])
            self.assertEqual(1, packaged["manifest_schema_version"])
            self.assertEqual("evidence/raw/", packaged["package_sections"]["raw_evidence"])
            self.assertEqual("evidence/derived/", packaged["package_sections"]["derived_reports"])
            self.assertEqual(
                {"raw", "derived"},
                {item["evidence_kind"] for item in packaged["artifact_catalog"]},
            )
            outcome.manifest.manifest_schema_version = 2
            with self.assertRaisesRegex(ValueError, "unsupported run manifest schema_version"):
                outcome.manifest.to_dict()

    def test_configured_output_root_colocates_packages_without_touching_legacy_evidence(self) -> None:
        """Custom roots own runs and ZIPs; old ACCESS evidence is neither moved nor deleted."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            legacy_evidence = root / "ACCESS" / "RUNS" / "legacy-evidence.txt"
            legacy_evidence.parent.mkdir(parents=True)
            legacy_evidence.write_text("preserve existing evidence", encoding="utf-8")
            (root / "logicytics.json").write_text(
                '{"schema_version":4,"runtime":{"output_root":"custom/evidence"}}',
                encoding="utf-8",
            )
            configuration = load_config(root)
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, configuration).run(plan)
            expected_root = root / "custom" / "evidence"
            package_path = Path(outcome.manifest.package["path"])
            hash_path = Path(outcome.manifest.package["sha256_path"])

            self.assertEqual(expected_root, outcome.run_directory.parent)
            self.assertEqual(outcome.run_directory / "packages", package_path.parent)
            self.assertEqual(outcome.run_directory / "hashes", hash_path.parent)
            self.assertEqual("preserve existing evidence", legacy_evidence.read_text(encoding="utf-8"))
            self.assertFalse((root / "custom" / "PACKAGES").exists())
            self.assertEqual(
                hashlib.sha256(package_path.read_bytes()).hexdigest(),
                hash_path.read_text(encoding="ascii").split()[0],
            )

    def test_partial_collector_evidence_is_packaged_and_clearly_labeled(self) -> None:
        """Partial terminal outcomes preserve evidence without masquerading as success."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    '        return CollectorResult.partial(\n'
                    '            "only one evidence source was available",\n'
                    '            (artifact,),\n'
                    '            errors=("secondary evidence source was unavailable",),\n'
                    '        )',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            outcome = RunSupervisor(root, default_config(root)).run(
                build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            )
            record = outcome.manifest.collectors[0]

            self.assertEqual("partial", outcome.manifest.status.value)
            self.assertEqual("partial", record.status)
            self.assertEqual(1, len(record.artifacts))
            self.assertIn("secondary evidence source", outcome.manifest.errors[0]["message"])
            package_path = Path(outcome.manifest.package["path"])
            self.assertTrue(package_path.is_file())
            with zipfile.ZipFile(package_path) as archive:
                packaged = json.loads(archive.read("metadata/manifest.json"))
                summary = archive.read("reports/summary.txt").decode("utf-8")
                self.assertEqual(
                    ["ok"],
                    archive.read("evidence/derived/core_system_system_info/system.txt").decode("utf-8").splitlines(),
                )
            self.assertEqual("partial", packaged["status"])
            self.assertEqual("partial", packaged["collectors"][0]["status"])
            self.assertIn("Status: partial", summary)
            self.assertIn("secondary evidence source was unavailable", summary)

    def test_selected_collector_rerun_preserves_original_run_and_separates_evidence(self) -> None:
        """Explicit reruns execute only selected original IDs and own a distinct evidence package."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            for filename in ("a_original", "z_selected"):
                (core_directory / f"{filename}.py").write_text(
                    _delayed_collector_source(filename, 0.0),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            configuration = default_config(root)
            original = RunSupervisor(root, configuration).run(
                build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            )
            original_manifest = original.manifest_path.read_bytes()
            original_package_path = Path(original.manifest.package["path"])
            original_package = original_package_path.read_bytes()
            selected_id = "core.system.z_selected"
            arguments = parser().parse_args(
                [
                    "run",
                    "--rerun-from",
                    str(original.run_directory),
                    "--include",
                    selected_id,
                    "--acknowledge-authorization",
                    "--sequential",
                ]
            )
            request = request(arguments, default_workers=4)
            self.assertEqual(original.manifest.run_id, request.rerun_from)
            rerun = RunSupervisor(root, configuration).run(build_plan(report, request))

            self.assertEqual("succeeded", rerun.manifest.status.value)
            self.assertEqual("rerun", rerun.manifest.action)
            self.assertEqual(original.manifest.run_id, rerun.manifest.parent_run_id)
            self.assertEqual((selected_id,), rerun.manifest.resolved_plan)
            self.assertEqual([selected_id], [record.id for record in rerun.manifest.collectors])
            self.assertNotEqual(original.run_directory, rerun.run_directory)
            self.assertNotEqual(original_package_path, Path(rerun.manifest.package["path"]))
            self.assertTrue(original_package_path.name.startswith("run-"))
            self.assertTrue(Path(rerun.manifest.package["path"]).name.startswith("rerun-"))
            self.assertIn(rerun.manifest.run_id, Path(rerun.manifest.package["path"]).name)
            self.assertEqual(original_manifest, original.manifest_path.read_bytes())
            self.assertEqual(original_package, original_package_path.read_bytes())
            with zipfile.ZipFile(Path(rerun.manifest.package["path"])) as archive:
                packaged_manifest = json.loads(archive.read("metadata/manifest.json"))
                summary = archive.read("reports/summary.txt").decode("utf-8")
                artifact_names = [name for name in archive.namelist() if name.startswith("evidence/")]
            self.assertEqual(original.manifest.run_id, packaged_manifest["parent_run_id"])
            self.assertEqual("rerun", packaged_manifest["action"])
            self.assertEqual(["evidence/derived/core_system_z_selected/system.txt"], artifact_names)
            self.assertIn("Action: rerun", summary)
            self.assertIn(f"Parent run: {original.manifest.run_id}", summary)

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
            configuration = replace(
                default_config(root),
                collector_settings={
                    "core.system.system_info": {
                        "password": "evidence-password",
                        "nested": {"access_token": "evidence-token"},
                    }
                },
            )
            outcome = RunSupervisor(root, configuration).run(plan)
            self.assertEqual("succeeded", outcome.manifest.status.value)
            self.assertEqual("password=[REDACTED]", outcome.manifest.collectors[0].summary)
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                artifact = outcome.manifest.artifact_list()[0]
                self.assertEqual(
                    "password=evidence-password token=evidence-token",
                    archive.read(f"evidence/{artifact.evidence_kind.value}/{artifact.relative_path}").decode("utf-8"),
                )
                diagnostics = "\n".join(
                    archive.read(name).decode("utf-8")
                    for name in archive.namelist()
                    if not name.startswith("evidence/")
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
            self.assertEqual("core.system.system_info", outcome.manifest.errors[0]["collector_id"])
            self.assertNotIn("crash-password", outcome.manifest.errors[0]["message"])
            self.assertNotIn("crash-token", outcome.manifest.errors[0]["message"])
            failure = outcome.manifest.collectors[0].failure
            self.assertIsNotNone(failure)
            self.assertNotIn("crash-password", failure["platform_error"])
            self.assertNotIn("crash-token", failure["platform_error"])
            self.assertIn("[REDACTED]", failure["platform_error"])
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                packaged_record = json.loads(archive.read("metadata/manifest.json"))["collectors"][0]
                self.assertEqual(failure, packaged_record["failure"])
                diagnostics = "\n".join(
                    archive.read(name).decode("utf-8")
                    for name in archive.namelist()
                    if not name.startswith("evidence/")
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
                _COLLECTOR.replace("from pathlib import Path\n",
                                   "from pathlib import Path\nfrom time import sleep\n").replace(
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
                packaged_record = json.loads(archive.read("metadata/manifest.json"))["collectors"][0]
                report = json.loads(archive.read("logs/performance.json"))["collectors"][0]
                summary = archive.read("reports/summary.txt").decode("utf-8")
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
                self.assertNotIn("evidence/derived/unregistered.txt", archive.namelist())

            original_write = packaging._stream_archive_member

            def tampering_write(
                    archive: zipfile.ZipFile,
                    filename: Path,
                    arcname: str,
            ) -> None:
                if arcname.startswith("evidence/"):
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
                self.assertIn("logs/collectors/core_system_system_info/events.jsonl", archive.namelist())
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

    def test_package_name_rejects_unsafe_action_run_id_and_naive_timestamps(self) -> None:
        """Manipulated manifest identity cannot escape the output root or obscure run timing."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(_COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            invalid = (
                ("action", "../outside", "action"),
                ("run_id", "../outside", "run_id"),
                ("requested_at", "2026-01-01T00:00:00", "timezone"),
                ("requested_at", "not a timestamp", "timestamp"),
            )
            for field, value, message in invalid:
                with self.subTest(field=field, value=value):
                    original = getattr(outcome.manifest, field)
                    setattr(outcome.manifest, field, value)
                    try:
                        with self.assertRaisesRegex(ValueError, message):
                            package_run(outcome)
                    finally:
                        setattr(outcome.manifest, field, original)
            self.assertFalse((root / "outside").exists())

    def test_package_preserves_nested_registered_artifact_directories(self) -> None:
        """Deep collector-owned evidence paths survive catalog publication and ZIP creation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '        output = context.workspace / "system.txt"',
                    '        output = context.workspace / "nested" / "reports" / "system.txt"\n'
                    '        output.parent.mkdir(parents=True)',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            expected = "core_system_system_info/nested/reports/system.txt"
            self.assertEqual(expected, outcome.manifest.artifact_list()[0].relative_path)
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                self.assertIn(f"evidence/derived/{expected}", archive.namelist())
                self.assertEqual(
                    expected,
                    json.loads(archive.read("metadata/manifest.json"))["artifact_catalog"][0]["relative_path"],
                )

    def test_package_rejects_forged_names_status_and_duplicate_catalog_identifiers(self) -> None:
        """A package is refused whenever its finalized evidence catalog is inconsistent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace(
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    '        second = context.workspace / "second.txt"\n'
                    '        second.write_text("second", encoding="utf-8")\n'
                    '        extra = context.artifacts.register_file(second, media_type="text/plain")\n'
                    '        return CollectorResult.succeeded("test artifacts created", (artifact, extra))',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            first, second = outcome.manifest.collectors[0].artifacts
            original_name = first["name"]
            first["name"] = "forged.txt"
            with self.assertRaisesRegex(ValueError, "name does not match"):
                package_run(outcome)
            first["name"] = original_name
            first["status"] = "deleted"
            with self.assertRaisesRegex(ValueError, "status must be registered"):
                package_run(outcome)
            first["status"] = "registered"
            second["id"] = first["id"]
            with self.assertRaisesRegex(ValueError, "duplicate artifact id"):
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

    def test_explicit_execution_modes_control_isolated_worker_overlap(self) -> None:
        """First-class CLI policies determine real sequential versus bounded worker overlap."""
        for execution_mode, expect_overlap in (("--sequential", False), ("--parallel", True)):
            with self.subTest(execution_mode=execution_mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                core_directory = root / "core" / "system"
                core_directory.mkdir(parents=True)
                (root / "plugins").mkdir()
                for filename in ("a_first", "z_second"):
                    (core_directory / f"{filename}.py").write_text(
                        _delayed_collector_source(filename, 0.3),
                        encoding="utf-8",
                    )
                arguments = parser().parse_args(
                    ["run", execution_mode, "--acknowledge-authorization"]
                )
                request = request(arguments, default_workers=2)
                report = preflight(root)
                self.assertEqual((), report.invalid)
                outcome = RunSupervisor(root, default_config(root)).run(build_plan(report, request))
                records = {record.id: record for record in outcome.manifest.collectors}
                first = records["core.system.a_first"]
                second = records["core.system.z_second"]

                self.assertEqual("succeeded", first.status, first.errors)
                self.assertEqual("succeeded", second.status, second.errors)
                if expect_overlap:
                    self.assertLess(second.started_at, first.finished_at)
                else:
                    self.assertLessEqual(first.finished_at, second.started_at)

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

    def test_dependency_closure_and_plan_fingerprint_are_deterministic(self) -> None:
        """Diamond dependencies resolve once and hash identically regardless of discovery order."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            dependencies = {
                "a_root": (),
                "b_left": ("core.system.a_root",),
                "c_right": ("core.system.a_root",),
                "z_target": ("core.system.c_right", "core.system.b_left"),
            }
            for filename, required in dependencies.items():
                (core_directory / f"{filename}.py").write_text(
                    _delayed_collector_source(filename, 0.0, dependencies=required),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            request = RunRequest(
                include=("core.system.z_target",),
                rerun_from="run-" + "a" * 32,
                acknowledge_authorization=True,
            )
            first = build_plan(report, request)
            reversed_plan = build_plan(PreflightReport(tuple(reversed(report.candidates))), request)
            expected = (
                "core.system.a_root",
                "core.system.b_left",
                "core.system.c_right",
                "core.system.z_target",
            )

            self.assertEqual(expected, tuple(candidate.metadata.id for candidate in first.collectors))
            self.assertEqual(expected, tuple(candidate.metadata.id for candidate in reversed_plan.collectors))
            self.assertEqual(first.fingerprint, reversed_plan.fingerprint)
            self.assertEqual(64, len(first.fingerprint))

    def test_planner_rejects_excluded_missing_and_sensitive_dependency_conflicts(self) -> None:
        """Dependency expansion cannot override explicit exclusions or sensitive opt-in boundaries."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            dependency_id = "core.system.a_dependency"
            target_id = "core.system.z_target"
            (core_directory / "a_dependency.py").write_text(
                _delayed_collector_source("a_dependency", 0.0),
                encoding="utf-8",
            )
            (core_directory / "z_target.py").write_text(
                _delayed_collector_source("z_target", 0.0, dependencies=(dependency_id,)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PlanError, "explicitly excluded"):
                build_plan(preflight(root), RunRequest(include=(target_id,), exclude=(dependency_id,)))
            sensitive_source = _delayed_collector_source("a_dependency", 0.0).replace(
                "from logicytics import CollectorMetadata,",
                "from logicytics import Capability, CollectorMetadata,",
            ).replace(
                '            supported_platforms=("win32",),',
                '            supported_platforms=("win32",),\n'
                '            capabilities=(Capability.SENSITIVE_FILES,),\n'
                '            sensitive_data_categories=("credentials",),\n'
                '            default_profiles=("deep",),',
            )
            (core_directory / "a_dependency.py").write_text(sensitive_source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual((), report.invalid)
            with self.assertRaisesRegex(PlanError, "sensitive dependency"):
                build_plan(report,
                           RunRequest(include=(target_id,), approved_capabilities=(Capability.SENSITIVE_FILES,)))
            approved = build_plan(
                report,
                RunRequest(
                    include=(target_id, dependency_id),
                    approved_capabilities=(Capability.SENSITIVE_FILES,),
                ),
            )
            self.assertEqual([dependency_id, target_id], [candidate.metadata.id for candidate in approved.collectors])
            (core_directory / "a_dependency.py").unlink()
            with self.assertRaisesRegex(PlanError, "unavailable collector"):
                build_plan(preflight(root), RunRequest(include=(target_id,)))
            (core_directory / "a_dependency.py").write_text(
                _delayed_collector_source("a_dependency", 0.0, dependencies=(target_id,)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PlanError, "dependency cycle"):
                build_plan(preflight(root), RunRequest(include=(target_id,)))

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
            self.assertEqual(["core.system.b_dependent"], outcome.manifest.skipped_collectors)
            self.assertEqual(
                {dependency_id, "core.system.b_dependent"},
                {error["collector_id"] for error in outcome.manifest.errors},
            )
            with zipfile.ZipFile(Path(outcome.manifest.package["path"])) as archive:
                packaged = json.loads(archive.read("metadata/manifest.json"))
                summary = archive.read("reports/summary.txt").decode("utf-8")
            self.assertEqual(["core.system.b_dependent"], packaged["skipped_collectors"])
            self.assertIn("Skipped collectors: 1", summary)

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
                summary = archive.read("reports/summary.txt").decode("utf-8")
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
                packaged_record = json.loads(archive.read("metadata/manifest.json"))["collectors"][0]
                summary = archive.read("reports/summary.txt").decode("utf-8")
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
                self.assertIn("evidence/derived/core_system_system_info/system.txt", archive.namelist())
                packaged = json.loads(archive.read("metadata/manifest.json"))["collectors"][0]
                summary = archive.read("reports/summary.txt").decode("utf-8")
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
                packaged = {
                    item["id"]: item
                    for item in json.loads(archive.read("metadata/manifest.json"))["collectors"]
                }
                summary = archive.read("reports/summary.txt").decode("utf-8")
                self.assertIn("evidence/derived/core_system_a_failed/system.txt", archive.namelist())
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
                self.assertIn("evidence/derived/core_system_system_info/system.txt", archive.namelist())

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
                    '            supported_platforms=("win32",),\n'
                    '            capabilities=(Capability.NETWORK,),\n'
                    '            network_access=NetworkAccess.LOCAL,',
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

    def test_worker_blocks_application_update_power_and_peer_collector_commands_without_stopping_peers(self) -> None:
        """Subprocess approval cannot escape the collector role or affect an independent worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            peer_path = core_directory / "z_independent.py"
            peer_path.write_text(_delayed_collector_source("z_independent", 0.0), encoding="utf-8")
            protected_configuration = root / "logicytics.json"
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
                    attacker = _delayed_collector_source("a_attacker", 0.0).replace(
                        "from pathlib import Path\n",
                        "from pathlib import Path\nimport subprocess\nfrom logicytics import Capability\n",
                    ).replace(
                        '            supported_platforms=("win32",),',
                        '            supported_platforms=("win32",),\n'
                        '            capabilities=(Capability.SUBPROCESS,),',
                    ).replace(
                        '        output = context.workspace / "system.txt"',
                        f"        subprocess.run({command!r}, check=True)\n"
                        '        output = context.workspace / "system.txt"',
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

    def test_preflight_rejects_collector_imports_of_application_and_orchestration_services(self) -> None:
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
                "import logicytics\nlogicytics.configuration",
                "from logicytics.api import query_run",
            )
            for statement in attempts:
                with self.subTest(statement=statement):
                    collector_path.write_text(
                        _COLLECTOR.replace("from pathlib import Path\n", f"from pathlib import Path\n{statement}\n"),
                        encoding="utf-8",
                    )
                    report = preflight(root)
                    self.assertEqual(1, len(report.invalid))
                    self.assertIn("forbidden application import", "\n".join(report.invalid[0].static_errors))
                    diagnostic = next(
                        item for item in report.invalid[0].diagnostics
                        if "forbidden application import" in item.message
                    )
                    self.assertEqual("static.engine_boundary", diagnostic.rule)

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
            self.assertTrue(outcome.manifest.cancellation_requested)
            record = outcome.manifest.collectors[0]
            self.assertEqual("cancelled", record.status)
            self.assertEqual("run cancelled by user", record.summary)
            self.assertTrue((outcome.run_directory / ".cancelled").is_file())
            self.assertIsNotNone(outcome.manifest.package)
            package_path = Path(outcome.manifest.package["path"])
            self.assertTrue(package_path.is_file())
            with zipfile.ZipFile(package_path) as archive:
                summary = archive.read("reports/summary.txt").decode("utf-8")
                packaged_manifest = json.loads(archive.read("metadata/manifest.json"))
            self.assertTrue(packaged_manifest["cancellation_requested"])
            self.assertIn("Status: cancelled", summary)
            self.assertIn("Cancellation requested: true", summary)
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
            payload = report.to_dict()
            self.assertEqual(["core.system.system_info"], [item["id"] for item in payload["valid"]])
            self.assertEqual([], payload["invalid"])
            self.assertEqual(["plugin.broken_plugin"], [item["id"] for item in payload["quarantined"]])
            diagnostic = payload["quarantined"][0]["diagnostics"][0]
            self.assertEqual(str(plugin_path), diagnostic["path"])
            self.assertEqual(1, diagnostic["line"])
            self.assertEqual("static.class_name", diagnostic["rule"])
            self.assertIn("exactly one public collector class", diagnostic["message"])
            selected = report.to_dict(selected_plugins=("plugin.broken_plugin",))
            self.assertEqual([], selected["quarantined"])
            self.assertEqual(["plugin.broken_plugin"], [item["id"] for item in selected["invalid"]])
            enabled = report.to_dict(enable_plugins=True)
            self.assertEqual(["plugin.broken_plugin"], [item["id"] for item in enabled["invalid"]])
            plan = build_plan(report, RunRequest())
            self.assertEqual(
                ["core.system.system_info"],
                [candidate.metadata.id for candidate in plan.collectors],
            )

    def test_invalid_selected_folder_or_enabled_plugin_blocks_planning(self) -> None:
        """Folder-owned invalid plugins fail closed by logical ID and explicit enablement."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_path = root / "core" / "system" / "system_info.py"
            core_path.parent.mkdir(parents=True)
            core_path.write_text(_COLLECTOR, encoding="utf-8")
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
            core_path.write_text(_COLLECTOR, encoding="utf-8")
            plugin_path = root / "plugins" / "broken_plugin.py"
            plugin_path.parent.mkdir()
            plugin_path.write_text('"""Invalid plugin collector."""\n', encoding="utf-8")

            for arguments, expected_exit, key in (
                    (["preflight"], 0, "quarantined"),
                    (["preflight", "--include", "plugin.broken_plugin"], 2, "invalid"),
                    (["preflight", "--plugins"], 2, "invalid"),
            ):
                with self.subTest(arguments=arguments):
                    output = io.StringIO()
                    with patch("logicytics.cli._project_root", return_value=root), patch("sys.stdout", output):
                        exit_code = main(arguments)
                    payload = json.loads(output.getvalue())
                    self.assertEqual(expected_exit, exit_code)
                    self.assertEqual("plugin.broken_plugin", payload[key][0]["id"])
                    self.assertEqual(str(plugin_path), payload[key][0]["diagnostics"][0]["path"])
                    self.assertIn("rule", payload[key][0]["diagnostics"][0])

    def test_preflight_rejects_wrong_lifecycle_return_type(self) -> None:
        """Lifecycle annotations must match the strict collector contract exactly."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                _COLLECTOR.replace("def collect(self, context: CollectorContext) -> CollectorResult:",
                                   "def collect(self, context: CollectorContext) -> ValidationResult:"),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("collect must return CollectorResult", report.invalid[0].static_errors)
            diagnostic = next(
                item for item in report.invalid[0].diagnostics
                if item.message == "collect must return CollectorResult"
            )
            source_lines = collector_path.read_text(encoding="utf-8").splitlines()
            expected_line = next(
                index for index, line in enumerate(source_lines, start=1)
                if "def collect(self, context: CollectorContext) -> ValidationResult:" in line
            )
            self.assertEqual(expected_line, diagnostic.line)
            self.assertEqual("static.return_annotation", diagnostic.rule)
            self.assertEqual(str(collector_path), diagnostic.path)

    def test_worker_runs_typed_prepare_collect_finalize_and_cleanup_lifecycle(self) -> None:
        """Every accepted collector executes the complete typed lifecycle inside its worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            source = _COLLECTOR.replace(
                "    def collect(self, context: CollectorContext) -> CollectorResult:",
                "    def prepare(self, context: CollectorContext) -> ValidationResult:\n"
                "        \"\"\"Prepare worker-local lifecycle state.\"\"\"\n"
                "        context.settings['prepared'] = 'yes'\n"
                "        return ValidationResult(True)\n\n"
                "    def collect(self, context: CollectorContext) -> CollectorResult:",
            ).replace(
                "    def cleanup(self, context: CollectorContext) -> None:",
                "    def finalize(self, context: CollectorContext, result: CollectorResult) -> CollectorResult:\n"
                "        \"\"\"Finalize the typed result before it crosses the worker boundary.\"\"\"\n"
                "        if context.settings.get('prepared') != 'yes':\n"
                "            raise RuntimeError('prepare phase did not run')\n"
                "        return CollectorResult.succeeded('finalized lifecycle', result.artifacts)\n\n"
                "    def cleanup(self, context: CollectorContext) -> None:",
            )
            collector_path.write_text(source, encoding="utf-8")

            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            plan = build_plan(
                report,
                RunRequest(max_workers=1, acknowledge_authorization=True),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]
            self.assertEqual("finalized lifecycle", record.summary)

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
                    "logicytics.discovery.process_adapter.run",
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
                _COLLECTOR.replace("return CollectorResult.succeeded(\"test artifact created\", (artifact,))",
                                   "print('unexpected output')\n        return CollectorResult.succeeded(\"test artifact created\", (artifact,))"),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn("collectors must not print; use structured progress or logging",
                          report.invalid[0].static_errors)

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
                'supported_platforms=("win32",), capabilities=(Capability.ELEVATED_PRIVILEGES,), '
                'privilege_level=PrivilegeLevel.ELEVATED,',
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
            source = _plugin_collector_source()
            source = source.replace("SystemInfoCollector", "AdminPluginCollector")
            source = source.replace("core.system.system_info", "plugin.admin_plugin")
            source = source.replace(
                "from logicytics import CollectorMetadata",
                "from logicytics import Capability, CollectorMetadata",
            ).replace(
                "            capabilities=(),",
                "            capabilities=(Capability.ELEVATED_PRIVILEGES,),",
            ).replace(
                "            privilege_level=PrivilegeLevel.STANDARD,",
                "            privilege_level=PrivilegeLevel.ELEVATED,",
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
