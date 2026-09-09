"""Regression coverage for collector planning and profile selection."""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logicytics.cli import CLI, cli_methods, main
from logicytics.contracts import (
    Capability,
    OutputPolicy,
    PostRunAction,
    RunRequest,
    RunStatus,
)
from logicytics.module.configuration import (
    default_config,
    load_config,
)
from logicytics.module.discovery import PreflightReport, preflight
from logicytics.module.environment import EnvironmentReport
from logicytics.module.errors import PlanError
from logicytics.module.logging import (
    FileEventLogger,
)
from logicytics.module.planner import BUILTIN_PROFILES, build_plan
from logicytics.module.runtime import RunSupervisor
from logicytics.platform_adapters import ProcessAdapter
from tests.fixtures.collectors import COLLECTOR, delayed_collector_source


class PlanningTests(unittest.TestCase):
    """Selection, profiles, authorization, dependencies, and planning behavior."""

    def _collector_ids(self, candidates) -> list[str]:
        """Return collector IDs while requiring discovered metadata to be present."""
        collector_ids: list[str] = []
        for candidate in candidates:
            metadata = candidate.metadata
            if metadata is None:
                self.fail("planned collector is missing metadata")
            collector_ids.append(metadata.id)
        return collector_ids

    def test_post_run_actions_are_typed_exclusive_and_require_verified_packaging(
        self,
    ) -> None:
        """Power actions remain explicit and cannot run before durable package publication."""
        arguments = cli_methods.parser().parse_args(
            [
                "run",
                "--shutdown",
                "--performance-check",
                "--acknowledge-authorization",
            ]
        )

        request = cli_methods.request(arguments, default_workers=4)

        self.assertEqual(PostRunAction.SHUTDOWN, request.post_run_action)
        self.assertEqual(OutputPolicy.PACKAGE, request.output_policy)
        self.assertEqual(1, request.max_workers)

        with self.assertRaises(SystemExit):
            cli_methods.parser().parse_args(
                [
                    "run",
                    "--reboot",
                    "--shutdown",
                ]
            )

        with self.assertRaisesRegex(ValueError, "require packaged output"):
            cli_methods.request(
                cli_methods.parser().parse_args(
                    [
                        "run",
                        "--reboot",
                        "--no-package",
                    ]
                ),
                default_workers=1,
            )

        with tempfile.TemporaryDirectory() as temporary:
            logger = FileEventLogger(
                Path(temporary) / "events.jsonl",
                run_id="run-" + "a" * 32,
            )

            manifest = type(
                "Manifest",
                (),
                {
                    "status": RunStatus.SUCCEEDED,
                    "package": {
                        "path": "evidence.zip",
                        "sha256": "a" * 64,
                    },
                },
            )()

            completed = subprocess.CompletedProcess(
                ["shutdown"],
                0,
                "",
                "",
            )

            with patch.object(
                ProcessAdapter,
                ProcessAdapter.run.__name__,
                return_value=completed,
            ) as command:
                RunSupervisor._execute_post_run_action(
                    PostRunAction.REBOOT,
                    manifest,
                    logger,
                )

                self.assertEqual("shutdown", command.call_args.args[0][0])
                self.assertEqual("/r", command.call_args.args[0][1])
                self.assertEqual("60", command.call_args.args[0][3])

    def test_sensitive_collectors_require_explicit_profile_or_include_opt_in(self) -> None:
        """Default collection excludes sensitive evidence unless explicitly selected."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            (core_directory / "a_standard.py").write_text(
                delayed_collector_source("a_standard", 0.0),
                encoding="utf-8",
            )
            sensitive_id = "core.system.z_sensitive"
            source = (
                delayed_collector_source("z_sensitive", 0.0)
                .replace(
                    "from logicytics import CollectorMetadata",
                    "from logicytics import Capability, CollectorMetadata",
                )
                .replace(
                    "            capabilities=(),",
                    "            capabilities=(Capability.SENSITIVE_FILES,),\n"
                    '            sensitive_data_categories=("credentials",),\n'
                    '            default_profiles=("deep",),',
                )
            )
            (core_directory / "z_sensitive.py").write_text(source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual((), report.invalid)

            standard = build_plan(report, RunRequest())
            self.assertEqual(["core.system.a_standard"], self._collector_ids(standard.collectors))
            with self.assertRaisesRegex(PlanError, "capability blocked by policy"):
                build_plan(
                    report,
                    RunRequest(
                        include=(sensitive_id,),
                        blocked_capabilities=(Capability.SENSITIVE_FILES,),
                    ),
                )
            opted_in = build_plan(
                report,
                RunRequest(
                    include=(sensitive_id,),
                    exclude=("core.system.a_standard",),
                    approved_capabilities=(Capability.SENSITIVE_FILES,),
                ),
            )
            self.assertEqual([sensitive_id], self._collector_ids(opted_in.collectors))
            deep = build_plan(
                report,
                RunRequest(profile="deep", approved_capabilities=(Capability.SENSITIVE_FILES,)),
            )
            self.assertEqual([sensitive_id], self._collector_ids(deep.collectors))

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
                source = delayed_collector_source(filename, 0.0).replace(
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
                    self.assertEqual(collector_ids, self._collector_ids(plan.collectors))
            with self.assertRaisesRegex(PlanError, "unknown collection profile"):
                build_plan(report, RunRequest(profile="invented"))
            with patch("sys.stderr", new_callable=io.StringIO) as errors:
                with self.assertRaises(SystemExit):
                    cli_methods.parser().parse_args(["plan", "--profile", "invented"])
            self.assertIn("invalid choice", errors.getvalue())

    def test_offline_profile_rejects_network_collectors_even_with_explicit_approval(self) -> None:
        """Offline collection remains local-only regardless of include and capability overrides."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_id = "core.system.network_source"
            source = (
                delayed_collector_source("network_source", 0.0)
                .replace(
                    "from logicytics import CollectorMetadata",
                    "from logicytics import Capability, CollectorMetadata",
                )
                .replace(
                    "            capabilities=(),",
                    "            capabilities=(Capability.NETWORK,),\n"
                    "            network_access=NetworkAccess.LOCAL,\n"
                    '            default_profiles=("standard",),',
                )
            )
            (core_directory / "network_source.py").write_text(source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual((), report.invalid)
            standard = build_plan(report, RunRequest(approved_capabilities=(Capability.NETWORK,)))
            self.assertEqual([collector_id], self._collector_ids(standard.collectors))
            self.assertEqual((), build_plan(report, RunRequest(profile="offline")).collectors)
            with self.assertRaisesRegex(PlanError, "offline profile prohibits network-capable"):
                build_plan(
                    report,
                    RunRequest(
                        profile="offline",
                        include=(collector_id,),
                        approved_capabilities=(Capability.NETWORK,),
                    ),
                )

    def test_policy_error_reports_every_selected_collector(self) -> None:
        """Planning reports all selected capability failures in one deterministic result."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()

            for filename, capability in (
                ("a_subprocess", "Capability.SUBPROCESS"),
                ("b_network", "Capability.NETWORK"),
                ("c_subprocess", "Capability.SUBPROCESS"),
            ):
                source = (
                    delayed_collector_source(filename, 0.0)
                    .replace(
                        "from logicytics import CollectorMetadata",
                        "from logicytics import Capability, CollectorMetadata",
                    )
                    .replace(
                        "            capabilities=(),",
                        f"            capabilities=({capability},),\n"
                        + ("            network_access=NetworkAccess.LOCAL,\n" if capability == "Capability.NETWORK" else ""),
                    )
                )
                (core_directory / f"{filename}.py").write_text(source, encoding="utf-8")

            with self.assertRaisesRegex(
                PlanError,
                "selected collector policy validation failed",
            ) as rejected:
                build_plan(
                    preflight(root),
                    RunRequest(
                        blocked_capabilities=(Capability.NETWORK, Capability.SUBPROCESS),
                    ),
                )

            message = str(rejected.exception)
            self.assertLess(
                message.index("core.system.a_subprocess"),
                message.index("core.system.b_network"),
            )
            self.assertLess(
                message.index("core.system.b_network"),
                message.index("core.system.c_subprocess"),
            )
            for collector_id in (
                "core.system.a_subprocess",
                "core.system.b_network",
                "core.system.c_subprocess",
            ):
                self.assertIn(collector_id, message)
            self.assertIn(
                "Blocked by: --block-capability network --block-capability subprocess",
                message,
            )

    def test_authorization_error_summarizes_categories_and_sensitive_outputs(self) -> None:
        """Collection consent must explain requested evidence before creating a workspace."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(
                COLLECTOR.replace(
                    "from logicytics import CollectorMetadata",
                    "from logicytics import Capability, CollectorMetadata",
                ).replace(
                    "            capabilities=(),",
                    "            capabilities=(Capability.SENSITIVE_FILES,),\n"
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
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            config_path = root / "logicytics.yaml"
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

    def test_collector_command_runs_only_the_selected_id_and_declared_dependencies(
        self,
    ) -> None:
        """Direct execution never pulls unrelated profile members into its supervised run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core = root / "core" / "system"
            core.mkdir(parents=True)
            (root / "plugins").mkdir()

            (core / "first.py").write_text(
                delayed_collector_source("first", 0),
                encoding="utf-8",
            )
            (core / "second.py").write_text(
                delayed_collector_source("second", 0),
                encoding="utf-8",
            )

            output = io.StringIO()

            with (
                patch.object(
                    CLI,
                    "project_root",
                    return_value=root,
                ),
                patch(
                    "sys.stderr",
                    output,
                ),
            ):
                self.assertEqual(
                    0,
                    main(
                        [
                            "collector",
                            "core.system.first",
                            "--acknowledge-authorization",
                        ]
                    ),
                )

            manifests = list((root / "output" / "data" / "run").glob("*/manifest.json"))
            self.assertEqual(1, len(manifests))

            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))

            self.assertEqual(
                ["core.system.first"],
                [record["id"] for record in manifest["collectors"]],
            )
            self.assertIn("Status: succeeded", output.getvalue())

            dependent = core / "dependent.py"
            dependent.write_text(
                delayed_collector_source(
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
                self._collector_ids(plan.collectors),
            )

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
            parser = cli_methods.parser()

            with self.assertRaisesRegex(ValueError, "explicit --include"):
                cli_methods.request(parser.parse_args(["run", "--rerun-from", str(manifest_path)]), 2)
            with self.assertRaisesRegex(ValueError, "not present in the original"):
                cli_methods.request(
                    parser.parse_args(
                        [
                            "run",
                            "--rerun-from",
                            str(manifest_path),
                            "--include",
                            "core.system.other",
                        ]
                    ),
                    2,
                )
            manifest_path.write_text(json.dumps({**valid_manifest, "status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finalized"):
                cli_methods.request(
                    parser.parse_args(["run", "--rerun-from", str(manifest_path), "--include", collector_id]),
                    2,
                )
            for schema_version in (None, True, 2):
                with self.subTest(manifest_schema_version=schema_version):
                    invalid_manifest = {**valid_manifest, "manifest_schema_version": schema_version}
                    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
                        cli_methods.request(
                            parser.parse_args(
                                [
                                    "run",
                                    "--rerun-from",
                                    str(manifest_path),
                                    "--include",
                                    collector_id,
                                ]
                            ),
                            2,
                        )
            manifest_path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be loaded"):
                cli_methods.request(
                    parser.parse_args(["run", "--rerun-from", str(manifest_path), "--include", collector_id]),
                    2,
                )

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
                    delayed_collector_source(filename, 0.0, dependencies=required),
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

            self.assertEqual(expected, tuple(self._collector_ids(first.collectors)))
            self.assertEqual(expected, tuple(self._collector_ids(reversed_plan.collectors)))
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
                delayed_collector_source("a_dependency", 0.0),
                encoding="utf-8",
            )
            (core_directory / "z_target.py").write_text(
                delayed_collector_source("z_target", 0.0, dependencies=(dependency_id,)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PlanError, "explicitly excluded"):
                build_plan(preflight(root), RunRequest(include=(target_id,), exclude=(dependency_id,)))
            sensitive_source = (
                delayed_collector_source("a_dependency", 0.0)
                .replace(
                    "from logicytics import CollectorMetadata,",
                    "from logicytics import Capability, CollectorMetadata,",
                )
                .replace(
                    "            capabilities=(),",
                    "            capabilities=(Capability.SENSITIVE_FILES,),\n"
                    '            sensitive_data_categories=("credentials",),\n'
                    '            default_profiles=("deep",),',
                )
            )
            (core_directory / "a_dependency.py").write_text(sensitive_source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual((), report.invalid)
            with self.assertRaisesRegex(PlanError, "sensitive dependency"):
                build_plan(
                    report,
                    RunRequest(include=(target_id,), approved_capabilities=(Capability.SENSITIVE_FILES,)),
                )
            approved = build_plan(
                report,
                RunRequest(
                    include=(target_id, dependency_id),
                    approved_capabilities=(Capability.SENSITIVE_FILES,),
                ),
            )
            self.assertEqual([dependency_id, target_id], self._collector_ids(approved.collectors))
            (core_directory / "a_dependency.py").unlink()
            with self.assertRaisesRegex(PlanError, "unavailable collector"):
                build_plan(preflight(root), RunRequest(include=(target_id,)))
            (core_directory / "a_dependency.py").write_text(
                delayed_collector_source("a_dependency", 0.0, dependencies=(target_id,)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PlanError, "dependency cycle"):
                build_plan(preflight(root), RunRequest(include=(target_id,)))

    def test_capability_gate_requires_explicit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            secured_collector = COLLECTOR.replace(
                "from logicytics import CollectorMetadata",
                "from logicytics import Capability, CollectorMetadata",
            ).replace(
                "capabilities=(),",
                "capabilities=(Capability.FILESYSTEM_READ,),",
            )
            collector_path.write_text(secured_collector, encoding="utf-8")
            report = preflight(root)
            plan = build_plan(
                report,
                RunRequest(),
            )
            self.assertEqual(1, len(plan.collectors))

    def test_elevated_collectors_require_approval_and_administrator_privileges(self) -> None:
        """Privilege-sensitive collection must fail closed before launching a worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            secured_collector = COLLECTOR.replace(
                "from logicytics import CollectorMetadata",
                "from logicytics import Capability, CollectorMetadata",
            ).replace(
                "capabilities=(),",
                "capabilities=(Capability.ELEVATED_PRIVILEGES,), privilege_level=PrivilegeLevel.ELEVATED,",
            )
            collector_path.write_text(secured_collector, encoding="utf-8")
            report = preflight(root)
            with self.assertRaisesRegex(PlanError, "capability blocked by policy"):
                build_plan(
                    report,
                    RunRequest(blocked_capabilities=(Capability.ELEVATED_PRIVILEGES,)),
                )
            approved = RunRequest()
            for administrator_state in (False, None):
                with self.subTest(administrator_state=administrator_state):
                    environment = EnvironmentReport(administrator_state, True, None)
                    with patch("logicytics.module.planner.inspect_environment", return_value=environment):
                        with self.assertRaisesRegex(PlanError, "administrator account"):
                            build_plan(report, approved)
            with patch(
                "logicytics.module.planner.inspect_environment",
                return_value=EnvironmentReport(True, True, None),
            ):
                plan = build_plan(report, approved)
            self.assertEqual(1, len(plan.collectors))


if __name__ == "__main__":
    unittest.main()
