"""Regression coverage for the public command-line interface."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from logicytics.cli import CLI, cli_methods, main
from logicytics.contracts import Capability
from logicytics.module.configuration import default_config
from logicytics.module.modes import EXECUTION_MODES, LEGACY_MODE_ALIASES, mode_matrix
from logicytics.platform_adapters import process_adapter


class CliTests(unittest.TestCase):
    """CLI modes, parser constraints, reruns, and launch behavior."""

    def test_cli_requires_activating_the_existing_local_environment(self) -> None:
        """A local environment gets an activation instruction instead of an installer prompt."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activation_script = root / ".venv" / "Scripts" / "Activate.ps1"
            activation_script.parent.mkdir(parents=True)
            activation_script.write_text("", encoding="utf-8")
            output = io.StringIO()

            with (
                patch("logicytics.cli._project_root", return_value=root),
                patch.object(
                    sys,
                    "prefix",
                    sys.base_prefix,
                ),
                patch("sys.stderr", output),
            ):
                self.assertEqual(2, main(["preflight"]))

            self.assertIn(r".\.venv\Scripts\Activate.ps1", output.getvalue())
            self.assertNotIn("installer first", output.getvalue())

    def test_cli_requires_the_installer_when_the_local_environment_is_missing(self) -> None:
        """A missing or incomplete local environment gets the installer instruction."""
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()

            with (
                patch("logicytics.cli._project_root", return_value=Path(temporary)),
                patch.object(
                    sys,
                    "prefix",
                    sys.base_prefix,
                ),
                patch("sys.stderr", output),
            ):
                self.assertEqual(2, main(["preflight"]))

            rendered = " ".join(output.getvalue().split())
            self.assertIn("python -m logicytics.cli.installer", rendered)

    def test_run_parser_accepts_performance_check(self) -> None:
        """The run flag serially measures the explicitly selected collection mode."""
        arguments = cli_methods.parser().parse_args(["run", "--mode", "thorough", "--performance-check"])
        self.assertTrue(arguments.performance_check)
        request = cli_methods.request(arguments, default_workers=4)
        self.assertTrue(request.performance_check)
        self.assertEqual("deep", request.profile)
        self.assertEqual(1, request.max_workers)

    def test_run_parser_rejects_removed_script_extension_flag(self) -> None:
        """Only typed plugin collectors remain an opt-in extension surface."""
        with self.assertRaises(SystemExit):
            cli_methods.parser().parse_args(["run", "--mods"])

    def test_config_flag_is_accepted_before_or_after_a_subcommand(self) -> None:
        """Configuration selection must not depend on flag placement around the action."""
        parser = cli_methods.parser()
        expected = Path("custom.yaml")

        self.assertEqual(expected, parser.parse_args(["--config", "custom.yaml", "preflight"]).config)
        self.assertEqual(expected, parser.parse_args(["preflight", "--config", "custom.yaml"]).config)

    def test_preflight_and_plan_expose_cache_invalidation(self) -> None:
        """Validation-heavy commands can explicitly discard reusable probe metadata."""
        parser = cli_methods.parser()

        self.assertTrue(parser.parse_args(["preflight", "--invalidate-cache"]).invalidate_cache)
        self.assertTrue(parser.parse_args(["plan", "--invalidate-cache"]).invalidate_cache)

    def test_ctrl_c_during_command_is_rendered_without_a_traceback(self) -> None:
        """A user interrupt returns 130 with a concise cancellation section."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()
            with (
                patch.object(CLI, CLI.project_root.__name__, return_value=root),
                patch("logicytics.cli.commands.load_config", return_value=default_config(root)),
                patch("logicytics.cli.commands.preflight", side_effect=KeyboardInterrupt),
                patch("sys.stderr", output),
            ):
                self.assertEqual(130, main(["preflight"]))

            rendered = output.getvalue()
            self.assertIn("Command cancelled", rendered)
            self.assertIn("Interrupted by user.", rendered)
            self.assertNotIn("Traceback", rendered)

    def test_capabilities_default_to_metadata_and_cli_can_block_them(self) -> None:
        """Declared capabilities run by default and can be disabled per invocation."""
        parser = cli_methods.parser()

        request = cli_methods.request(
            parser.parse_args(["run", "--profile", "standard"]),
            default_workers=4,
        )
        self.assertEqual((), request.blocked_capabilities)

        blocked = cli_methods.request(
            parser.parse_args(
                [
                    "run",
                    "--profile",
                    "standard",
                    "--block-capability",
                    "network",
                    "--block-capability",
                    "filesystem_read",
                ]
            ),
            default_workers=4,
        )
        self.assertEqual(
            (
                Capability.NETWORK,
                Capability.FILESYSTEM_READ,
            ),
            blocked.blocked_capabilities,
        )

    def test_configuration_capabilities_are_combined_with_cli_blocks(self) -> None:
        """Configuration and invocation policy blocks are merged without duplicates."""
        parser = cli_methods.parser()
        request = cli_methods.request(
            parser.parse_args(["run", "--block-capability", "subprocess"]),
            default_workers=4,
            configured_blocked_capabilities=(Capability.NETWORK, Capability.SUBPROCESS),
        )
        self.assertEqual(
            (Capability.NETWORK, Capability.SUBPROCESS),
            request.blocked_capabilities,
        )

    def test_typed_mode_registry_maps_every_user_mode_and_legacy_alias(self) -> None:
        """One immutable matrix owns profile and scheduling behavior."""
        parser = cli_methods.parser()
        expected = {
            "standard": ("standard", 1),
            "balanced": ("standard", 4),
            "quick": ("minimal", 4),
            "thorough": ("deep", 4),
            "offline": ("offline", 4),
        }

        self.assertEqual(set(expected), set(EXECUTION_MODES))

        modes = mode_matrix()["modes"]
        assert isinstance(modes, list)
        assert all(isinstance(item, dict) for item in modes)

        self.assertEqual(
            set(expected),
            {item["name"] for item in modes},
        )

        for name, contract in expected.items():
            with self.subTest(mode=name):
                run_request = cli_methods.request(
                    parser.parse_args(["run", "--mode", name]),
                    4,
                )
                self.assertEqual(
                    contract,
                    (
                        run_request.profile,
                        run_request.max_workers,
                    ),
                )

        alias_flags = {
            "default_mode": "--default",
            "threaded": "--threaded",
            "minimal": "--minimal",
            "depth": "--depth",
        }

        for field, mode_name in LEGACY_MODE_ALIASES.items():
            with self.subTest(alias=alias_flags[field]):
                legacy = cli_methods.request(
                    parser.parse_args(["run", alias_flags[field]]),
                    4,
                )
                named = cli_methods.request(
                    parser.parse_args(["run", "--mode", mode_name]),
                    4,
                )
                self.assertEqual(named, legacy)

        with self.assertRaisesRegex(ValueError, "--profile"):
            cli_methods.request(
                parser.parse_args(
                    [
                        "run",
                        "--mode",
                        "quick",
                        "--profile",
                        "thorough",
                    ]
                ),
                4,
            )

        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--mode", "quick", "--minimal"])
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--mode", "performance"])

    def test_modes_action_renders_a_summary_and_writes_the_machine_readable_matrix(self) -> None:
        """Users see a readable summary while release checks use the saved mode registry."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                self.assertEqual(0, main(["--modes"]))

            rendered = output.getvalue()
            self.assertIn("Execution modes", rendered)
            self.assertIn("Machine-readable matrix:", rendered)
            self.assertNotIn("{", rendered)
            payload = json.loads((root / "output" / "logs" / "debug" / "modes.json").read_text(encoding="utf-8"))

            self.assertEqual(1, payload["schema_version"])
            self.assertEqual(
                list(EXECUTION_MODES),
                [item["name"] for item in payload["modes"]],
            )
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
        request = cli_methods.request(
            cli_methods.parser().parse_args(["run", "--default", "--performance-check"]),
            default_workers=4,
        )
        self.assertTrue(request.performance_check)
        self.assertEqual(1, request.max_workers)

    def test_argument_error_keeps_the_complete_command_reference(self) -> None:
        """Invalid arguments retain every available flag and capability in a readable section."""
        output = io.StringIO()

        with patch("sys.stderr", output), self.assertRaises(SystemExit):
            cli_methods.parser().parse_args(["collector"])

        rendered = output.getvalue()
        self.assertIn("Command-line error", rendered)
        self.assertIn("Available command options", rendered)
        self.assertIn("collector_id\n\nAvailable command options", rendered)
        self.assertIn("--block-capability", rendered)
        self.assertIn("private_keys", rendered)
        self.assertIn("--interactive", rendered)
        self.assertIn("Confirm you are authorized", rendered)
        self.assertIn("collector_id", rendered)

    def test_update_can_explicitly_launch_an_allowlisted_action_in_a_new_window(
            self,
    ) -> None:
        """The paired update options launch exactly one shell-free visible Windows action."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()

            def reachable_git(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if command == ["git", "--version"]:
                    return subprocess.CompletedProcess(command, 0, "git version 2.0\n", "")
                if command == ["git", "rev-parse", "--is-inside-work-tree"]:
                    return subprocess.CompletedProcess(command, 0, "true\n", "")
                if command == ["git", "remote", "get-url", "origin"]:
                    return subprocess.CompletedProcess(command, 0, "https://github.com/example/logicytics.git\n", "")
                if command == ["git", "ls-remote", "--exit-code", "origin", "HEAD"]:
                    return subprocess.CompletedProcess(command, 0, "head\tHEAD\n", "")
                self.fail(f"unexpected Git command: {command}")

            output = io.StringIO()

            with (
                patch.object(
                    CLI,
                    CLI.project_root.__name__,
                    return_value=root,
                ),
                patch.object(
                    process_adapter,
                    process_adapter.run.__name__,
                    side_effect=reachable_git,
                ),
                patch.object(
                    CLI,
                    CLI.launch_action_window.__name__,
                    return_value=321,
                ) as launch,
                patch(
                    "sys.stderr",
                    output,
                ),
            ):
                self.assertEqual(
                    0,
                    main(
                        [
                            "update",
                            "--launch-action",
                            "debug",
                            "--new-window",
                        ]
                    ),
                )

            rendered = output.getvalue()
            self.assertIn("Update summary", rendered)
            self.assertIn("Update started", rendered)
            self.assertIn("Launched action: debug", rendered)
            self.assertIn("Launched process id: 321", rendered)
            self.assertNotIn("{", rendered)
            payload = json.loads((root / "output" / "logs" / "debug" / "update.json").read_text(encoding="utf-8"))
            self.assertEqual("debug", payload["launched_action"])
            self.assertEqual(321, payload["launched_process_id"])
            self.assertTrue(payload["remote_reachable"])
            launch.assert_called_once_with(root, "debug")

            error_logger = MagicMock()
            with (
                patch.object(
                    CLI,
                    CLI.project_root.__name__,
                    return_value=root,
                ),
                patch(
                    "logicytics.cli.commands.get_application_logger",
                    return_value=error_logger,
                ),
            ):
                self.assertEqual(
                    2,
                    main(["update", "--new-window"]),
                )

            error_logger.box.assert_called_once_with(
                "Command error",
                ("--new-window and --launch-action must be provided together",),
            )
            event_messages = [call.args[1] for call in error_logger.event.call_args_list if len(call.args) > 1]
            self.assertIn("command_finished", event_messages)

    def test_update_refuses_an_unreachable_git_remote(self) -> None:
        """A local Git directory is insufficient when its configured remote cannot be reached."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()

            def unreachable_git(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if command == ["git", "--version"]:
                    return subprocess.CompletedProcess(command, 0, "git version 2.0\n", "")
                if command == ["git", "rev-parse", "--is-inside-work-tree"]:
                    return subprocess.CompletedProcess(command, 0, "true\n", "")
                if command == ["git", "remote", "get-url", "origin"]:
                    return subprocess.CompletedProcess(command, 0, "https://github.com/example/logicytics.git\n", "")
                if command == ["git", "ls-remote", "--exit-code", "origin", "HEAD"]:
                    return subprocess.CompletedProcess(command, 128, "", "network unavailable")
                self.fail(f"unexpected Git command: {command}")

            with (
                patch.object(CLI, CLI.project_root.__name__, return_value=root),
                patch.object(process_adapter, process_adapter.run.__name__, side_effect=unreachable_git),
                patch("sys.stderr", output),
            ):
                self.assertEqual(2, main(["update", "--apply"]))

            payload = json.loads((root / "output" / "logs" / "debug" / "update.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["remote_reachable"])
            self.assertFalse(payload["applied"])
            self.assertIn("Result: update was not applied.", output.getvalue())

    def test_update_reports_a_missing_git_executable(self) -> None:
        """Maintenance status remains readable when Git is not installed."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()

            with (
                patch.object(CLI, CLI.project_root.__name__, return_value=root),
                patch.object(
                    process_adapter,
                    process_adapter.run.__name__,
                    side_effect=FileNotFoundError("git executable is unavailable"),
                ),
                patch("sys.stderr", output),
            ):
                self.assertEqual(2, main(["update", "--apply"]))

            payload = json.loads((root / "output" / "logs" / "debug" / "update.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["git_available"])
            self.assertFalse(payload["remote_reachable"])
            self.assertIn("Install Git, then run the update command again.", output.getvalue())

    def test_new_window_launcher_uses_current_interpreter_without_a_shell(
            self,
    ) -> None:
        """Visible maintenance windows preserve argument boundaries and repository cwd."""
        root = Path("C:/repo").resolve()
        process = MagicMock(pid=42)

        with (
            patch(
                "sys.platform",
                "win32",
            ),
            patch.object(
                process_adapter,
                process_adapter.popen.__name__,
                return_value=process,
            ) as popen,
        ):
            self.assertEqual(
                42,
                cli_methods.launch_action_window(root, "preflight"),
            )

        popen.assert_called_once_with(
            [sys.executable, "-m", "logicytics", "preflight"],
            cwd=root,
            shell=False,
            creationflags=process_adapter.create_new_console,
            close_fds=True,
        )

        with (
            patch(
                "sys.platform",
                "linux",
            ),
            self.assertRaisesRegex(
                OSError,
                "only on Windows",
            ),
        ):
            cli_methods.launch_action_window(root, "debug")

    def test_run_parser_exposes_explicit_sequential_and_bounded_parallel_modes(self) -> None:
        """Execution policy is selectable directly instead of relying on compatibility modes."""
        parser = cli_methods.parser()
        sequential = cli_methods.request(parser.parse_args(["run", "--sequential"]), default_workers=4)
        parallel = cli_methods.request(parser.parse_args(["run", "--parallel"]), default_workers=4)
        bounded_parallel = cli_methods.request(
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
        parser = cli_methods.parser()
        conflicts = (
            (["run", "--sequential", "--workers", "2"], 4, "sequential execution"),
            (["run", "--sequential", "--threaded"], 4, "legacy --threaded"),
            (["run", "--parallel", "--performance-check"], 4, "performance checking requires sequential"),
            (["run", "--parallel", "--default"], 4, "sequential default mode"),
            (["run", "--parallel", "--workers", "1"], 4, "at least two"),
            (["run", "--parallel"], 1, "at least two"),
        )
        for arguments, default_workers, error in conflicts:
            with self.subTest(arguments=arguments, default_workers=default_workers):
                with self.assertRaisesRegex(ValueError, error):
                    cli_methods.request(parser.parse_args(arguments), default_workers=default_workers)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--sequential", "--parallel"])


if __name__ == "__main__":
    unittest.main()
