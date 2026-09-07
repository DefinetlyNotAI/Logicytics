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
from logicytics.module.modes import EXECUTION_MODES, LEGACY_MODE_ALIASES, mode_matrix
from logicytics.platform_adapters import process_adapter


class CliTests(unittest.TestCase):
    """CLI modes, parser constraints, reruns, and launch behavior."""

    def test_run_parser_accepts_performance_check(self) -> None:
        """The run command must expose the performance mode used by the request builder."""
        arguments = cli_methods.parser().parse_args(["run", "--performance-check"])
        self.assertTrue(arguments.performance_check)
        request = cli_methods.request(arguments, default_workers=4)
        self.assertTrue(request.performance_check)
        self.assertEqual(1, request.max_workers)

    def test_typed_mode_registry_maps_every_user_mode_and_legacy_alias(self) -> None:
        """One immutable matrix owns profile, scheduling, MODS, and performance behavior."""
        parser = cli_methods.parser()
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
                        run_request.enable_mods,
                        run_request.non_python_only,
                        run_request.performance_check,
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
                parser.parse_args([
                    "run",
                    "--mode",
                    "quick",
                    "--profile",
                    "deep",
                ]),
                4,
            )

        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--mode", "quick", "--minimal"])

    def test_modes_action_prints_the_complete_machine_readable_matrix(self) -> None:
        """Users and release checks can inspect the same authoritative mode registry."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()

            with patch.object(
                    CLI,
                    "project_root",
                    return_value=root,
            ), patch(
                "sys.stdout",
                output,
            ):
                self.assertEqual(0, main(["--modes"]))

            payload = json.loads(output.getvalue())

            self.assertEqual(1, payload["schema_version"])
            self.assertEqual(
                list(EXECUTION_MODES),
                [item["name"] for item in payload["modes"]],
            )
            self.assertTrue(
                all("legacy_aliases" in item for item in payload["modes"])
            )
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
                cli_methods.parser().parse_args(["run", "--default", "--performance-check"])
        self.assertIn("not allowed with argument", errors.getvalue())

    def test_update_can_explicitly_launch_an_allowlisted_action_in_a_new_window(
            self,
    ) -> None:
        """The paired update options launch exactly one shell-free visible Windows action."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()

            git = subprocess.CompletedProcess(
                ["git", "--version"],
                0,
                "git version 2.0\n",
                "",
            )
            output = io.StringIO()

            with patch.object(
                    CLI,
                    CLI.project_root.__name__,
                    return_value=root,
            ), patch.object(
                process_adapter,
                process_adapter.run.__name__,
                return_value=git,
            ), patch.object(
                CLI,
                CLI.launch_action_window.__name__,
                return_value=321,
            ) as launch, patch(
                "sys.stdout",
                output,
            ):
                self.assertEqual(
                    0,
                    main([
                        "update",
                        "--launch-action",
                        "debug",
                        "--new-window",
                    ]),
                )

            payload = json.loads(output.getvalue())
            self.assertEqual("debug", payload["launched_action"])
            self.assertEqual(321, payload["launched_process_id"])
            launch.assert_called_once_with(root, "debug")

            with patch.object(
                    CLI,
                    CLI.project_root.__name__,
                    return_value=root,
            ), patch(
                "sys.stdout",
                new_callable=io.StringIO,
            ) as invalid_output:
                self.assertEqual(
                    2,
                    main(["update", "--new-window"]),
                )

            self.assertIn(
                "must be provided together",
                invalid_output.getvalue(),
            )

    def test_new_window_launcher_uses_current_interpreter_without_a_shell(
            self,
    ) -> None:
        """Visible maintenance windows preserve argument boundaries and repository cwd."""
        root = Path("C:/repo").resolve()
        process = MagicMock(pid=42)

        with patch(
                "sys.platform",
                "win32",
        ), patch.object(
            process_adapter,
            process_adapter.popen.__name__,
            return_value=process,
        ) as popen:
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

        with patch(
                "sys.platform",
                "linux",
        ), self.assertRaisesRegex(
            OSError,
            "only on Windows",
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
            (["run", "--parallel", "--performance-check"], 4, "performance/default"),
            (["run", "--parallel", "--default"], 4, "performance/default"),
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
