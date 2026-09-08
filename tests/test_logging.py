from __future__ import annotations

import ast
import io
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from logicytics.module.command_runner import parse_level_messages, run_command
from logicytics.module.configuration import (
    LoggingSettings,
    load_config,
)
from logicytics.module.errors import PlanError
from logicytics.module.logging import (
    ApplicationLogger,
    FileEventLogger,
    deprecated,
    get_application_logger,
    get_event_logger,
    raise_logged,
    timed,
)
from logicytics.module.output_layout import ensure_output_layout
from logicytics.virtual_environment import render_virtual_environment_error, virtual_environment_error
from tests.fixtures.file_listing import list_files


class LoggingTests(unittest.TestCase):
    """Logging, output layout, file listing, and command-runner behavior."""

    def test_application_sources_do_not_call_print(self) -> None:
        """Production output must use structured logging instead of direct console prints."""
        root = Path(__file__).resolve().parent.parent
        violations: list[str] = []
        for source_root in (root / "logicytics", root / "core"):
            for source in source_root.rglob("*.py"):
                tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
                violations.extend(
                    f"{source.relative_to(root)}:{node.lineno}"
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                )

        self.assertEqual([], violations)

    def test_virtual_environment_error_uses_the_logging_alert_formatter(self) -> None:
        """Bootstrap errors retain the exact redacted, wrapped logger alert presentation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activation_script = root / ".venv" / "Scripts" / "Activate.ps1"
            activation_script.parent.mkdir(parents=True)
            activation_script.write_text("", encoding="utf-8")
            expected = io.StringIO()
            actual = io.StringIO()

            ApplicationLogger.render_alert(
                expected,
                "Logicytics startup error",
                virtual_environment_error(root),
            )
            render_virtual_environment_error(actual, root)

            self.assertEqual(expected.getvalue(), actual.getvalue())
            self.assertIn("╭", actual.getvalue())
            self.assertIn("│ ×", actual.getvalue())
            self.assertIn("│ →", actual.getvalue())

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
            logger.box("Presentation", ("non-log output stays out of the event file",))
            for index in range(40):
                logger.event("INFO", "bounded row " + str(index) + " " + "x" * 80)

            contents = path.read_text(encoding="utf-8")
            self.assertNotIn("previous secret", contents)
            self.assertNotIn("hidden", contents)
            self.assertLessEqual(path.stat().st_size, settings.maximum_bytes)
            self.assertFalse(old.exists())
            self.assertIn("\033[", console.getvalue())
            self.assertIn("Presentation", console.getvalue())
            self.assertNotIn("Presentation", contents)
            self.assertNotIn("raw", contents)
            self.assertTrue(all(" | " in line for line in contents.splitlines()))
            self.assertIn("\033[91m\033[1m  \u00d7 ", console.getvalue())
            self.assertIn("typed event", console.getvalue())
            self.assertNotIn("\u256d", console.getvalue())
            first_row = contents.splitlines()[0]
            self.assertRegex(first_row, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})? \|")
            self.assertLessEqual(max(map(len, contents.splitlines())), 140)
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
            config_path = root / "logicytics.yaml"
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

    def test_application_logging_wraps_console_rows_and_falls_back_to_ascii(self) -> None:
        """Compact console records stay aligned on terminals without Unicode support."""

        class AsciiTerminalBuffer(io.StringIO):
            encoding = "ascii"

            def isatty(self) -> bool:
                return False

        with tempfile.TemporaryDirectory() as temporary:
            console = AsciiTerminalBuffer()
            logger = ApplicationLogger(
                Path(temporary) / "Logicytics.log",
                LoggingSettings(file_enabled=False, color_enabled=False),
                console=console,
            )
            logger.event("INFO", "a long message " * 12)
            logger.box("Summary", ("finished",))

            rows = console.getvalue().splitlines()
            self.assertTrue(rows[0].startswith("  * "))
            self.assertTrue(rows[1].startswith("    "))
            self.assertEqual(
                [
                    "[ Summary ]",
                    "-" * ApplicationLogger._console_width(),
                    "  > finished",
                ],
                rows[-3:],
            )
            self.assertNotIn("\u25cf", console.getvalue())
            self.assertNotIn("\u256d", console.getvalue())

    def test_application_logging_humanizes_structured_values_and_lowercase_settings(self) -> None:
        """Human sinks normalize structured messages, collections, binary values, and levels."""
        with tempfile.TemporaryDirectory() as temporary:
            console = io.StringIO()
            logger = ApplicationLogger(
                Path(temporary) / "Logicytics.log",
                LoggingSettings(level="debug", color_enabled=False),
                console=console,
            )
            logger.event(
                "info",
                '{"status": "complete", "token": "secret-value"}',
                capabilities={"network", "subprocess"},
                payload=b"binary evidence",
            )

            rendered = console.getvalue()
            file_contents = (Path(temporary) / "Logicytics.log").read_text(encoding="utf-8")
            for output in (rendered, file_contents):
                self.assertIn("Structured details", output)
                self.assertIn("Status: complete", output)
                self.assertIn("Capabilities: network, subprocess", output)
                self.assertIn("Payload: <15 bytes>", output)
                self.assertNotIn("{\"status\"", output)
                self.assertNotIn("capabilities=", output)
                self.assertNotIn("secret-value", output)

    def test_application_logging_rejects_invalid_configured_level(self) -> None:
        """A direct logger API cannot silently fail because of an invalid level policy."""
        with tempfile.TemporaryDirectory() as temporary:
            logger = ApplicationLogger(
                Path(temporary) / "Logicytics.log",
                LoggingSettings(level="trace", console_enabled=False),
            )
            with self.assertRaisesRegex(ValueError, "unsupported log level"):
                logger.event("INFO", "event")

    def test_application_logging_wraps_long_plain_sections(self) -> None:
        """Human summaries keep long diagnostics readable within the console width."""
        with tempfile.TemporaryDirectory() as temporary:
            console = io.StringIO()
            logger = ApplicationLogger(
                Path(temporary) / "Logicytics.log",
                LoggingSettings(file_enabled=False, color_enabled=False),
                console=console,
            )
            logger.box("Summary", ("detail " * 40,))

            rows = console.getvalue().splitlines()
            self.assertEqual("[ Summary ]", rows[0])
            self.assertEqual("-" * ApplicationLogger._console_width(), rows[1])
            self.assertGreater(len(rows), 2)
            self.assertTrue(all(len(row) <= ApplicationLogger._console_width() for row in rows))

    def test_application_logging_groups_lifecycle_events_into_titled_steps(self) -> None:
        """Lifecycle output has titled ASCII steps and consistently marked details."""
        with tempfile.TemporaryDirectory() as temporary:
            console = io.StringIO()
            logger = ApplicationLogger(
                Path(temporary) / "Logicytics.log",
                LoggingSettings(file_enabled=False, color_enabled=False),
                console=console,
            )
            logger.event("INFO", "command_started", command="preflight")
            logger.event("INFO", "preflight_started", configuration_hash="a" * 64)
            logger.event("INFO", "run_packaging_started", run_id="run-test")
            logger.event("INFO", "run_packaged", run_id="run-test")

            rendered = console.getvalue()
            self.assertIn("[ Command ]\n", rendered)
            self.assertIn("[ Preflight ]\n", rendered)
            self.assertIn("  ● Command started\n    > Command: preflight", rendered)
            self.assertIn(
                "  ● Preflight started\n    > Configuration hash: aaaaaaa",
                rendered,
            )
            self.assertEqual(1, rendered.count("[ Packaging ]"))

    def test_application_logging_compacts_configuration_hash_only_outside_debug(self) -> None:
        """Normal console output stays concise without losing the logged hash."""
        configuration_hash = "0123456789abcdef" * 4
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Logicytics.log"
            normal_console = io.StringIO()
            normal_logger = ApplicationLogger(
                path,
                LoggingSettings(color_enabled=False),
                console=normal_console,
            )
            normal_logger.event(
                "INFO",
                "preflight_started",
                configuration_hash=configuration_hash,
            )

            self.assertIn("> Configuration hash: 0123456", normal_console.getvalue())
            self.assertNotIn(configuration_hash, normal_console.getvalue())
            self.assertIn(configuration_hash, path.read_text(encoding="utf-8"))

            debug_console = io.StringIO()
            debug_logger = ApplicationLogger(
                path,
                LoggingSettings(level="DEBUG", file_enabled=False, color_enabled=False),
                console=debug_console,
            )
            debug_logger.event(
                "INFO",
                "preflight_started",
                configuration_hash=configuration_hash,
            )

            self.assertIn(configuration_hash, debug_console.getvalue())

    def test_application_logging_colors_info_text_white_and_debug_text_gray(self) -> None:
        """Status markers retain their severity color while message text stays readable."""

        class TerminalBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as temporary:
            console = TerminalBuffer()
            logger = ApplicationLogger(
                Path(temporary) / "Logicytics.log",
                LoggingSettings(level="DEBUG", file_enabled=False, color_enabled=True),
                console=console,
            )
            logger.event("INFO", "info message")
            logger.event("DEBUG", "debug message")
            logger.event("INFO", "collector_finished", collector_id="core.system.system_info")

            output = console.getvalue()
            self.assertIn(
                "\033[96m\033[1m  \u25cf \033[0m\033[97m\033[1minfo message",
                output,
            )
            self.assertIn(
                "\033[90m\033[1m  \u00b7 \033[0m\033[90m\033[1mdebug message",
                output,
            )
            self.assertIn("Collector finished\n", output)
            self.assertIn("Collector id: core.system.system_info", output)
            self.assertNotIn("collector_id=", output)

    def test_deprecation_decorator_logs_removal_context(self) -> None:
        """Deprecated functions must preserve behavior while reporting removal context."""
        events: list[tuple[str, str, dict[str, Any]]] = []

        class Logger:
            @staticmethod
            def event(level: str, message: str, **fields: Any) -> None:
                events.append((level, message, fields))

        @deprecated(cast(Any, Logger()), removal_version="5.0", reason="replacement exists")
        def old() -> str:
            return "still works"

        self.assertEqual("still works", old())
        self.assertEqual("function_deprecated", events[0][1])
        self.assertEqual("5.0", events[0][2]["removal_version"])

    def test_exception_helper_logs_before_raising(self) -> None:
        """Exception helpers must preserve the requested exception type and context."""
        events: list[tuple[str, str, dict[str, Any]]] = []

        class Logger:
            @staticmethod
            def event(level: str, message: str, **fields: Any) -> None:
                events.append((level, message, fields))

        with self.assertRaises(ValueError):
            raise_logged(cast(Any, Logger()), ValueError, "invalid setting", setting="workers")
        self.assertEqual("exception", events[0][0])
        self.assertEqual("ValueError", events[0][2]["exception_type"])

    def test_timed_decorator_records_function_lifecycle(self) -> None:
        """Timing instrumentation must report start and finish through EventLogger."""
        events: list[tuple[str, str, dict[str, Any]]] = []

        class Logger:
            @staticmethod
            def event(level: str, message: str, **fields: Any) -> None:
                events.append((level, message, fields))

        @timed(cast(Any, Logger()))
        def add(left: int, right: int) -> int:
            return left + right

        self.assertEqual(3, add(1, 2))
        self.assertEqual(["function_started", "function_finished"], [event[1] for event in events])

    def test_structured_event_logger_redacts_secret_fields_and_inline_credentials(self) -> None:
        """Structured diagnostics must preserve useful fields without exposing secrets."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            logger = FileEventLogger(
                path,
                run_id="test-run",
                collector_id="core.system.private_keys",
            )
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
            with self.assertRaisesRegex(ValueError, "unsupported log level"):
                logger.event("TRACE", "unsupported")

    def test_structured_event_logger_serializes_concurrent_jsonl_events(self) -> None:
        """Concurrent diagnostics stay complete, independently parseable, and redacted."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            logger = FileEventLogger(path, run_id="concurrent-run", collector_id="core.system.test")

            def write_event(index: int) -> None:
                logger.event(
                    "info",
                    "collector_progress",
                    sequence=index,
                    password=f"secret-{index}",
                )

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
            self.assertEqual(((root / "collectors" / "valid.py").resolve(),), files)

    def test_command_runner_captures_output_and_parses_structured_levels(self) -> None:
        """Core command execution must avoid a shell and preserve structured output."""
        result = run_command(("python", "-c", "print('INFO: collected'); print('ordinary')"))
        self.assertEqual(0, result.returncode)
        self.assertEqual((("INFO", "collected"),), parse_level_messages(result.stdout))


if __name__ == "__main__":
    unittest.main()
