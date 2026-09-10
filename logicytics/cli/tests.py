"""Discover and execute the complete Logicytics test suite."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

if TYPE_CHECKING:
    from unittest.runner import _ResultClassType

from logicytics.terminal import isolated_terminal_lifecycle, terminal_lifecycle
from logicytics.virtual_environment import (
    is_running_in_virtual_environment,
    render_virtual_environment_error,
)

ProgressCallback: TypeAlias = Callable[[int, int, str], None]


def _progress_result_class(total: int, progress: ProgressCallback) -> type[unittest.TextTestResult]:
    """Create a result class that advances one dependency-free test progress bar."""

    class ProgressResult(unittest.TextTestResult):
        """Report each completed test without changing unittest result semantics."""

        completed = 0
        current = ""

        def startTest(self, test: unittest.case.TestCase) -> None:
            self.current = test.id()
            progress(self.completed, total, self.current)
            super().startTest(test)

        def stopTest(self, test: unittest.case.TestCase) -> None:
            super().stopTest(test)
            self.completed += 1
            progress(self.completed, total, self.current)

    return ProgressResult


def _failure_details(result: unittest.TestResult) -> tuple[tuple[str, str, str], ...]:
    """Extract category, test ID, and final actionable line for every failed test."""
    details: list[tuple[str, str, str]] = []
    for category, failures in (("Failure", result.failures), ("Error", result.errors)):
        for test, traceback_text in failures:
            test_name = test.id() if hasattr(test, "id") else str(test)
            tail = next((line.strip() for line in reversed(traceback_text.splitlines()) if line.strip()),
                        "No detail provided")
            details.append((category, test_name, tail))
    return tuple(details)


def _failure_rows(result: unittest.TestResult) -> tuple[str, ...]:
    """Summarize every failed test without echoing a full unittest traceback wall."""
    return tuple(
        item
        for category, test_name, reason in _failure_details(result)
        for item in (f"{category}: {test_name}", f"Reason: {reason}")
    )


def project_root() -> Path:
    """Return the repository root containing the dynamically discovered test package."""
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Dynamically discover tests and return a CI-appropriate result code."""
    root = project_root()
    if not is_running_in_virtual_environment():
        render_virtual_environment_error(sys.stderr, root)
        return 2
    from logicytics.module.configuration import load_config
    from logicytics.module.errors import PlanError
    from logicytics.module.logging import (
        ApplicationLogger,
        HumanArgumentParser,
        get_application_logger,
    )
    from logicytics.module.output_layout import ensure_output_layout

    parser = HumanArgumentParser(description="Run all discovered Logicytics tests.")
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=2)
    arguments = parser.parse_args(argv)
    try:
        configuration = load_config(root)
        layout = ensure_output_layout(configuration.runtime.output_root)
    except (OSError, PlanError) as error:
        ApplicationLogger.render_section(
            sys.stderr,
            "Test runner error",
            (f"Unable to prepare test presentation: {error}",),
        )
        return 2
    logger = get_application_logger(layout.application_log, configuration.logging)
    debug = str(configuration.logging.level).upper() == "DEBUG"
    transcript = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
    total = suite.countTestCases()
    logger.event(
        "INFO",
        "test_suite_started",
        source="logicytics.cli.tests",
        tests=total,
        mode="debug" if debug else "normal",
    )
    with isolated_terminal_lifecycle():
        if debug:
            result = unittest.TextTestRunner(stream=sys.stderr, verbosity=arguments.verbosity).run(suite)
        else:
            def progress(checked: int, count: int, current: str) -> None:
                """Render the current normal-mode test position without a third-party dependency."""
                logger.progress("Tests", checked, count, current)

            result_class = _progress_result_class(total, progress)
            with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
                result = unittest.TextTestRunner(
                    stream=transcript,
                    verbosity=0,
                    # The stdlib decorates its stream before invoking this result factory.
                    resultclass=cast("_ResultClassType", result_class),
                ).run(suite)
    summary = (
        f"Tests: {result.testsRun}",
        f"Failures: {len(result.failures)}",
        f"Errors: {len(result.errors)}",
        f"Skipped: {len(result.skipped)}",
        f"Result: {'passed' if result.wasSuccessful() else 'failed'}",
    )
    if not result.wasSuccessful():
        logger.event(
            "ERROR",
            "test_suite_failed",
            source="logicytics.cli.tests",
            failures=len(result.failures),
            errors=len(result.errors),
        )
        for category, test_name, reason in _failure_details(result):
            logger.event(
                "ERROR",
                "test_case_failed",
                source="logicytics.cli.tests",
                category=category,
                test=test_name,
                reason=reason,
            )
        logger.box("Test failures", _failure_rows(result) or ("The test runner did not provide failure details.",))
    else:
        logger.event("INFO", "test_suite_finished", source="logicytics.cli.tests", tests=result.testsRun)
    logger.box("Test suite", summary)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        with terminal_lifecycle():
            raise SystemExit(main())
    except KeyboardInterrupt:
        from logicytics.module.logging import ApplicationLogger

        ApplicationLogger.render_section(sys.stderr, "Command cancelled", ("Interrupted by user.",))
        raise SystemExit(130)
