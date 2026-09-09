"""Discover and execute the complete Logicytics test suite."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

from logicytics.terminal import terminal_lifecycle
from logicytics.virtual_environment import (
    is_running_in_virtual_environment,
    render_virtual_environment_error,
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
    logger.event("INFO", "test_suite_started", source="logicytics.cli.tests")
    transcript = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
    result = unittest.TextTestRunner(stream=transcript, verbosity=arguments.verbosity).run(suite)
    summary = (
        f"Tests: {result.testsRun}",
        f"Failures: {len(result.failures)}",
        f"Errors: {len(result.errors)}",
        f"Skipped: {len(result.skipped)}",
        f"Result: {'passed' if result.wasSuccessful() else 'failed'}",
    )
    logger.box("Test suite", summary)
    if not result.wasSuccessful():
        details = tuple(line for line in transcript.getvalue().splitlines() if line.strip())
        logger.box("Test failures", details or ("The test runner did not provide failure details.",))
        logger.event("ERROR", "test_suite_failed", source="logicytics.cli.tests")
    else:
        logger.event("INFO", "test_suite_finished", source="logicytics.cli.tests")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        with terminal_lifecycle():
            raise SystemExit(main())
    except KeyboardInterrupt:
        from logicytics.module.logging import ApplicationLogger

        print("\n")
        ApplicationLogger.render_section(sys.stderr, "Command cancelled", ("Interrupted by user.",))
        raise SystemExit(130)
