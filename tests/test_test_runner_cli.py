from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from logicytics.cli import tests as test_runner


class TestRunnerCliTests(unittest.TestCase):
    """Presentation and result handling for the dynamically discovered test command."""

    def test_runner_reports_discovered_suite_through_the_application_console(self) -> None:
        """A successful dynamic run renders a boxed summary and returns a CI-success status."""
        result = MagicMock()
        result.failures = ()
        result.errors = ()
        result.skipped = (object(),)
        result.testsRun = 7
        result.wasSuccessful.return_value = True
        logger = MagicMock()
        runner = MagicMock()
        runner.run.return_value = result

        with patch.object(test_runner, "_require_virtual_environment"), \
                patch.object(test_runner, "load_config", return_value=MagicMock()), \
                patch.object(test_runner, "ensure_output_layout", return_value=MagicMock()), \
                patch.object(test_runner, "get_application_logger", return_value=logger), \
                patch.object(test_runner.unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()), \
                patch.object(test_runner.unittest, "TextTestRunner", return_value=runner):
            self.assertEqual(0, test_runner.main(["--verbosity", "0"]))

        logger.box.assert_called_once_with(
            "Test suite",
            ("Tests: 7", "Failures: 0", "Errors: 0", "Skipped: 1", "Result: passed"),
        )
        logger.event.assert_any_call("INFO", "test_suite_started", source="logicytics.cli.tests")
        logger.event.assert_any_call("INFO", "test_suite_finished", source="logicytics.cli.tests")

    def test_runner_boxes_failure_details_and_returns_nonzero(self) -> None:
        """Failed discovery remains CI-visible while preserving the runner's diagnostic transcript."""
        result = MagicMock()
        result.failures = ((object(), "assertion failed"),)
        result.errors = ()
        result.skipped = ()
        result.testsRun = 1
        result.wasSuccessful.return_value = False
        logger = MagicMock()

        class FailingRunner:
            """Emit a representative unittest transcript before returning a failed result."""

            def __init__(self, *, stream, **_: object) -> None:
                self.stream = stream

            def run(self, _: unittest.TestSuite) -> MagicMock:
                self.stream.write("FAIL: example test\nassertion failed\n")
                return result

        with patch.object(test_runner, "_require_virtual_environment"), \
                patch.object(test_runner, "load_config", return_value=MagicMock()), \
                patch.object(test_runner, "ensure_output_layout", return_value=MagicMock()), \
                patch.object(test_runner, "get_application_logger", return_value=logger), \
                patch.object(test_runner.unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()), \
                patch.object(test_runner.unittest, "TextTestRunner", FailingRunner):
            self.assertEqual(1, test_runner.main([]))

        logger.box.assert_any_call("Test failures", ("FAIL: example test", "assertion failed"))
        logger.event.assert_any_call("ERROR", "test_suite_failed", source="logicytics.cli.tests")


if __name__ == "__main__":
    unittest.main()
