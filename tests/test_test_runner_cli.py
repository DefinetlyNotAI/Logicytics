from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from logicytics.cli import tests as test_runner


class TestRunnerCliTests(unittest.TestCase):
    """Presentation and result handling for the dynamically discovered test command."""

    def test_runner_requires_activating_the_existing_local_environment(self) -> None:
        """The test script must not run outside the local virtual environment."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activation_script = root / ".venv" / "Scripts" / "Activate.ps1"
            activation_script.parent.mkdir(parents=True)
            activation_script.write_text("", encoding="utf-8")
            output = io.StringIO()

            with patch.object(test_runner, "project_root", return_value=root), patch.object(
                    sys,
                    "prefix",
                    sys.base_prefix,
            ), patch("sys.stderr", output):
                self.assertEqual(2, test_runner.main([]))

            self.assertIn(r".\.venv\Scripts\Activate.ps1", output.getvalue())

    def test_runner_requires_the_installer_when_the_local_environment_is_missing(self) -> None:
        """The test script points to the installer when no usable local environment exists."""
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()

            with patch.object(test_runner, "project_root", return_value=Path(temporary)), patch.object(
                    sys,
                    "prefix",
                    sys.base_prefix,
            ), patch("sys.stderr", output):
                self.assertEqual(2, test_runner.main([]))

            rendered = " ".join(output.getvalue().split())
            self.assertIn("python -m logicytics.cli.installer", rendered)

    def test_runner_reports_discovered_suite_through_the_application_console(self) -> None:
        """A successful dynamic run renders a line-based summary and returns CI success."""
        result = MagicMock()
        result.failures = ()
        result.errors = ()
        result.skipped = (object(),)
        result.testsRun = 7
        result.wasSuccessful.return_value = True
        logger = MagicMock()
        runner = MagicMock()
        runner.run.return_value = result

        with patch("logicytics.module.configuration.load_config", return_value=MagicMock()), \
                patch("logicytics.module.output_layout.ensure_output_layout", return_value=MagicMock()), \
                patch("logicytics.module.logging.get_application_logger", return_value=logger), \
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

        with patch("logicytics.module.configuration.load_config", return_value=MagicMock()), \
                patch("logicytics.module.output_layout.ensure_output_layout", return_value=MagicMock()), \
                patch("logicytics.module.logging.get_application_logger", return_value=logger), \
                patch.object(test_runner.unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()), \
                patch.object(test_runner.unittest, "TextTestRunner", FailingRunner):
            self.assertEqual(1, test_runner.main([]))

        logger.box.assert_any_call("Test failures", ("FAIL: example test", "assertion failed"))
        logger.event.assert_any_call("ERROR", "test_suite_failed", source="logicytics.cli.tests")


if __name__ == "__main__":
    unittest.main()
