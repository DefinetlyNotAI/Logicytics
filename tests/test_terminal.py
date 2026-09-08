"""Terminal lifecycle tests for user-facing command startup and completion."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from logicytics import terminal


class _InteractiveBuffer(io.StringIO):
    """A writable test terminal that reports itself as interactive."""

    def isatty(self) -> bool:
        """Report an interactive terminal for lifecycle coverage."""
        return True


class TerminalLifecycleTests(unittest.TestCase):
    """Interactive session clearing and final-newline behavior."""

    def test_windows_clear_uses_the_real_console_command(self) -> None:
        """Windows startup clears the shared screen buffer through cls, not escape bytes."""
        with patch.object(terminal.os, "name", "nt"), patch.object(
                terminal.os,
                "system",
                return_value=0,
        ) as system:
            terminal._clear_terminal()

        system.assert_called_once_with("cls")

    def test_interactive_lifecycle_clears_once_and_ends_with_a_newline(self) -> None:
        """The outer lifecycle performs real clearing once and leaves a clean prompt line."""
        output = _InteractiveBuffer()
        errors = _InteractiveBuffer()

        with patch.object(terminal.sys, "stdout", output), patch.object(
                terminal.sys,
                "stderr",
                errors,
        ), patch.object(terminal, "_clear_terminal") as clear_terminal:
            with terminal.terminal_lifecycle():
                with terminal.terminal_lifecycle():
                    output.write("command output")

        clear_terminal.assert_called_once_with()
        self.assertEqual("command output\n", output.getvalue())

    def test_noninteractive_lifecycle_preserves_machine_output(self) -> None:
        """Redirected output must not receive screen controls or an extra newline."""
        output = io.StringIO()
        errors = io.StringIO()

        with patch.object(terminal.sys, "stdout", output), patch.object(
                terminal.sys,
                "stderr",
                errors,
        ), patch.object(terminal, "_clear_terminal") as clear_terminal:
            with terminal.terminal_lifecycle():
                output.write("machine output")

        clear_terminal.assert_not_called()
        self.assertEqual("machine output", output.getvalue())


if __name__ == "__main__":
    unittest.main()
