"""Terminal lifecycle tests for user-facing command startup and completion."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from logicytics import terminal
from logicytics.module.presentation import render_banner


class _InteractiveBuffer(io.StringIO):
    """A writable test terminal that reports itself as interactive."""

    def isatty(self) -> bool:
        """Report an interactive terminal for lifecycle coverage."""
        return True


class TerminalLifecycleTests(unittest.TestCase):
    """Interactive session clearing and final-newline behavior."""

    def test_windows_clear_uses_the_real_console_command(self) -> None:
        """Windows startup clears the shared screen buffer through cls, not escape bytes."""
        output = _InteractiveBuffer()
        errors = _InteractiveBuffer()
        with patch.object(terminal.sys, "stdout", output), patch.object(
                terminal.sys,
                "stderr",
                errors,
        ), patch.object(
                terminal.os,
                "name",
                "nt",
        ), patch.object(
                terminal.os,
                "system",
                return_value=0,
        ) as system:
            terminal._clear_terminal()

        system.assert_called_once_with("cls")
        self.assertEqual("", output.getvalue())
        self.assertIn("LOGICYTICS", errors.getvalue())

    def test_banner_is_full_width_and_strictly_ascii(self) -> None:
        """The startup banner spans the terminal without Unicode border glyphs."""
        console = io.StringIO()
        render_banner(console, width=lambda: 101)

        rows = console.getvalue().splitlines()
        self.assertEqual(101, len(rows[0]))
        self.assertEqual("+", rows[0][0])
        self.assertEqual("+", rows[0][-1])
        self.assertTrue(all(ord(character) < 128 for row in rows for character in row))

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

    def test_interactive_lifecycle_ends_with_a_newline_after_an_exception(self) -> None:
        """An exception unwinds the terminal lifecycle and leaves a clean prompt line."""
        output = _InteractiveBuffer()
        errors = _InteractiveBuffer()

        with patch.object(terminal.sys, "stdout", output), patch.object(
                terminal.sys,
                "stderr",
                errors,
        ), patch.object(terminal, "_clear_terminal") as clear_terminal:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with terminal.terminal_lifecycle():
                    output.write("command output")
                    raise RuntimeError("interrupted")

        clear_terminal.assert_called_once_with()
        self.assertEqual("command output\n", output.getvalue())


if __name__ == "__main__":
    unittest.main()
