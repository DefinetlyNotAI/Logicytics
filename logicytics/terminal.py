"""Interactive terminal lifecycle controls for user-facing Logicytics commands."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from logicytics.module.presentation import render_banner

_ACTIVE_SESSIONS = 0


def _is_interactive_terminal() -> bool:
    """Return whether this process owns a terminal that can be visibly managed."""
    return sys.stdout.isatty() or sys.stderr.isatty()


def _clear_terminal() -> None:
    """Clear the shared terminal screen through the platform's real console command."""
    os.system("cls" if os.name == "nt" else "clear")
    render_banner(sys.stderr)


@contextmanager
def terminal_lifecycle() -> Iterator[None]:
    """Clear an interactive screen once and leave one final newline on normal unwind."""
    global _ACTIVE_SESSIONS
    interactive = _is_interactive_terminal()
    outermost = _ACTIVE_SESSIONS == 0
    if outermost and interactive:
        _clear_terminal()
    _ACTIVE_SESSIONS += 1
    try:
        yield
    finally:
        _ACTIVE_SESSIONS -= 1
        if outermost and interactive:
            sys.stdout.write("\n")
            sys.stdout.flush()
