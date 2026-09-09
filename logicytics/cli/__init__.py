"""User-facing Logicytics command entry points."""

from __future__ import annotations

import sys
from pathlib import Path

from logicytics.cli.commands import cli_methods, CLI
from logicytics.terminal import terminal_lifecycle
from logicytics.virtual_environment import (
    is_running_in_virtual_environment,
    render_virtual_environment_error,
)

__all__ = ["CLI", "cli_methods", "main"]


def _project_root() -> Path:
    """Return the repository root without importing the normal CLI implementation."""
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Guard the interpreter before loading the full CLI implementation."""
    try:
        with terminal_lifecycle():
            if not is_running_in_virtual_environment():
                render_virtual_environment_error(sys.stderr, _project_root())
                return 2
            from logicytics.cli.commands import main as command_main

            return command_main(argv)
    except KeyboardInterrupt:
        from logicytics.module.logging import ApplicationLogger

        ApplicationLogger.render_section(sys.stderr, "Command cancelled", ("Interrupted by user.",))
        return 130


def __getattr__(name: str) -> object:
    """Load the normal CLI API only when a caller asks for it."""
    if name in {"CLI", "cli_methods"}:
        from logicytics.cli.commands import CLI, cli_methods

        return {"CLI": CLI, "cli_methods": cli_methods}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
