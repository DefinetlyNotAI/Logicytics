"""Safe, reusable command execution and structured output parsing for collectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from logicytics.platform_adapters import process_adapter


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured result of a non-shell command invocation."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run_command(command: Iterable[str], *, timeout_seconds: float = 30) -> CommandResult:
    """Run an explicit command without a shell and return decoded captured streams."""
    normalized = tuple(str(argument) for argument in command)
    if not normalized:
        raise ValueError("command must contain at least one argument")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    completed = process_adapter.run(
        normalized, capture_output=True, check=False, text=True, timeout=timeout_seconds
    )
    return CommandResult(normalized, completed.returncode, completed.stdout, completed.stderr)


def parse_level_messages(output: str) -> tuple[tuple[str, str], ...]:
    """Parse `LEVEL: message` lines into normalized structured entries."""
    messages: list[tuple[str, str]] = []
    for line in output.splitlines():
        level, separator, message = line.partition(":")
        normalized_level = level.strip().upper()
        if not separator or normalized_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "INTERNAL",
                                                     "EXCEPTION"}:
            continue
        messages.append((normalized_level, message.strip()))
    return tuple(messages)
