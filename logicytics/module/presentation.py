"""Shared redacted console presentation for logging and bootstrap errors."""

from __future__ import annotations

import shutil
import textwrap
from collections.abc import Callable, Iterable
from typing import TextIO

from logicytics.module.redaction import redact_text


_DEFAULT_CONSOLE_WIDTH = 100
_MIN_CONSOLE_WIDTH = 40
_RIGHT_EDGE_MARGIN = 2

MessageLines = Callable[[str], tuple[str, ...]]
ConsoleWidth = Callable[[], int]


def console_width() -> int:
    """Return the shared terminal width with a safety margin and minimum."""
    width = shutil.get_terminal_size((_DEFAULT_CONSOLE_WIDTH, 24)).columns
    return max(width - _RIGHT_EDGE_MARGIN, _MIN_CONSOLE_WIDTH)


def _plain_message_lines(message: str) -> tuple[str, ...]:
    """Render bootstrap text safely without importing the full logger stack."""
    return (redact_text(message),)


def render_section(
        console: TextIO,
        title: str,
        lines: Iterable[str],
        *,
        message_lines: MessageLines = _plain_message_lines,
        width: ConsoleWidth = console_width,
) -> None:
    """Render the shared indented, wrapped console section presentation."""
    rows = list(message_lines(title))
    for line in lines:
        for raw_row in line.splitlines() or [""]:
            safe_row = redact_text(raw_row)
            indentation = safe_row[:len(safe_row) - len(safe_row.lstrip())]
            remaining = safe_row.lstrip().rstrip()
            prefix = f"  {indentation}"
            continuation_prefix = f"    {indentation}"
            available = max(width() - len(continuation_prefix), 1)
            presentation_rows = message_lines(remaining)
            for presentation_index, presentation_row in enumerate(presentation_rows):
                wrapped = textwrap.wrap(
                    presentation_row,
                    width=available,
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
                row_prefix = prefix if presentation_index == 0 else continuation_prefix
                rows.append(row_prefix + wrapped[0])
                rows.extend(f"{continuation_prefix}{part}" for part in wrapped[1:])
    console.write("\n".join(rows) + "\n")
    console.flush()
