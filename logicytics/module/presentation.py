"""Shared redacted console presentation for logging and bootstrap errors."""

from __future__ import annotations

import shutil
import textwrap
from collections.abc import Callable, Iterable
from typing import TextIO

from logicytics.module.redaction import redact_text

_DEFAULT_CONSOLE_WIDTH = 82
_MIN_CONSOLE_WIDTH = 60
_RIGHT_EDGE_MARGIN = 4
_DETAIL_MARKER_COLOR = "\033[95m"
_INFO_COLOR = "\033[96m"
_WARNING_COLOR = "\033[93m"
_ERROR_COLOR = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

MessageLines = Callable[[str], tuple[str, ...]]
ConsoleWidth = Callable[[], int]


def terminal_width() -> int:
    """Return the full physical terminal width for full-span presentation."""
    width = shutil.get_terminal_size((_DEFAULT_CONSOLE_WIDTH, 24)).columns
    return max(width, _MIN_CONSOLE_WIDTH)


def console_width() -> int:
    """Return the shared terminal width with a safety margin and minimum."""
    width = shutil.get_terminal_size((_DEFAULT_CONSOLE_WIDTH, 24)).columns
    return max(width - _RIGHT_EDGE_MARGIN, _MIN_CONSOLE_WIDTH)


def _plain_message_lines(message: str) -> tuple[str, ...]:
    """Render bootstrap text safely without importing the full logger stack."""
    return (redact_text(message),)


def _heading(title: str, *, width: ConsoleWidth = console_width) -> str:
    """Return a title followed by a full-width ASCII rule."""
    safe_title = redact_text(title).strip() or "Logicytics"
    available = max(width(), _MIN_CONSOLE_WIDTH)
    return f"{safe_title}\n" + "-" * available


def render_banner(console: TextIO, *, width: ConsoleWidth = terminal_width) -> None:
    """Render a full-width, strictly ASCII startup banner after a real clear."""
    available = max(width(), _MIN_CONSOLE_WIDTH)
    title = "LOGICYTICS"
    subtitle = "Local evidence collection framework"
    rule = "-" * available
    if console.isatty():
        rows = (
            "",
            f"{_INFO_COLOR}{_BOLD}{title}{_RESET}",
            f"{_INFO_COLOR}{rule}{_RESET}",
            f"\033[97m{subtitle}{_RESET}",
            "\n",
        )
    else:
        rows = ("", title, rule, subtitle, "")
    console.write("\n".join(rows))
    console.flush()


def render_step_heading(
    console: TextIO,
    title: str,
    *,
    width: ConsoleWidth = console_width,
) -> None:
    """Separate a lifecycle phase with a titled ASCII rule and breathing room."""
    console.write(f"\n{_heading(title, width=width)}\n\n")
    console.flush()


def render_section(
    console: TextIO,
    title: str,
    lines: Iterable[str],
    *,
    message_lines: MessageLines = _plain_message_lines,
    width: ConsoleWidth = console_width,
    color_enabled: bool | None = None,
) -> None:
    """Render the shared indented, wrapped console section presentation."""
    use_color = (console.isatty() if color_enabled is None else color_enabled) and console.isatty()
    heading = _heading(title, width=width)
    if use_color:
        heading_title, heading_rule = heading.split("\n", 1)
        heading = f"{_INFO_COLOR}{_BOLD}{heading_title}{_RESET}\n{_INFO_COLOR}{heading_rule}{_RESET}"
    rows = [heading, ""]
    for line in lines:
        for raw_row in line.splitlines() or [""]:
            safe_row = redact_text(raw_row)
            indentation = safe_row[: len(safe_row) - len(safe_row.lstrip())]
            remaining = safe_row.lstrip().rstrip()
            prefix = f"  > {indentation}"
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
                first_row = row_prefix + wrapped[0]
                if use_color and presentation_index == 0:
                    first_row = f"{prefix[:-2]}{_DETAIL_MARKER_COLOR}{_BOLD}>{_RESET} {indentation}{wrapped[0]}"
                rows.append(first_row)
                rows.extend(f"{continuation_prefix}{part}" for part in wrapped[1:])
    console.write("\n" + "\n".join(rows) + "\n")
    console.flush()


def render_alert(
    console: TextIO,
    title: str,
    lines: Iterable[str],
    *,
    message_lines: MessageLines = _plain_message_lines,
    width: ConsoleWidth = console_width,
    color_enabled: bool | None = None,
    error: bool = False,
) -> None:
    """Render a severity-marked alert with a clean title and ASCII rule."""
    available = max(width(), _MIN_CONSOLE_WIDTH)
    safe_title = redact_text(title).strip()
    use_color = (console.isatty() if color_enabled is None else color_enabled) and console.isatty()
    alert_color = _ERROR_COLOR if error else _WARNING_COLOR
    if use_color:
        rows = [
            f"{alert_color}{_BOLD}{safe_title}{_RESET}",
            f"{alert_color}{'-' * available}{_RESET}",
            "",
        ]
    else:
        rows = [safe_title, "-" * available, ""]
    for line_index, line in enumerate(lines):
        marker = "!" if line_index == 0 else ">"
        prefix = f"  {marker} "
        continuation_prefix = "    "
        for raw_row in line.splitlines() or [""]:
            for presentation_row in message_lines(raw_row.strip()):
                wrapped = textwrap.wrap(
                    presentation_row,
                    width=max(available - len(prefix), 1),
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
                first_row = prefix + wrapped[0]
                if use_color:
                    marker_prefix = (
                        f"{alert_color}{_BOLD}{prefix}{_RESET}"
                        if marker == "!"
                        else f"  {_DETAIL_MARKER_COLOR}{_BOLD}>{_RESET} "
                    )
                    first_row = f"{marker_prefix}{alert_color}{wrapped[0]}{_RESET}"
                rows.append(first_row)
                rows.extend(
                    f"{alert_color}{continuation_prefix}{row}{_RESET}" if use_color else f"{continuation_prefix}{row}"
                    for row in wrapped[1:]
                )
    console.write("\n" + "\n".join(rows) + "\n")
    console.flush()
