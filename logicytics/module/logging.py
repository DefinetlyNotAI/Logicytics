"""Run-scoped structured logging without global mutable logging configuration."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import sys
import textwrap
import traceback
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import perf_counter, time
from typing import ParamSpec, TextIO, TypeVar

from logicytics.contracts import EventLogger
from logicytics.module.configuration import LoggingSettings
from logicytics.module.presentation import (
    console_width as presentation_console_width,
)
from logicytics.module.presentation import (
    render_alert as render_presentation_alert,
)
from logicytics.module.presentation import (
    render_section as render_presentation_section,
)
from logicytics.module.presentation import (
    render_step_heading as render_presentation_step_heading,
)
from logicytics.module.redaction import redact_mapping, redact_text

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")
_LEVEL_ORDER = {
    "DEBUG": 10,
    "INTERNAL": 15,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "EXCEPTION": 45,
    "CRITICAL": 50,
}
_LEVEL_PRESENTATION = {
    "DEBUG": ("\u00b7", "\033[90m", "\033[90m"),
    "INTERNAL": ("\u00b7", "\033[90m", "\033[90m"),
    "INFO": ("\u25cf", "\033[96m", "\033[97m"),
    "WARNING": ("!", "\033[93m", "\033[93m"),
    "ERROR": ("\u00d7", "\033[91m", "\033[91m"),
    "EXCEPTION": ("\u00d7", "\033[91m", "\033[91m"),
    "CRITICAL": ("\u00d7", "\033[31m", "\033[31m"),
}
_DETAIL_MARKER_COLOR = "\033[95m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_FILE_LOG_LINE_WIDTH = 140
_TIME_WIDTH = 23
_SEVERITY_WIDTH = 9
_SOURCE_WIDTH = 28
_RECORD_START = re.compile(rb"(?m)^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})? \|")
_EVENT_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+$")
_LOGGER_LOCK = RLock()
_APPLICATION_LOGGERS: dict[Path, ApplicationLogger] = {}
_EVENT_LOGGERS: dict[tuple[Path, str, str | None], FileEventLogger] = {}
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _caller_location() -> tuple[str, int]:
    """Return the first caller outside this logging module and its source line."""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            module = str(frame.f_globals.get("__name__", ""))
            if module and module != __name__:
                return module, frame.f_lineno
            frame = frame.f_back
    finally:
        del frame
    return __name__, 0


def _script_initials(script: str) -> str:
    """Use the first letter of each underscore-delimited script word."""
    return "".join(part[0] for part in script.split("_") if part)


def collector_log_source(collector_id: str | None) -> str | None:
    """Render a stable human-facing source name for one collector identifier."""
    if not collector_id:
        return None
    parts = collector_id.split(".")
    if len(parts) < 2:
        return collector_id
    if parts[0] == "core" and len(parts) >= 3:
        return f"core.{parts[1]}.{_script_initials(parts[-1])}"
    if parts[0] == "plugin":
        return f"plugins.{'.'.join(parts[1:])}"
    return collector_id


def _source_with_line(source: str, line: int, *, debug: bool) -> str:
    """Add a source line in debug logs, except for library implementation sources."""
    if debug and line > 0 and not source.startswith("library."):
        return f"{source}:{line}"
    return source


def _normalize_level(level: str) -> str:
    """Normalize one severity spelling and reject values no sink can interpret."""
    if not isinstance(level, str):
        raise ValueError(f"unsupported log level: {level!r}")
    normalized = level.strip().upper()
    if normalized not in _LEVEL_ORDER:
        raise ValueError(f"unsupported log level: {level}")
    return normalized


class ApplicationLogger(EventLogger):
    """Thread-safe human-readable file and colored-console event sink."""

    def __init__(
            self,
            path: Path,
            settings: LoggingSettings,
            *,
            console: TextIO | None = None,
    ) -> None:
        """Initialize a bounded application logger with explicit file and console sinks."""
        self.path = path.resolve()
        self.settings = settings
        self.console = sys.stderr if console is None else console
        self._lock = RLock()
        self._last_console_step: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prepare_file()

        self._last_console_line = ""
        self._progress_indent = ""
        self._progress_checked = 0
        self._progress_total = 0
        self._progress_current = ""

    @staticmethod
    def _visible_text(text: str) -> str:
        """Return text with ANSI escape sequences removed."""
        return _ANSI_ESCAPE.sub("", text)

    def _leading_indent(self, text: str) -> str:
        """Return the leading whitespace from visible terminal text."""
        visible = self._visible_text(text)

        return visible[: len(visible) - len(visible.lstrip())]

    def _remember_console_line(self, text: str) -> None:
        """Remember the last visible non-empty console line."""
        visible = self._visible_text(text).rstrip("\n")

        if "\n" in visible:
            visible = visible.split("\n")[-1]

        if visible:
            self._last_console_line = visible

    @staticmethod
    def _shorten_component(component: str) -> str:
        """
        Shorten one dotted-name component to initials.

        Examples:
            packet_capture -> pc
            process_memory_map -> pmm
            network -> n
            collector -> c
        """
        parts = [part for part in component.split("_") if part]

        if not parts:
            return component

        if len(parts) == 1:
            return parts[0][0]

        return "".join(part[0] for part in parts)

    def _shorten_current(self, current: str, max_width: int) -> str:
        """
        Render dotted collector names in compact code form.

        The final component is always abbreviated from underscore-separated
        words, while namespaces remain readable until width requires further
        shortening.

        Examples:
            collector.network.packet_capture
                -> collector.network.pc
                -> collector.n.pc
                -> c.n.pc

            collector.system.process_memory_map
                -> collector.system.pmm
                -> collector.s.pmm
                -> c.s.pmm
        """
        if not current:
            return ""

        components = current.split(".")

        if len(components) <= 1:
            return current

        shortened = components.copy()

        # Always use compact code form for the final component.
        shortened[-1] = self._shorten_component(shortened[-1])

        result = ".".join(shortened)

        if len(result) <= max_width:
            return result

        # Shorten namespace components from right to left while preserving
        # the leading namespace for as long as possible.
        for index in range(len(shortened) - 2, 0, -1):
            shortened[index] = self._shorten_component(shortened[index])
            result = ".".join(shortened)

            if len(result) <= max_width:
                return result

        # Finally abbreviate the first component.
        shortened[0] = self._shorten_component(shortened[0])
        result = ".".join(shortened)

        if len(result) <= max_width:
            return result

        if max_width <= 0:
            return ""

        if max_width <= 3:
            return result[-max_width:]

        return f"…{result[-(max_width - 1):]}"

    def _prepare_file(self) -> None:
        """Apply explicit deletion, retention, and bounded truncation policies."""
        if self.settings.delete_previous and self.path.exists():
            self.path.unlink()
        cutoff = time() - self.settings.retention_days * 86400
        for candidate in self.path.parent.glob("Logicytics*.log"):
            if candidate != self.path and candidate.is_file() and candidate.stat().st_mtime < cutoff:
                candidate.unlink()
        self._truncate_file()

    def _truncate_file(self) -> None:
        """Retain the newest complete rows when the configured byte limit is exceeded."""
        if self.path.is_file() and self.path.stat().st_size > self.settings.maximum_bytes:
            with self.path.open("rb") as stream:
                stream.seek(-self.settings.maximum_bytes, 2)
                stream.readline()
                retained = stream.read()
            record = _RECORD_START.search(retained)
            retained = retained[record.start():] if record is not None else b""
            self.path.write_bytes(retained)

    @staticmethod
    def _rows(level: str, source: str, message: str) -> tuple[str, ...]:
        """Format AIBrain-style fixed columns and aligned wrapped rows."""
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        source_column = source
        prefix = f"{timestamp:<{_TIME_WIDTH}} | {level:<{_SEVERITY_WIDTH}} | {source_column:<{_SOURCE_WIDTH}} | "
        continuation = f"{'':<{_TIME_WIDTH}} | {'':<{_SEVERITY_WIDTH}} | {'':<{_SOURCE_WIDTH}} | "
        available = max(_FILE_LOG_LINE_WIDTH - len(prefix), 1)
        wrapped = [
            segment
            for line in message.splitlines() or [""]
            for segment in (
                    textwrap.wrap(
                        line,
                        width=available,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                    or [""]
            )
        ]
        first = prefix + wrapped[0]
        return tuple([first, *(continuation + row for row in wrapped[1:])])

    @staticmethod
    def _console_width() -> int:
        """Return the AIBrain console width with its safety margin and minimum."""
        return presentation_console_width()

    def _supports_unicode(self) -> bool:
        """Return whether this console can encode AIBrain's presentation glyphs."""
        encoding = getattr(self.console, "encoding", None) or "utf-8"
        try:
            "\u00b7\u25cf\u00d7\u256d\u2500\u256e\u2502\u251c\u2524\u2570\u256f".encode(encoding)
        except (LookupError, UnicodeEncodeError):
            return False
        return True

    @classmethod
    def _console_rows(cls, marker: str, message: str) -> tuple[str, ...]:
        """Word-wrap compact status text with aligned continuation indentation."""
        prefix = f"  {marker} "
        continuation = " " * len(prefix)
        width = cls._console_width()
        rows: list[str] = []
        for index, raw_line in enumerate(message.expandtabs(4).splitlines() or [""]):
            indentation = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            remaining = raw_line.lstrip().rstrip()
            current_prefix = (prefix if index == 0 else continuation) + indentation
            while len(remaining) > max(width - len(current_prefix), 1):
                available = max(width - len(current_prefix), 1)
                split_at = remaining.rfind(" ", 0, available + 1)
                if split_at <= 0:
                    split_at = available
                rows.append(current_prefix + remaining[:split_at].rstrip())
                remaining = remaining[split_at:].lstrip()
                current_prefix = continuation + indentation
            rows.append(current_prefix + remaining)
        return tuple(rows)

    @staticmethod
    def _color_console_text(row: str, text_color: str) -> str:
        """Color a leading detail marker purple while retaining the event color."""
        indentation_length = len(row) - len(row.lstrip())
        detail = row[indentation_length:]
        if detail.startswith("> "):
            return f"{text_color}{_BOLD}{row[:indentation_length]}{_DETAIL_MARKER_COLOR}{_BOLD}>{_RESET}{text_color}{_BOLD}{detail[1:]}"
        return f"{text_color}{_BOLD}{row}"

    @staticmethod
    def _console_message(message: str) -> str:
        """Make machine-oriented lifecycle event names readable on the console."""
        if _EVENT_NAME.fullmatch(message):
            return message.replace("_", " ").capitalize()
        return message

    @classmethod
    def _message_lines(cls, message: str) -> tuple[str, ...]:
        """Render ordinary or JSON-looking messages as readable presentation lines."""
        safe_message = redact_text(message)
        stripped = safe_message.strip()
        if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                rows = ["Structured details"]
                rows.extend(cls._console_fields(parsed))
                return tuple(rows)
            if isinstance(parsed, list):
                rows = ["Structured details"]
                rows.extend(f"Item {index}: {cls._console_value(item)}" for index, item in enumerate(parsed, start=1))
                return tuple(rows)
        return (cls._console_message(safe_message),)

    @classmethod
    def _console_fields(
            cls,
            fields: Mapping[str, object],
            *,
            compact_configuration_hash: bool = False,
    ) -> tuple[str, ...]:
        """Render structured fields as readable labels instead of JSON fragments."""
        rows: list[str] = []
        for key, value in fields.items():
            label = key.replace("_", " ").capitalize().replace("Github", "GitHub")
            rendered_value = cls._console_value(value)
            if compact_configuration_hash and key in {"configuration_hash", "fingerprint"}:
                rendered_value = rendered_value[:7]
            rendered = rendered_value.splitlines() or ["none"]
            rows.append(f"> {label}: {rendered[0]}")
            rows.extend(f"  {line}" for line in rendered[1:])
        return tuple(rows)

    @staticmethod
    def _lifecycle_step(message: str) -> str | None:
        """Map lifecycle events to the console phase that owns their details."""
        if not _EVENT_NAME.fullmatch(message):
            return None
        if message.startswith("run_packag") or message.startswith("package_"):
            return "Packaging"
        prefix = message.partition("_")[0]
        return {
            "command": "Command",
            "preflight": "Preflight",
            "plan": "Planning",
            "run": "Collection",
            "collector": "Collector execution",
            "package": "Packaging",
            "update": "Update",
            "development": "Development",
        }.get(prefix)

    def _render_step(self, message: str) -> None:
        """Insert a visual break when lifecycle output moves to a new phase."""
        step = self._lifecycle_step(message)
        if step is None or step == self._last_console_step:
            return
        render_presentation_step_heading(
            self.console,
            step,
            width=self._console_width,
        )
        self._last_console_step = step

    @classmethod
    def _console_value(cls, value: object) -> str:
        """Normalize common structured values for compact, human-readable console output."""
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, Mapping):
            if not value:
                return "none"
            return "; ".join(
                f"{str(key).replace('_', ' ')}: {cls._console_value(item)}"
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            )
        if isinstance(value, (set, frozenset)):
            if not value:
                return "none"
            rendered_items = sorted(cls._console_value(item) for item in value)
            return ", ".join(rendered_items)
        if isinstance(value, (list, tuple)):
            if not value:
                return "none"
            return ", ".join(cls._console_value(item) for item in value)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"<{len(value)} bytes>"
        if isinstance(value, float):
            if not math.isfinite(value):
                return "non-finite float"
            return f"{value:.6f}".rstrip("0").rstrip(".")
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None:
                    return cls._console_value(parsed)
        return str(value)

    def event(self, level: str, message: str, *, console: bool = True, **fields: float | str) -> None:
        """Dispatch one typed, redacted event to configured console and file sinks."""
        normalized = _normalize_level(level)
        minimum_level = _normalize_level(self.settings.level)
        if _LEVEL_ORDER[normalized] < _LEVEL_ORDER[minimum_level]:
            return
        safe_message = redact_text(message)
        safe_fields = redact_mapping(fields)
        caller_module, caller_line = _caller_location()
        source = str(safe_fields.pop("source", caller_module))
        if source in {"cli", "logicytics.cli", "runtime", "logicytics.runtime"}:
            source = caller_module
        source = _source_with_line(source, caller_line, debug=minimum_level == "DEBUG")
        file_lines = list(self._message_lines(safe_message))
        file_lines.extend(self._console_fields(safe_fields))
        rendered_message = "\n".join(file_lines)
        console_lines = list(self._message_lines(safe_message))
        console_lines.extend(
            self._console_fields(
                safe_fields,
                compact_configuration_hash=minimum_level != "DEBUG",
            )
        )
        console_message = "\n".join(console_lines)
        rows = self._rows(normalized, source, rendered_message)
        with self._lock:
            if self.settings.file_enabled:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write("\n".join(rows) + "\n")
                self._truncate_file()
            if self.settings.console_enabled and console:
                self._render_step(message)
                marker, marker_color, text_color = _LEVEL_PRESENTATION[normalized]

                if not self._supports_unicode():
                    marker = {"●": "*", "×": "X", "·": "."}.get(marker, marker)

                console_rows = self._console_rows(marker, console_message)

                if self.settings.color_enabled and self.console.isatty():
                    marker_prefix = f"  {marker} "
                    first_row = self._color_console_text(
                        console_rows[0][len(marker_prefix):],
                        text_color,
                    )
                    colored_rows = (
                        f"{marker_color}{_BOLD}"
                        f"{marker_prefix}{_RESET}"
                        f"{first_row}"
                    )

                    if len(console_rows) > 1:
                        colored_rows += "\n" + "\n".join(
                            self._color_console_text(row, text_color)
                            for row in console_rows[1:]
                        )

                    self.console.write(f"{colored_rows}{_RESET}\n")
                else:
                    self.console.write("\n".join(console_rows) + "\n")

                self.console.flush()
                self._remember_console_line(console_rows[-1])

    def incomplete_progress(
            self,
            label: str,
            checked: int | None = None,
            total: int | None = None,
            status: str = "Incomplete",
    ) -> None:
        """Replace the active progress bar with an incomplete x/y status."""
        if not self.settings.console_enabled or not self.console.isatty():
            return

        bounded_total = max(
            total if total is not None else self._progress_total,
            1,
        )
        bounded_checked = min(
            max(
                checked if checked is not None else self._progress_checked,
                0,
            ),
            bounded_total,
        )

        rendered = (
            f"{self._progress_indent}"
            f"{label} [{status}] "
            f"{bounded_checked}/{bounded_total}"
        )

        with self._lock:
            if self.settings.color_enabled:
                rendered = f"\033[91m\033[1m{rendered}\033[0m"

            self.console.write(f"\r\033[2K{rendered}\n")
            self.console.flush()

        self._remember_console_line(rendered)

        self._progress_indent = ""
        self._progress_checked = 0
        self._progress_total = 0
        self._progress_current = ""

    def progress(
            self,
            label: str,
            checked: int,
            total: int,
            current: str = "",
    ) -> None:
        """Render one full-width adaptive in-place progress bar."""
        if not self.settings.console_enabled or not self.console.isatty():
            return

        bounded_total = max(total, 1)
        bounded_checked = min(max(checked, 0), bounded_total)

        if checked == 0 or not self._progress_indent:
            self._progress_indent = self._leading_indent(
                self._last_console_line
            )

        self._progress_checked = bounded_checked
        self._progress_total = bounded_total
        self._progress_current = current

        console_width = self._console_width()
        count = f"{bounded_checked}/{bounded_total}"

        if current:
            display_current = self._shorten_current(
                current,
                max_width=console_width,
            )
        else:
            display_current = ""

        suffix = f" {display_current}" if display_current else ""

        # Layout:
        #
        # <indent><label> <bar> <x/y> <current>
        #
        # The bar consumes every remaining terminal column.
        fixed_width = (
                len(self._progress_indent)
                + len(label)
                + 1
                + 1
                + len(count)
                + len(suffix)
        )

        bar_width = max(console_width - fixed_width, 1)

        # If the current value still makes the line too wide, shorten it again
        # using the actual space available after reserving one bar character.
        if display_current:
            maximum_current_width = max(
                console_width
                - len(self._progress_indent)
                - len(label)
                - len(count)
                - 4,
                1,
            )

            display_current = self._shorten_current(
                current,
                maximum_current_width,
            )
            suffix = f" {display_current}"

            fixed_width = (
                    len(self._progress_indent)
                    + len(label)
                    + 1
                    + 1
                    + len(count)
                    + len(suffix)
            )

            bar_width = max(console_width - fixed_width, 1)

        ratio = bounded_checked / bounded_total
        filled = min(int(bar_width * ratio), bar_width)
        unfilled = bar_width - filled

        prefix = (
            f"{self._progress_indent}"
            f"{label} "
        )

        postfix = (
            f" {count}"
            f"{suffix}"
        )

        with self._lock:
            self.console.write("\r\033[2K")

            if self.settings.color_enabled:
                # Prefix.
                self.console.write(
                    f"\033[96m\033[1m{prefix}"
                )

                # Completed section, bright white.
                if filled:
                    self.console.write(
                        f"\033[97m\033[1m"
                        f"{'━' * filled}"
                    )

                # Remaining section, dim gray.
                if unfilled:
                    self.console.write(
                        f"\033[90m"
                        f"{'━' * unfilled}"
                    )

                # Counter/current text.
                self.console.write(
                    f"\033[96m\033[1m"
                    f"{postfix}"
                    f"\033[0m"
                )
            else:
                self.console.write(
                    f"{prefix}"
                    f"{'━' * filled}"
                    f"{'─' * unfilled}"
                    f"{postfix}"
                )

            if bounded_checked >= bounded_total:
                self.console.write("\r\033[2K")

            self.console.flush()

    def raw(self, message: str, *, end: str = "\n") -> None:
        """Write redacted console-only presentation that never pollutes the event log."""
        if end not in {"", "\n"}:
            raise ValueError("raw log end must be empty or a newline")

        safe = redact_text(message)

        with self._lock:
            if self.settings.console_enabled:
                self.console.write(safe + end)
                self.console.flush()

                if end == "\n":
                    self._remember_console_line(safe)

    def separator(self) -> None:
        """Write one presentation-only blank line to the configured console."""
        self.raw("")

    @classmethod
    def render_section(
            cls,
            console: TextIO,
            title: str,
            lines: Iterable[str],
    ) -> None:
        """Render a plain, indented console section for startup and fallback paths."""
        render_presentation_section(
            console,
            title,
            lines,
            message_lines=cls._message_lines,
            width=cls._console_width,
        )

    @classmethod
    def render_alert(
            cls,
            console: TextIO,
            title: str,
            lines: Iterable[str],
    ) -> None:
        """Render a bordered startup or error alert using the shared logger presentation."""
        render_presentation_alert(
            console,
            title,
            lines,
            message_lines=cls._message_lines,
            width=cls._console_width,
            color_enabled=console.isatty(),
        )

    @classmethod
    def render_error(cls, console: TextIO, title: str, lines: Iterable[str]) -> None:
        """Render a red error alert while retaining purple detail markers."""
        render_presentation_alert(
            console,
            title,
            lines,
            message_lines=cls._message_lines,
            width=cls._console_width,
            color_enabled=console.isatty(),
            error=True,
        )

    def box(self, title: str, lines: Iterable[str]) -> None:
        """Render console-only output as plain redacted lines."""
        rendered_lines = tuple(lines)

        with self._lock:
            if self.settings.console_enabled:
                render_presentation_section(
                    self.console,
                    title,
                    rendered_lines,
                    message_lines=self._message_lines,
                    width=self._console_width,
                    color_enabled=self.settings.color_enabled,
                )

                self._last_console_step = title

                if rendered_lines:
                    self._remember_console_line(str(rendered_lines[-1]))

    def dispatch(self, messages: Iterable[str]) -> None:
        """Parse and dispatch a batch of optional `LEVEL: message` rows."""
        for message in messages:
            level, separator, text = message.partition(":")
            normalized = level.strip().upper()
            if separator and normalized in _LEVEL_ORDER:
                self.event(normalized, text.strip())
            else:
                self.event("INFO", message.strip())


class HumanArgumentParser(argparse.ArgumentParser):
    """Present argparse failures through the same readable CLI section layout."""

    def reference_lines(self) -> tuple[str, ...]:
        """Return every visible option as compact, presentation-ready help rows."""
        usage = " ".join(self.format_usage().split())
        while usage.lower().startswith("usage:"):
            usage = usage[len("usage:"):].strip()
        usage = usage.replace(",", ", ")
        rows = [f"Usage: {usage}", "Options:"]
        formatter = self._get_formatter()
        for action in self._actions:
            if action.help is argparse.SUPPRESS:
                continue
            invocation = re.sub(r",\s*", ", ", formatter._format_action_invocation(action))
            description = formatter._expand_help(action).strip() if action.help else "No description provided."
            rows.append(f"{invocation} - {description}")
        return tuple(rows)

    def error(self, message: str) -> None:
        """Render a concise error followed by the complete command reference."""
        ApplicationLogger.render_error(
            sys.stderr,
            "Command-line error",
            (f"Error: {message}",),
        )
        ApplicationLogger.render_section(
            sys.stderr,
            "Available command options",
            self.reference_lines(),
        )
        raise SystemExit(2)


def get_application_logger(
        path: Path,
        settings: LoggingSettings,
        *,
        console: TextIO | None = None,
) -> ApplicationLogger:
    """Return one configured logger instance per canonical application log path."""
    resolved = path.resolve()
    with _LOGGER_LOCK:
        logger = _APPLICATION_LOGGERS.get(resolved)
        if logger is None or logger.settings != settings or console is not None or (
                console is None and logger.console is not sys.stderr):
            logger = ApplicationLogger(resolved, settings, console=console)
            if console is None:
                _APPLICATION_LOGGERS[resolved] = logger
        return logger


class FileEventLogger(EventLogger):
    """Append JSONL events to a single run- or collector-owned log file."""

    def __init__(self, path: Path, *, run_id: str, collector_id: str | None = None) -> None:
        """Initialize an append-only JSONL logger scoped to one run and collector."""
        self.path = path
        self.run_id = run_id
        self.collector_id = collector_id
        self._event_lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, level: str, message: str, *, console: bool = True, **fields: float | str) -> None:
        """Write a timestamped, structured event without relying on global handlers."""
        normalized = _normalize_level(level)
        caller_module, _ = _caller_location()
        source = collector_log_source(self.collector_id) or caller_module
        payload: dict[str, object] = {
            "at": datetime.now(UTC).isoformat(),
            "level": normalized.lower(),
            "message": redact_text(message),
            "run_id": self.run_id,
            "source": source,
        }
        if self.collector_id is not None:
            payload["collector_id"] = self.collector_id
        if fields:
            payload["fields"] = redact_mapping(fields)
        with self._event_lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def get_event_logger(path: Path, *, run_id: str, collector_id: str | None = None) -> FileEventLogger:
    """Return the process-local singleton for one canonical engine or collector channel."""
    identity = (path.resolve(), run_id, collector_id)
    with _LOGGER_LOCK:
        logger = _EVENT_LOGGERS.get(identity)
        if logger is None:
            logger = FileEventLogger(identity[0], run_id=run_id, collector_id=collector_id)
            _EVENT_LOGGERS[identity] = logger
        return logger


def timed(logger: EventLogger, *, level: str = "info") -> Callable[
    [Callable[Parameters, Result]], Callable[Parameters, Result]]:
    """Decorate a function so structured start, finish, error, and duration events are written."""

    def decorate(function: Callable[Parameters, Result]) -> Callable[Parameters, Result]:
        """Wrap one callable with lifecycle timing events."""

        def wrapped(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
            """Emit start, completion, or failure events around one invocation."""
            logger.event(level, "function_started", function=function.__qualname__)
            started = perf_counter()
            try:
                value = function(*args, **kwargs)
            except Exception as error:
                logger.event(
                    "error",
                    "function_failed",
                    function=function.__qualname__,
                    duration_seconds=round(perf_counter() - started, 6),
                    error_type=type(error).__name__,
                )
                raise
            logger.event(
                level,
                "function_finished",
                function=function.__qualname__,
                duration_seconds=round(perf_counter() - started, 6),
            )
            return value

        return wrapped

    return decorate


def raise_logged(logger: EventLogger, exception_type: type[Exception], message: str, **fields: float | str) -> None:
    """Record a structured exception event, then raise the requested exception type."""
    logger.event("exception", message, exception_type=exception_type.__name__, **fields)
    raise exception_type(message)


def deprecated(
        logger: EventLogger, *, removal_version: str, reason: str, include_stack: bool = False
) -> Callable[[Callable[Parameters, Result]], Callable[Parameters, Result]]:
    """Decorate a function so each invocation emits a structured deprecation warning."""

    def decorate(function: Callable[Parameters, Result]) -> Callable[Parameters, Result]:
        """Wrap one callable with a structured deprecation warning."""

        def wrapped(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
            """Emit the configured warning and then invoke the deprecated callable."""
            fields: dict[str, str] = {
                "function": function.__qualname__,
                "removal_version": removal_version,
                "reason": reason,
            }
            if include_stack:
                fields["stack"] = "".join(traceback.format_stack(limit=8))
            logger.event("warning", "function_deprecated", **fields)
            return function(*args, **kwargs)

        return wrapped

    return decorate
