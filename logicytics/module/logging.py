"""Run-scoped structured logging without global mutable logging configuration."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from time import perf_counter, time
from typing import Callable, Iterable, Mapping, ParamSpec, TextIO, TypeVar

from logicytics.contracts import EventLogger
from logicytics.module.configuration import LoggingSettings
from logicytics.module.presentation import render_section as render_presentation_section
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
    "INTERNAL": ("\u00b7", "\033[95m", "\033[95m"),
    "INFO": ("\u25cf", "\033[96m", "\033[97m"),
    "WARNING": ("!", "\033[93m", "\033[93m"),
    "ERROR": ("\u00d7", "\033[91m", "\033[91m"),
    "EXCEPTION": ("\u00d7", "\033[91m", "\033[91m"),
    "CRITICAL": ("\u00d7", "\033[91m", "\033[91m"),
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_FILE_LOG_LINE_WIDTH = 140
_TIME_WIDTH = 23
_SEVERITY_WIDTH = 9
_SOURCE_WIDTH = 28
_DEFAULT_CONSOLE_WIDTH = 82
_MIN_CONSOLE_WIDTH = 60
_RIGHT_EDGE_MARGIN = 4
_RECORD_START = re.compile(rb"(?m)^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})? \|")
_EVENT_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+$")
_LOGGER_LOCK = RLock()
_APPLICATION_LOGGERS: dict[Path, "ApplicationLogger"] = {}
_EVENT_LOGGERS: dict[tuple[Path, str, str | None], "FileEventLogger"] = {}


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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prepare_file()

    def _prepare_file(self) -> None:
        """Apply explicit deletion, retention, and bounded truncation policies."""
        if self.settings.delete_previous and self.path.exists():
            self.path.unlink()
        cutoff = time() - self.settings.retention_days * 86400
        for candidate in self.path.parent.glob("Logicytics*.log"):
            if (
                    candidate != self.path
                    and candidate.is_file()
                    and candidate.stat().st_mtime < cutoff
            ):
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
        source_column = source.removeprefix("logicytics.")
        if len(source_column) > _SOURCE_WIDTH:
            source_column = source_column[:_SOURCE_WIDTH - 3] + "..."
        prefix = (
            f"{timestamp:<{_TIME_WIDTH}} | {level:<{_SEVERITY_WIDTH}} | "
            f"{source_column:<{_SOURCE_WIDTH}} | "
        )
        continuation = (
            f"{'':<{_TIME_WIDTH}} | {'':<{_SEVERITY_WIDTH}} | "
            f"{'':<{_SOURCE_WIDTH}} | "
        )
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
                ) or [""]
            )
        ]
        first = prefix + wrapped[0]
        return tuple([first, *(continuation + row for row in wrapped[1:])])

    @staticmethod
    def _console_width() -> int:
        """Return the AIBrain console width with its safety margin and minimum."""
        width = shutil.get_terminal_size((_DEFAULT_CONSOLE_WIDTH, 24)).columns
        return max(width - _RIGHT_EDGE_MARGIN, _MIN_CONSOLE_WIDTH)

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
            indentation = raw_line[:len(raw_line) - len(raw_line.lstrip())]
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
                rows.extend(
                    f"Item {index}: {cls._console_value(item)}"
                    for index, item in enumerate(parsed, start=1)
                )
                return tuple(rows)
        return (cls._console_message(safe_message),)

    @classmethod
    def _console_fields(cls, fields: Mapping[str, object]) -> tuple[str, ...]:
        """Render structured fields as readable labels instead of JSON fragments."""
        rows: list[str] = []
        for key, value in sorted(fields.items()):
            label = key.replace("_", " ").capitalize()
            rendered = cls._console_value(value).splitlines() or ["none"]
            rows.append(f"{label}: {rendered[0]}")
            rows.extend(f"  {line}" for line in rendered[1:])
        return tuple(rows)

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

    def event(self, level: str, message: str, **fields: int | float | str) -> None:
        """Dispatch one typed, redacted event to configured console and file sinks."""
        normalized = _normalize_level(level)
        minimum_level = _normalize_level(self.settings.level)
        if _LEVEL_ORDER[normalized] < _LEVEL_ORDER[minimum_level]:
            return
        safe_message = redact_text(message)
        safe_fields = redact_mapping(fields)
        source = str(safe_fields.pop("source", "logicytics.module"))
        file_lines = list(self._message_lines(safe_message))
        file_lines.extend(self._console_fields(safe_fields))
        rendered_message = "\n".join(file_lines)
        console_lines = list(self._message_lines(safe_message))
        console_lines.extend(self._console_fields(safe_fields))
        console_message = "\n".join(console_lines)
        rows = self._rows(normalized, source, rendered_message)
        with self._lock:
            if self.settings.file_enabled:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write("\n".join(rows) + "\n")
                self._truncate_file()
            if self.settings.console_enabled:
                marker, marker_color, text_color = _LEVEL_PRESENTATION[normalized]
                if not self._supports_unicode():
                    marker = {"\u25cf": "*", "\u00d7": "X", "\u00b7": "."}.get(marker, marker)
                console_rows = self._console_rows(marker, console_message)
                if self.settings.color_enabled and self.console.isatty():
                    marker_prefix = f"  {marker} "
                    colored_rows = (
                        f"{marker_color}{_BOLD}{marker_prefix}{_RESET}"
                        f"{text_color}{_BOLD}{console_rows[0][len(marker_prefix):]}"
                    )
                    if len(console_rows) > 1:
                        colored_rows += "\n" + "\n".join(
                            f"{text_color}{_BOLD}{row}" for row in console_rows[1:]
                        )
                    self.console.write(f"{colored_rows}{_RESET}\n")
                else:
                    self.console.write("\n".join(console_rows) + "\n")
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

    def box(self, title: str, lines: Iterable[str]) -> None:
        """Render console-only output as plain redacted lines."""
        with self._lock:
            if self.settings.console_enabled:
                self.render_section(self.console, title, lines)

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

    def error(self, message: str) -> None:
        """Render one concise argument error and its usage without raw argparse output."""
        usage = self.format_usage().strip()
        if usage.lower().startswith("usage:"):
            usage = usage[len("usage:"):].strip()
        ApplicationLogger.render_section(
            sys.stderr,
            "Command-line error",
            (f"Error: {message}", f"Usage: {usage}"),
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
        if (
                logger is None
                or logger.settings != settings
                or console is not None
                or (console is None and logger.console is not sys.stderr)
        ):
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

    def event(self, level: str, message: str, **fields: int | float | str) -> None:
        """Write a timestamped, structured event without relying on global handlers."""
        normalized = _normalize_level(level)
        payload: dict[str, object] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "level": normalized.lower(),
            "message": redact_text(message),
            "run_id": self.run_id,
        }
        if self.collector_id is not None:
            payload["collector_id"] = self.collector_id
        if fields:
            payload["fields"] = redact_mapping(fields)
        with self._event_lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")


def get_event_logger(
        path: Path, *, run_id: str, collector_id: str | None = None
) -> FileEventLogger:
    """Return the process-local singleton for one canonical engine or collector channel."""
    identity = (path.resolve(), run_id, collector_id)
    with _LOGGER_LOCK:
        logger = _EVENT_LOGGERS.get(identity)
        if logger is None:
            logger = FileEventLogger(identity[0], run_id=run_id, collector_id=collector_id)
            _EVENT_LOGGERS[identity] = logger
        return logger


def timed(
        logger: EventLogger, *,
        level: str = "info"
) -> Callable[[Callable[Parameters, Result]], Callable[Parameters, Result]]:
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
            logger.event(level, "function_finished", function=function.__qualname__,
                         duration_seconds=round(perf_counter() - started, 6))
            return value

        return wrapped

    return decorate


def raise_logged(logger: EventLogger, exception_type: type[Exception], message: str,
                 **fields: int | float | str) -> None:
    """Record a structured exception event, then raise the requested exception type."""
    logger.event("exception", message, exception_type=exception_type.__name__, **fields)
    raise exception_type(message)


def deprecated(
        logger: EventLogger, *,
        removal_version: str,
        reason: str,
        include_stack: bool = False
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
