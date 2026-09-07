"""Run-scoped structured logging without global mutable logging configuration."""

from __future__ import annotations

import json
import sys
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from time import perf_counter, time
from typing import Callable, Iterable, ParamSpec, TextIO, TypeVar

from logicytics.contracts import EventLogger
from logicytics.module.configuration import LoggingSettings
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
_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INTERNAL": "\033[35m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "EXCEPTION": "\033[91m",
    "CRITICAL": "\033[97;41m",
}
_BOX_COLOR = "\033[90m"
_LOGGER_LOCK = RLock()
_APPLICATION_LOGGERS: dict[Path, "ApplicationLogger"] = {}
_EVENT_LOGGERS: dict[tuple[Path, str, str | None], "FileEventLogger"] = {}


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
                stream.seek(-self.settings.maximum_bytes // 2, 2)
                retained = stream.read()
            newline = retained.find(b"\n")
            retained = retained[newline + 1:] if newline >= 0 else retained
            marker = "\n".join(
                self._rows("WARNING", "logicytics.logging", "log truncated to configured maximum")
            ).encode("utf-8") + b"\n"
            self.path.write_bytes(marker + retained)

    @staticmethod
    def _rows(level: str, source: str, message: str) -> tuple[str, ...]:
        """Format aligned, deterministic human log rows for file inspection."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        source_column = source[:24]
        wrapped = textwrap.wrap(message, width=76, break_long_words=False, break_on_hyphens=False) or [""]
        first = f"{timestamp} | {level:<9} | {source_column:<24} | {wrapped[0]}"
        continuation = f"{'':8} | {'':9} | {'':24} | "
        return tuple([first, *(continuation + row for row in wrapped[1:])])

    def event(self, level: str, message: str, **fields: int | float | str) -> None:
        """Dispatch one typed, redacted event to configured console and file sinks."""
        normalized = level.upper()
        if normalized not in _LEVEL_ORDER:
            raise ValueError(f"unsupported log level: {level}")
        if _LEVEL_ORDER[normalized] < _LEVEL_ORDER[self.settings.level]:
            return
        safe_message = redact_text(message)
        safe_fields = redact_mapping(fields)
        source = str(safe_fields.pop("source", "logicytics.module"))
        suffix = "" if not safe_fields else " " + " ".join(
            f"{key}={json.dumps(value, ensure_ascii=True, sort_keys=True)}"
            for key, value in sorted(safe_fields.items())
        )
        rows = self._rows(normalized, source, safe_message + suffix)
        with self._lock:
            if self.settings.file_enabled:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write("\n".join(rows) + "\n")
                self._truncate_file()
            if self.settings.console_enabled:
                if self.settings.color_enabled and self.console.isatty():
                    self.console.write(f"{_LEVEL_COLORS[normalized]}{rows[0]}\033[0m\n")
                    for row in rows[1:]:
                        self.console.write(row + "\n")
                else:
                    self.console.write("\n".join(rows) + "\n")
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

    def box(self, title: str, lines: Iterable[str]) -> None:
        """Render non-log command output in a wrapped, restrained grey ASCII panel."""
        content: list[str] = []
        for line in lines:
            safe = redact_text(line)
            wrapped = textwrap.wrap(
                safe,
                width=96,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            for row in wrapped:
                content.extend(row[index:index + 96] for index in range(0, len(row), 96))
        width = max(len(title) + 4, *(len(line) + 2 for line in content), 4)
        border = "+" + "-" * width + "+"
        rows = [border, f"| {title[:width - 2]:<{width - 2}} |", border]
        rows.extend(f"| {line:<{width - 2}} |" for line in content)
        rows.append(border)
        with self._lock:
            if self.settings.console_enabled:
                if self.settings.color_enabled and self.console.isatty():
                    self.console.write(f"{_BOX_COLOR}{chr(10).join(rows)}\033[0m\n")
                else:
                    self.console.write("\n".join(rows) + "\n")
                self.console.flush()

    def dispatch(self, messages: Iterable[str]) -> None:
        """Parse and dispatch a batch of optional `LEVEL: message` rows."""
        for message in messages:
            level, separator, text = message.partition(":")
            normalized = level.strip().upper()
            if separator and normalized in _LEVEL_ORDER:
                self.event(normalized, text.strip())
            else:
                self.event("INFO", message.strip())


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
        if logger is None or logger.settings != settings or console is not None:
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
        payload: dict[str, object] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "level": level.lower(),
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


def get_event_logger(path: Path, *, run_id: str, collector_id: str | None = None) -> FileEventLogger:
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
                logger.event("error", "function_failed", function=function.__qualname__,
                             duration_seconds=round(perf_counter() - started, 6), error_type=type(error).__name__)
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
            fields: dict[str, str] = {"function": function.__qualname__, "removal_version": removal_version,
                                      "reason": reason}
            if include_stack:
                fields["stack"] = "".join(traceback.format_stack(limit=8))
            logger.event("warning", "function_deprecated", **fields)
            return function(*args, **kwargs)

        return wrapped

    return decorate
