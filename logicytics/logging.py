"""Run-scoped structured logging without global mutable logging configuration."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar

from logicytics.contracts import EventLogger

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")


class FileEventLogger(EventLogger):
    """Append JSONL events to a single run- or collector-owned log file."""

    def __init__(self, path: Path, *, run_id: str, collector_id: str | None = None) -> None:
        self.path = path
        self.run_id = run_id
        self.collector_id = collector_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, level: str, message: str, **fields: int | float | str) -> None:
        """Write a timestamped, structured event without relying on global handlers."""
        payload: dict[str, object] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "level": level.lower(),
            "message": message,
            "run_id": self.run_id,
        }
        if self.collector_id is not None:
            payload["collector_id"] = self.collector_id
        if fields:
            payload["fields"] = fields
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def timed(logger: EventLogger, *, level: str = "info") -> Callable[
    [Callable[Parameters, Result]], Callable[Parameters, Result]]:
    """Decorate a function so structured start, finish, error, and duration events are written."""

    def decorate(function: Callable[Parameters, Result]) -> Callable[Parameters, Result]:
        def wrapped(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
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


def deprecated(logger: EventLogger, *, removal_version: str, reason: str, include_stack: bool = False) -> Callable[
    [Callable[Parameters, Result]], Callable[Parameters, Result]]:
    """Decorate a function so each invocation emits a structured deprecation warning."""

    def decorate(function: Callable[Parameters, Result]) -> Callable[Parameters, Result]:
        def wrapped(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
            fields: dict[str, str] = {"function": function.__qualname__, "removal_version": removal_version,
                                      "reason": reason}
            if include_stack:
                fields["stack"] = "".join(traceback.format_stack(limit=8))
            logger.event("warning", "function_deprecated", **fields)
            return function(*args, **kwargs)

        return wrapped

    return decorate
