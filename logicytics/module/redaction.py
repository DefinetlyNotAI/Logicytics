"""Centralized secret redaction for diagnostic metadata and structured logs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

_SECRET_KEY_SUFFIXES = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "apikey",
    "accesskey",
    "privatekey",
    "recoverykey",
    "sessionkey",
)
_BEARER_PATTERN = re.compile(r"(?i)(\bauthorization\s*[:=]\s*bearer\s+)[^\s,;'\"]+")
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b[\w.-]*(?:password|passwd|pwd|secret|token|authorization|cookie|"
    r"credentials?|api[_-]?key|access[_-]?key|private[_-]?key|recovery[_-]?key|"
    r"session[_-]?key)\s*[:=]\s*['\"]?)[^\s,;'\"]+"
)
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----[\s\S]*?-----END [^-\r\n]*PRIVATE KEY-----",
    re.IGNORECASE,
)


def _is_secret_key(key: str) -> bool:
    """Recognize secret-valued fields while preserving dotted collector identities."""
    if key.startswith(("core.", "plugin.")):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(normalized.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES)


def redact_text(value: str) -> str:
    """Remove common inline assignments, bearer credentials, and private-key blocks."""
    value = _PRIVATE_KEY_PATTERN.sub(REDACTED, value)
    value = _BEARER_PATTERN.sub(lambda match: f"{match.group(1)}{REDACTED}", value)
    return _ASSIGNMENT_PATTERN.sub(lambda match: f"{match.group(1)}{REDACTED}", value)


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively sanitized copy without changing worker-owned settings."""
    return {
        key: REDACTED if _is_secret_key(key) else _redact_value(value)
        for key, value in values.items()
    }
