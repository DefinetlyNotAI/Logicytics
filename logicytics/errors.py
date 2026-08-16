"""Domain exceptions raised by the v4 engine."""

from __future__ import annotations


class LogicyticsError(Exception):
    """Base error for expected application-level failures."""


class PreflightError(LogicyticsError):
    """Raised when selected collectors fail strict preflight validation."""


class PlanError(LogicyticsError):
    """Raised when a requested run cannot be converted into a valid plan."""


class ArtifactError(LogicyticsError):
    """Raised when an artifact violates workspace or output rules."""
