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


class CapabilityPolicyError(PermissionError):
    """Raised when a collector's declared capability policy is violated."""

    def __init__(self, code: str, capability: str, operation: str, detail: str = "") -> None:
        """Create a stable, machine-searchable capability diagnostic."""
        self.code = code
        self.capability = capability
        self.operation = operation
        action = "requested blocked" if code == "CAPABILITY_BLOCKED" else "used undeclared"
        suffix = f"; {detail}" if detail else ""
        super().__init__(f"{code}: collector {action} {capability} capability; operation={operation}{suffix}")
