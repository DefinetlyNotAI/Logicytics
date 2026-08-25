"""Logicytics v4 public engine API."""

from logicytics.contracts import (
    CONTRACT_VERSION,
    Capability,
    Collector,
    CollectionEstimate,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    PluginCollector,
    ResourceClass,
    RunRequest,
    RunStatus,
    Specialty,
    ValidationResult,
)

_APPLICATION_EXPORTS = frozenset({
    "CollectorSnapshot", "RunSnapshot", "load_configuration", "plan_run", "query_run", "read_artifact",
    "run_collection",
})


def __getattr__(name: str):
    """Load application services only when callers explicitly request them."""
    if name in _APPLICATION_EXPORTS:
        from logicytics import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CONTRACT_VERSION",
    "Capability",
    "Collector",
    "CollectionEstimate",
    "CollectorMetadata",
    "CollectorResult",
    "CollectorSnapshot",
    "CoreCollector",
    "PluginCollector",
    "ResourceClass",
    "RunRequest",
    "RunSnapshot",
    "RunStatus",
    "Specialty",
    "ValidationResult",
    "load_configuration",
    "plan_run",
    "query_run",
    "read_artifact",
    "run_collection",
]
