"""Logicytics v4 public engine API."""
from logicytics.api import run_collection, read_artifact, query_run, open_artifact, plan_run, load_configuration, \
    RunSnapshot, CollectorSnapshot, CollectorFailureSnapshot
from logicytics.contracts import (
    CONTRACT_VERSION,
    Capability,
    Collector,
    CollectionEstimate,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    EvidenceKind,
    EstimatedCost,
    NetworkAccess,
    OutputPolicy,
    PluginCollector,
    PostRunAction,
    PrivilegeLevel,
    ResourceClass,
    RunRequest,
    RunStatus,
    Specialty,
    ValidationResult,
)

_APPLICATION_EXPORTS = frozenset({
    "CollectorFailureSnapshot", "CollectorSnapshot", "RunSnapshot", "load_configuration", "open_artifact", "plan_run",
    "query_run", "read_artifact",
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
    "CollectorFailureSnapshot",
    "CollectorSnapshot",
    "CoreCollector",
    "EvidenceKind",
    "EstimatedCost",
    "NetworkAccess",
    "OutputPolicy",
    "PluginCollector",
    "PostRunAction",
    "PrivilegeLevel",
    "ResourceClass",
    "RunRequest",
    "RunSnapshot",
    "RunStatus",
    "Specialty",
    "ValidationResult",
    "load_configuration",
    "open_artifact",
    "plan_run",
    "query_run",
    "read_artifact",
    "run_collection"
]
