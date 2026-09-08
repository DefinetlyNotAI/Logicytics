"""Logicytics v4 public contracts and lazily loaded application API."""

import importlib

from logicytics.contracts import (
    CONTRACT_VERSION,
    Capability,
    CollectionEstimate,
    Collector,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    EstimatedCost,
    EvidenceKind,
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

_APPLICATION_EXPORTS = frozenset(
    {
        "CollectorFailureSnapshot",
        "CollectorSnapshot",
        "RunSnapshot",
        "load_configuration",
        "open_artifact",
        "plan_run",
        "query_run",
        "read_artifact",
        "run_collection",
    }
)


def __getattr__(name: str):
    """Load application services and global host infrastructure on explicit access."""
    if name in _APPLICATION_EXPORTS:
        from logicytics.module import api

        return getattr(api, name)
    if name == "ctypes_collector":
        return importlib.import_module("logicytics.global.ctypes_collector")
    global_infrastructure = importlib.import_module("logicytics.global.ctypes_collector")
    if hasattr(global_infrastructure, name):
        return getattr(global_infrastructure, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CONTRACT_VERSION",
    "Capability",
    "CollectionEstimate",
    "Collector",
    "CollectorFailureSnapshot",
    "CollectorMetadata",
    "CollectorResult",
    "CollectorSnapshot",
    "CoreCollector",
    "EstimatedCost",
    "EvidenceKind",
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
    "run_collection",
]
