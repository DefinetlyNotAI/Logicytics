"""Logicytics v4 public contracts and lazily loaded application API."""

import importlib
import sys
from typing import TYPE_CHECKING

from logicytics.module.contracts import (
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

if TYPE_CHECKING:
    from logicytics.module.api import (
        CollectorFailureSnapshot,
        CollectorSnapshot,
        RunSnapshot,
        load_configuration,
        open_artifact,
        plan_run,
        query_run,
        read_artifact,
        run_collection,
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

# Keep the former import paths working for shipped collectors and independently
# authored plugins while the implementation lives in its domain subpackages.
for _legacy_name, _module_name in {
    "contracts": "logicytics.module.contracts",
    "platform_adapters": "logicytics.module.platform_adapters",
}.items():
    sys.modules.setdefault(f"{__name__}.{_legacy_name}", importlib.import_module(_module_name))


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
