"""Deterministic conversion of validated collectors into a run plan."""

from __future__ import annotations

import sys
import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from types import MappingProxyType

from logicytics.contracts import Capability, CollectorKind, RunRequest
from logicytics.discovery import CollectorCandidate, PreflightReport
from logicytics.environment import inspect_environment
from logicytics.errors import PlanError, PreflightError

BUILTIN_PROFILES = MappingProxyType({
    "minimal": "Essential local system, memory, and storage inventory only.",
    "standard": "Shipped core collectors explicitly declaring standard membership.",
    "deep": "Extended declared inventory, subject to explicit capability approval.",
    "offline": "Declared local-only inventory with network and packet access prohibited.",
})
_OFFLINE_PROHIBITED_CAPABILITIES = frozenset({Capability.NETWORK, Capability.PACKET_CAPTURE})


@dataclass(frozen=True, slots=True)
class RunPlan:
    """The immutable collector order resolved before execution."""

    request: RunRequest
    collectors: tuple[CollectorCandidate, ...]
    fingerprint: str


def _fingerprint(request: RunRequest, collectors: tuple[CollectorCandidate, ...]) -> str:
    """Hash the immutable request and resolved metadata order into reproducibility evidence."""
    payload = {
        "request": asdict(request),
        "collectors": [candidate.metadata.to_dict() for candidate in collectors if candidate.metadata is not None],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_by_request(candidate: CollectorCandidate, request: RunRequest) -> bool:
    assert candidate.metadata is not None
    metadata = candidate.metadata
    if metadata.id in request.exclude:
        return False
    if metadata.id in request.include:
        return True
    if request.rerun_from is not None:
        return False
    if candidate.kind is CollectorKind.PLUGIN and not request.enable_plugins:
        return False
    if candidate.kind is CollectorKind.MOD:
        if not request.enable_mods:
            return False
        if request.non_python_only and candidate.execution_type == "mod_python":
            return False
        return True
    if request.non_python_only:
        return False
    return request.profile in metadata.default_profiles


def _topological_order(selected: dict[str, CollectorCandidate]) -> tuple[CollectorCandidate, ...]:
    ordered: list[CollectorCandidate] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(collector_id: str) -> None:
        if collector_id in visited:
            return
        if collector_id in visiting:
            raise PlanError(f"collector dependency cycle contains {collector_id}")
        candidate = selected.get(collector_id)
        if candidate is None:
            raise PlanError(f"selected collector depends on unavailable collector {collector_id}")
        assert candidate.metadata is not None
        visiting.add(collector_id)
        for dependency in sorted(candidate.metadata.dependencies):
            visit(dependency)
        visiting.remove(collector_id)
        visited.add(collector_id)
        ordered.append(candidate)

    for collector_id in sorted(selected):
        visit(collector_id)
    return tuple(ordered)


def build_plan(report: PreflightReport, request: RunRequest) -> RunPlan:
    """Fail closed for invalid selected work and resolve a dependency-safe plan."""
    if request.profile not in BUILTIN_PROFILES:
        supported = ", ".join(BUILTIN_PROFILES)
        raise PlanError(f"unknown collection profile {request.profile!r}; supported profiles: {supported}")
    if request.max_workers < 1:
        raise PlanError("max_workers must be positive")
    valid = {candidate.metadata.id: candidate for candidate in report.valid if candidate.metadata is not None}
    invalid_selected = [
        candidate
        for candidate in report.invalid
        if candidate.kind is CollectorKind.CORE
        or candidate.selection_id in request.include
        or (request.enable_plugins and candidate.kind is CollectorKind.PLUGIN)
        or (request.enable_mods and candidate.kind is CollectorKind.MOD)
    ]
    if invalid_selected:
        details = "; ".join(
            f"{candidate.path}: {', '.join(candidate.static_errors) or candidate.runtime_error or 'invalid'}"
            for candidate in invalid_selected
        )
        raise PreflightError(f"collector preflight failed: {details}")
    unknown_includes = sorted(set(request.include) - set(valid))
    if unknown_includes:
        raise PlanError(f"requested collectors are unavailable: {', '.join(unknown_includes)}")
    selected = {
        collector_id: candidate
        for collector_id, candidate in valid.items()
        if _selected_by_request(candidate, request)
    }
    pending_dependencies = list(selected)
    while pending_dependencies:
        collector_id = pending_dependencies.pop()
        candidate = selected[collector_id]
        assert candidate.metadata is not None
        for dependency_id in candidate.metadata.dependencies:
            if dependency_id in request.exclude:
                raise PlanError(f"selected collector {collector_id} depends on explicitly excluded {dependency_id}")
            dependency = valid.get(dependency_id)
            if dependency is None:
                raise PlanError(f"selected collector {collector_id} depends on unavailable collector {dependency_id}")
            assert dependency.metadata is not None
            if dependency.kind is CollectorKind.PLUGIN and not (
                    request.enable_plugins or dependency_id in request.include
            ):
                raise PlanError(f"plugin dependency {dependency_id} must be explicitly selected or plugins enabled")
            if dependency.kind is CollectorKind.MOD and not (
                    request.enable_mods or dependency_id in request.include
            ):
                raise PlanError(f"mod dependency {dependency_id} must be explicitly selected or mods enabled")
            if dependency.metadata.sensitive_data_categories and dependency_id not in request.include:
                raise PlanError(f"sensitive dependency {dependency_id} must be explicitly selected")
            if dependency_id not in selected:
                selected[dependency_id] = dependency
                pending_dependencies.append(dependency_id)
    for candidate in selected.values():
        assert candidate.metadata is not None
        if request.profile == "offline":
            prohibited = set(candidate.metadata.capabilities).intersection(_OFFLINE_PROHIBITED_CAPABILITIES)
            if prohibited:
                names = ", ".join(sorted(capability.value for capability in prohibited))
                raise PlanError(f"offline profile prohibits network-capable collector {candidate.metadata.id}: {names}")
        if sys.platform not in candidate.metadata.supported_platforms:
            raise PlanError(f"{candidate.metadata.id} does not support {sys.platform}")
        missing_capabilities = set(candidate.metadata.capabilities) - set(request.approved_capabilities)
        if missing_capabilities:
            required = ", ".join(sorted(capability.value for capability in missing_capabilities))
            raise PlanError(f"{candidate.metadata.id} requires unapproved capabilities: {required}")
    elevated_collectors = sorted(
        candidate.metadata.id
        for candidate in selected.values()
        if candidate.metadata is not None and Capability.ELEVATED_PRIVILEGES in candidate.metadata.capabilities
    )
    if elevated_collectors and inspect_environment().is_administrator is not True:
        raise PlanError(
            "selected collectors require an administrator account: "
            + ", ".join(elevated_collectors)
        )
    ordered = _topological_order(selected)
    return RunPlan(request=request, collectors=ordered, fingerprint=_fingerprint(request, ordered))
