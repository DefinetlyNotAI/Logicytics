"""Typed user-facing execution modes and legacy CLI compatibility aliases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, TypedDict, Mapping

from logicytics.contracts import CollectorKind
from logicytics.module.discovery import CollectorCandidate


class ModeMatrixCollector(TypedDict):
    """Serialized mode membership and validation details for one collector."""
    id: str
    kind: str
    valid: bool
    modes: list[str]
    manual_only: bool
    execution_type: str
    validation_errors: list[str]


class ModeMatrixMode(TypedDict):
    """Serialized execution mode definition and its selected collectors."""
    name: str
    description: str
    strategy: str
    legacy_aliases: list[str]
    collector_ids: list[str]


class ModeMatrix(TypedDict):
    """Versioned machine-readable matrix of modes and collector memberships."""
    schema_version: int
    modes: list[ModeMatrixMode]
    collectors: list[ModeMatrixCollector]


class ExecutionStrategy(str, Enum):
    """Worker scheduling policy owned by an execution mode."""

    CONFIGURED = "configured"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass(frozen=True, slots=True)
class ExecutionMode:
    """Complete profile, extension, and scheduling behavior for one named mode."""

    name: str
    description: str
    profile: str
    strategy: ExecutionStrategy = ExecutionStrategy.CONFIGURED
    enable_mods: bool = False
    performance_check: bool = False


_MODE_LIST = (
    ExecutionMode(
        "standard",
        "Standard built-in collection with deterministic sequential execution.",
        "standard",
        ExecutionStrategy.SEQUENTIAL,
    ),
    ExecutionMode(
        "balanced",
        "Standard built-in collection using the configured bounded worker pool.",
        "standard",
        ExecutionStrategy.PARALLEL,
    ),
    ExecutionMode("quick", "Fast essential local inventory.", "minimal"),
    ExecutionMode("thorough", "Extended and potentially slower local inventory.", "deep"),
    ExecutionMode("offline", "Local-only collection with network access forbidden.", "offline"),
    ExecutionMode(
        "extensions",
        "Standard collection followed by explicitly declared MODS extensions.",
        "standard",
        enable_mods=True,
    ),
    ExecutionMode(
        "performance",
        "Sequential standard collection with per-collector duration reporting.",
        "standard",
        ExecutionStrategy.SEQUENTIAL,
        performance_check=True,
    ),
)

EXECUTION_MODES: Mapping[str, ExecutionMode] = MappingProxyType(
    {mode.name: mode for mode in _MODE_LIST}
)
LEGACY_MODE_ALIASES: Mapping[str, str] = MappingProxyType({
    "default_mode": "standard",
    "threaded": "balanced",
    "minimal": "quick",
    "depth": "thorough",
    "modded": "extensions",
    "performance_check": "performance",
})


def resolve_execution_mode(
        selected: str | None,
        legacy_flags: Mapping[str, bool],
) -> ExecutionMode | None:
    """Resolve a user-facing name or exactly one parser-exclusive legacy alias."""
    aliases = [name for flag, name in LEGACY_MODE_ALIASES.items() if legacy_flags.get(flag, False)]
    if selected is not None and aliases:
        raise ValueError("--mode cannot be combined with a legacy mode alias")
    if len(aliases) > 1:
        raise ValueError("legacy collection mode aliases are mutually exclusive")
    name = selected or (aliases[0] if aliases else None)
    if name is None:
        return None
    try:
        return EXECUTION_MODES[name]
    except KeyError as error:
        raise ValueError(f"unknown execution mode: {name}") from error


def _candidate_modes(candidate: CollectorCandidate) -> tuple[str, ...]:
    """Return every named mode that selects one validated collector by default."""
    if candidate.metadata is None:
        return ()
    metadata = candidate.metadata
    selected: list[str] = []
    for mode in EXECUTION_MODES.values():
        if candidate.kind is CollectorKind.MOD:
            enabled = mode.enable_mods
        elif candidate.kind is CollectorKind.PLUGIN:
            enabled = False
        else:
            enabled = mode.profile in metadata.default_profiles
        if enabled:
            selected.append(mode.name)
    return tuple(selected)


def mode_matrix(
        candidates: Iterable[CollectorCandidate] = (),
) -> ModeMatrix:
    """Return the versioned mode definitions and complete collector inclusion matrix."""
    aliases_by_mode = {
        name: sorted(
            f"--{flag.removesuffix('_mode').replace('_', '-')}"
            for flag, target in LEGACY_MODE_ALIASES.items()
            if target == name
        )
        for name in EXECUTION_MODES
    }

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.metadata.id
            if candidate.metadata is not None
            else candidate.selection_id,
            str(candidate.path),
        ),
    )

    collector_rows: list[ModeMatrixCollector] = []
    memberships: dict[str, list[str]] = {
        name: []
        for name in EXECUTION_MODES
    }

    for candidate in ordered_candidates:
        collector_id = (
            candidate.metadata.id
            if candidate.metadata is not None
            else candidate.selection_id
        )
        assigned_modes = _candidate_modes(candidate)

        for mode_name in assigned_modes:
            memberships[mode_name].append(collector_id)

        errors = [*candidate.static_errors]
        if candidate.runtime_error:
            errors.append(candidate.runtime_error)

        collector_rows.append(
            {
                "id": collector_id,
                "kind": candidate.kind.value,
                "valid": candidate.valid,
                "modes": list(assigned_modes),
                "manual_only": candidate.valid and not assigned_modes,
                "execution_type": candidate.execution_type,
                "validation_errors": errors,
            }
        )

    modes: list[ModeMatrixMode] = [
        {
            "name": mode.name,
            "description": mode.description,
            "strategy": mode.strategy.value,
            "legacy_aliases": aliases_by_mode[mode.name],
            "collector_ids": memberships[mode.name],
        }
        for mode in EXECUTION_MODES.values()
    ]

    return {
        "schema_version": 1,
        "modes": modes,
        "collectors": collector_rows,
    }


def render_mode_matrix_markdown(matrix: Mapping[str, object]) -> str:
    """Render the collector side of a mode matrix as deterministic v4 documentation."""
    rows = [
        "# Logicytics v4 collector mode matrix",
        "",
        "This file is generated from strict collector preflight metadata. Use",
        "`python -m logicytics --modes` for the complete machine-readable contract.",
        "",
        "| Collector | Kind | Valid | Included modes | Selection |",
        "| --- | --- | --- | --- | --- |",
    ]

    collectors = matrix.get("collectors", [])
    if not isinstance(collectors, list):
        collectors = []

    for item in collectors:
        if not isinstance(item, Mapping):
            continue

        collector = dict(item)

        raw_modes = collector.get("modes", [])
        modes_iterable = raw_modes if isinstance(raw_modes, (list, tuple)) else []
        modes = ", ".join(f"`{name}`" for name in modes_iterable) or "none"

        valid = collector.get("valid") is True
        manual_only = collector.get("manual_only") is True

        if not valid:
            selection = "quarantined"
        elif manual_only:
            selection = "explicit include only"
        else:
            selection = "mode selected"

        rows.append(
            f"| `{collector.get('id', '')}` | `{collector.get('kind', '')}` | "
            f"{'yes' if valid else 'no'} | {modes} | {selection} |"
        )

    rows.append("")
    return "\n".join(rows)
