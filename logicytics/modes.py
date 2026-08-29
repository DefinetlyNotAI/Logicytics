"""Typed user-facing execution modes and legacy CLI compatibility aliases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


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
    non_python_only: bool = False
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
        "non-python",
        "Run only declared PowerShell, batch, and executable MODS payloads.",
        "standard",
        enable_mods=True,
        non_python_only=True,
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
    "nopy": "non-python",
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


def mode_matrix() -> list[dict[str, object]]:
    """Return a stable JSON-ready mode inclusion and scheduling matrix."""
    aliases_by_mode = {
        name: sorted(
            f"--{flag.removesuffix('_mode').replace('_', '-')}"
            for flag, target in LEGACY_MODE_ALIASES.items()
            if target == name
        )
        for name in EXECUTION_MODES
    }
    return [
        {
            **asdict(mode),
            "strategy": mode.strategy.value,
            "legacy_aliases": aliases_by_mode[mode.name],
        }
        for mode in EXECUTION_MODES.values()
    ]
