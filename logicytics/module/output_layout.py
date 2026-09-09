"""Canonical global output directories shared by CLI maintenance and run actions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Resolved global output locations outside individual run directories."""

    data: Path
    runs: Path
    logs: Path
    debug_logs: Path
    performance_logs: Path
    packages: Path
    hashes: Path
    application_log: Path


def run_fingerprint(run_id: str) -> str:
    """Return the stable opaque storage fingerprint for one canonical run ID."""
    return sha256(run_id.encode("ascii")).hexdigest()


def output_layout(output_root: Path) -> OutputLayout:
    """Derive the stable output tree from the configured run-data root."""
    data = output_root.resolve()
    output = data.parent
    logs = output / "logs"
    return OutputLayout(
        data=data,
        runs=data / "run",
        logs=logs,
        debug_logs=logs / "debug",
        performance_logs=logs / "performance",
        packages=data / "zip",
        hashes=data / "zip" / "hashes",
        application_log=logs / "Logicytics.log",
    )


def ensure_output_layout(output_root: Path) -> OutputLayout:
    """Create the complete stable output tree for a state-changing application action."""
    layout = output_layout(output_root)
    for directory in (
        layout.data,
        layout.runs,
        layout.logs,
        layout.debug_logs,
        layout.performance_logs,
        layout.packages,
        layout.hashes,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout
