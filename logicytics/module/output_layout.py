"""Canonical global output directories shared by CLI maintenance and run actions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

_MINIMUM_FINGERPRINT_LENGTH = 8


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


def allocate_run_directory(runs: Path, run_id: str) -> Path:
    """Return the shortest unoccupied output directory for one run fingerprint."""
    fingerprint = run_fingerprint(run_id)
    for length in range(_MINIMUM_FINGERPRINT_LENGTH, len(fingerprint) + 1):
        candidate = runs / fingerprint[:length]
        if not candidate.exists():
            return candidate
    raise FileExistsError("no unique output fingerprint prefix is available")


def allocate_output_run_directory(layout: OutputLayout, run_id: str) -> Path:
    """Reserve the shortest unique fingerprint across every named output channel."""
    fingerprint = run_fingerprint(run_id)
    for length in range(_MINIMUM_FINGERPRINT_LENGTH, len(fingerprint) + 1):
        token = fingerprint[:length]
        occupied = (
            layout.runs / token,
            layout.packages / f"{token}.zip",
            layout.packages / f"mods-{token}.zip",
            layout.hashes / f"{token}.zip.sha256",
            layout.hashes / f"mods-{token}.zip.sha256",
            layout.performance_logs / f"{token}.log",
        )
        if not any(path.exists() for path in occupied):
            return layout.runs / token
    raise FileExistsError("no unique output fingerprint prefix is available")


def locate_run_directory(runs: Path, run_id: str) -> Path:
    """Locate a run directory by its full fingerprint while supporting prefix storage."""
    fingerprint = run_fingerprint(run_id)
    matches = sorted(
        (
            path
            for path in runs.iterdir()
            if path.is_dir()
            and _MINIMUM_FINGERPRINT_LENGTH <= len(path.name) <= len(fingerprint)
            and fingerprint.startswith(path.name)
        ),
        key=lambda path: len(path.name),
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"no output directory exists for run {run_id}")
    return matches[0]


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
        hashes=data / "hashes",
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
