"""Typed local configuration for v4 runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from logicytics.errors import PlanError

SCHEMA_VERSION = 4


def _positive_integer(value: object, *, minimum: int, maximum: int) -> bool:
    """Whether a JSON value is a bounded integer rather than a boolean or coercion."""
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _bounded_number(value: object, *, minimum: float, maximum: float) -> bool:
    """Whether a JSON value is a bounded finite numeric setting."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and minimum <= value <= maximum


def _validate_collector_settings(settings: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate documented bounded settings before any worker receives them."""
    bandwidth = settings.get("core.network.bandwidth_sample", {})
    for key, maximum in (("sample_count", 10), ("interval_seconds", 5)):
        if key in bandwidth and not _positive_integer(bandwidth[key], minimum=1, maximum=maximum):
            raise PlanError(f"core.network.bandwidth_sample.{key} must be an integer from 1 to {maximum}")

    capture = settings.get("core.packet.packet_capture", {})
    if "packet_count" in capture and not _positive_integer(capture["packet_count"], minimum=1, maximum=10_000):
        raise PlanError("core.packet.packet_capture.packet_count must be an integer from 1 to 10000")
    if "timeout_seconds" in capture and not _bounded_number(capture["timeout_seconds"], minimum=1, maximum=60):
        raise PlanError("core.packet.packet_capture.timeout_seconds must be a number from 1 to 60")
    if "retry_window_seconds" in capture and not _bounded_number(
            capture["retry_window_seconds"], minimum=0, maximum=60):
        raise PlanError("core.packet.packet_capture.retry_window_seconds must be a number from 0 to 60")
    if "interface" in capture and (
            not isinstance(capture["interface"], str) or not capture["interface"].strip()):
        raise PlanError("core.packet.packet_capture.interface must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Engine-wide limits that apply before a collector is started."""

    output_root: Path
    default_max_workers: int = 4
    maximum_workers: int = 16
    package_completed_runs: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated settings loaded from an optional JSON configuration file."""

    schema_version: int
    runtime: RuntimeSettings
    collector_settings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def settings_for(self, collector_id: str) -> Mapping[str, Any]:
        """Return the isolated settings declared for one collector."""
        return self.collector_settings.get(collector_id, {})

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, non-secret configuration snapshot."""
        data = asdict(self)
        data["runtime"]["output_root"] = str(self.runtime.output_root)
        return data


def default_config(project_root: Path) -> AppConfig:
    """Create safe defaults rooted at the checked-out project."""
    return AppConfig(schema_version=SCHEMA_VERSION,
                     runtime=RuntimeSettings(output_root=project_root / "ACCESS" / "RUNS"))


def load_config(project_root: Path, config_path: Path | None = None) -> AppConfig:
    """Load and validate `logicytics.json`, or use the documented defaults."""
    path = config_path or project_root / "logicytics.json"
    if not path.exists():
        return default_config(project_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"invalid configuration file {path}: {error}") from error
    if not isinstance(raw, dict):
        raise PlanError("configuration root must be a JSON object")
    schema_version = raw.get("schema_version", SCHEMA_VERSION)
    if not isinstance(schema_version, int):
        raise PlanError("configuration schema_version must be an integer")
    if schema_version != SCHEMA_VERSION:
        raise PlanError(f"unsupported configuration schema_version {schema_version}; expected {SCHEMA_VERSION}")

    runtime_raw = raw.get("runtime", {})
    if not isinstance(runtime_raw, dict):
        raise PlanError("runtime configuration must be an object")
    output_root = Path(runtime_raw.get("output_root", project_root / "ACCESS" / "RUNS"))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    default_workers = runtime_raw.get("default_max_workers", 4)
    maximum_workers = runtime_raw.get("maximum_workers", 16)
    if not isinstance(default_workers, int) or not isinstance(maximum_workers, int):
        raise PlanError("worker limits must be integers")
    if not 1 <= default_workers <= maximum_workers <= 64:
        raise PlanError("worker limits must satisfy 1 <= default <= maximum <= 64")

    collector_settings = raw.get("collectors", {})
    if not isinstance(collector_settings, dict) or not all(
            isinstance(key, str) and isinstance(value, dict)
            for key, value in collector_settings.items()
    ):
        raise PlanError("collectors configuration must map collector IDs to objects")
    _validate_collector_settings(collector_settings)
    return AppConfig(
        schema_version=schema_version,
        runtime=RuntimeSettings(
            output_root=output_root,
            default_max_workers=default_workers,
            maximum_workers=maximum_workers,
            package_completed_runs=bool(runtime_raw.get("package_completed_runs", True)),
        ),
        collector_settings=collector_settings,
    )
