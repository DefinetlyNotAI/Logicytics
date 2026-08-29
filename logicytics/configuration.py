"""Typed local configuration for v4 runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from logicytics.errors import PlanError
from logicytics.redaction import redact_mapping

SCHEMA_VERSION = 4
DEFAULT_MAXIMUM_RUN_OUTPUT_BYTES = 4 * 1024 * 1024 * 1024
MAXIMUM_RUN_OUTPUT_BYTES = 64 * 1024 * 1024 * 1024
_COLLECTOR_ID = re.compile(
    r"^(?:core\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*|(?:plugin|mod)\.[a-z][a-z0-9_]*)$"
)
_SETTING_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_ROOT_FIELDS = frozenset({
    "schema_version", "runtime", "interaction", "maintenance", "logging", "collectors",
})
_RUNTIME_FIELDS = frozenset({
    "output_root", "default_max_workers", "maximum_workers", "package_completed_runs", "maximum_run_output_bytes",
})
_INTERACTION_FIELDS = frozenset({"history_enabled", "similarity_threshold", "model_name", "model_debug"})
_MAINTENANCE_FIELDS = frozenset({
    "remote_manifest_url", "remote_manifest_sha256", "local_manifest_path",
    "minimum_python", "recommended_python",
})
_LOGGING_FIELDS = frozenset({
    "level", "console_enabled", "color_enabled", "file_enabled", "maximum_bytes",
    "delete_previous", "retention_days",
})
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "INTERNAL", "EXCEPTION"})
_LEGACY_ROOT_FIELDS = _ROOT_FIELDS.union({
    "collector_settings", "workers", "worker_count", "max_workers", "output_root",
    "package_completed_runs", "maximum_run_output_bytes",
})
_LEGACY_RUNTIME_ALIASES = MappingProxyType({
    "workers": "default_max_workers",
    "worker_count": "default_max_workers",
    "max_workers": "maximum_workers",
})


@dataclass(frozen=True, slots=True)
class CollectorSettingRule:
    """One immutable collector configuration field and its safe value boundary."""

    kind: str
    minimum: int | float = 0
    maximum: int | float = 0


_COLLECTOR_SETTING_SCHEMAS = MappingProxyType({
    "core.network.bandwidth_sample": {
        "sample_count": CollectorSettingRule("integer", 1, 10),
        "interval_seconds": CollectorSettingRule("integer", 1, 5),
    },
    "core.packet.packet_capture": {
        "packet_count": CollectorSettingRule("integer", 1, 10_000),
        "timeout_seconds": CollectorSettingRule("number", 1, 60),
        "retry_window_seconds": CollectorSettingRule("number", 0, 60),
        "interface": CollectorSettingRule("text"),
    },
    "core.filesystem.system_drive_tree": {
        "max_entries": CollectorSettingRule("integer", 1, 50_000),
        "max_depth": CollectorSettingRule("integer", 1, 32),
    },
    "core.filesystem.system_drive_listing": {
        "max_entries": CollectorSettingRule("integer", 1, 50_000),
        "max_depth": CollectorSettingRule("integer", 1, 32),
        "workers": CollectorSettingRule("integer", 1, 8),
    },
    "core.filesystem.sensitive_file_inventory": {
        "root": CollectorSettingRule("absolute_path"),
        "max_directories": CollectorSettingRule("integer", 1, 50_000),
        "max_matches": CollectorSettingRule("integer", 1, 5_000),
    },
    "core.process.memory_map": {
        "max_regions": CollectorSettingRule("integer", 1, 100_000),
        "output_limit_bytes": CollectorSettingRule("integer", 1_024, 64 * 1024 * 1024),
        "disk_safety_margin_bytes": CollectorSettingRule("integer", 0, MAXIMUM_RUN_OUTPUT_BYTES),
        "dump_directory": CollectorSettingRule("workspace_path"),
    },
})


def _positive_integer(value: object, *, minimum: int, maximum: int) -> bool:
    """Whether a JSON value is a bounded integer rather than a boolean or coercion."""
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _bounded_number(value: object, *, minimum: float, maximum: float) -> bool:
    """Whether a JSON value is a bounded finite numeric setting."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and minimum <= value <= maximum


def _validate_collector_settings(settings: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate collector identities and all documented fields before worker launch."""
    for collector_id, values in settings.items():
        if not isinstance(collector_id, str) or not _COLLECTOR_ID.fullmatch(collector_id):
            raise PlanError(f"collectors configuration contains an invalid collector ID: {collector_id!r}")
        for key in values:
            if not isinstance(key, str) or not _SETTING_NAME.fullmatch(key):
                raise PlanError(f"{collector_id} contains an invalid setting name: {key!r}")
        schema = _COLLECTOR_SETTING_SCHEMAS.get(collector_id)
        if schema is None:
            continue
        unknown = sorted(set(values) - set(schema))
        if unknown:
            raise PlanError(f"{collector_id} contains unsupported settings: {', '.join(unknown)}")
        for key, value in values.items():
            rule = schema[key]
            label = f"{collector_id}.{key}"
            if rule.kind == "integer":
                if not _positive_integer(value, minimum=int(rule.minimum), maximum=int(rule.maximum)):
                    raise PlanError(f"{label} must be an integer from {rule.minimum} to {rule.maximum}")
            elif rule.kind == "number":
                if not _bounded_number(value, minimum=rule.minimum, maximum=rule.maximum):
                    raise PlanError(f"{label} must be a number from {rule.minimum} to {rule.maximum}")
            elif not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise PlanError(f"{label} must be a non-empty path or string")
            elif rule.kind == "absolute_path" and not Path(value).is_absolute():
                raise PlanError(f"{label} must be an absolute filesystem path")
            elif rule.kind == "workspace_path":
                path = Path(value)
                if path.is_absolute() or path.drive or ".." in path.parts:
                    raise PlanError(f"{label} must remain a relative collector-workspace path")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous duplicate configuration keys instead of silently replacing them."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject non-finite nonstandard JSON values before settings can reach workers."""
    raise ValueError(f"non-finite configuration number {value!r}")


def _finite_json_number(value: str) -> float:
    """Reject syntactically valid decimals that overflow the local float range."""
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"non-finite configuration number {value!r}")
    return result


def _assign_migrated_runtime(runtime: dict[str, Any], target: str, value: Any, source: str) -> None:
    """Refuse legacy aliases that would silently override another runtime setting."""
    if target in runtime:
        raise PlanError(f"legacy configuration contains conflicting settings for {target}: {source}")
    runtime[target] = value


def _migrate_v3_configuration(raw: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Upgrade the explicitly supported v3 shape without writing or activating collectors."""
    unknown = sorted(set(raw) - _LEGACY_ROOT_FIELDS)
    if unknown:
        raise PlanError(f"legacy configuration contains unsupported root settings: {', '.join(unknown)}")
    runtime_value = raw.get("runtime", {})
    if not isinstance(runtime_value, dict):
        raise PlanError("legacy runtime configuration must be an object")
    runtime: dict[str, Any] = {}
    for key, value in runtime_value.items():
        _assign_migrated_runtime(runtime, _LEGACY_RUNTIME_ALIASES.get(key, key), value, f"runtime.{key}")
    root_aliases = {
        "workers": "default_max_workers",
        "worker_count": "default_max_workers",
        "max_workers": "maximum_workers",
        "output_root": "output_root",
        "package_completed_runs": "package_completed_runs",
        "maximum_run_output_bytes": "maximum_run_output_bytes",
    }
    for source, target in root_aliases.items():
        if source in raw:
            _assign_migrated_runtime(runtime, target, raw[source], source)
    if "collectors" in raw and "collector_settings" in raw:
        raise PlanError("legacy configuration contains conflicting collectors and collector_settings sections")
    collectors = raw.get("collectors", raw.get("collector_settings", {}))
    output_root = runtime.get("output_root")
    if isinstance(output_root, str):
        candidate = Path(output_root)
        relative_parts = tuple(part.casefold() for part in candidate.parts)
        if relative_parts == ("access", "runs") or candidate == project_root / "ACCESS" / "RUNS":
            runtime["output_root"] = str(project_root / "output" / "data")
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime": runtime,
        "interaction": raw.get("interaction", {}),
        "maintenance": raw.get("maintenance", {}),
        "logging": raw.get("logging", {}),
        "collectors": collectors,
    }


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Engine-wide limits that apply before a collector is started."""

    output_root: Path
    default_max_workers: int = 4
    maximum_workers: int = 16
    package_completed_runs: bool = True
    maximum_run_output_bytes: int = DEFAULT_MAXIMUM_RUN_OUTPUT_BYTES


@dataclass(frozen=True, slots=True)
class InteractionSettings:
    """Local-only matching, diagnostics, and optional history policy."""

    history_enabled: bool = False
    similarity_threshold: float = 0.55
    model_name: str = "stdlib-sequence-matcher"
    model_debug: bool = False


@dataclass(frozen=True, slots=True)
class MaintenanceSettings:
    """Optional integrity sources and supported Python policy."""

    remote_manifest_url: str | None = None
    remote_manifest_sha256: str | None = None
    local_manifest_path: Path = Path("project.manifest.json")
    minimum_python: str = "3.11"
    recommended_python: str = "3.11"


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Process-wide human log and console presentation policy."""

    level: str = "INFO"
    console_enabled: bool = True
    color_enabled: bool = True
    file_enabled: bool = True
    maximum_bytes: int = 4 * 1024 * 1024
    delete_previous: bool = False
    retention_days: int = 30


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated settings loaded from an optional JSON configuration file."""

    schema_version: int
    runtime: RuntimeSettings
    interaction: InteractionSettings = field(default_factory=InteractionSettings)
    maintenance: MaintenanceSettings = field(default_factory=MaintenanceSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    collector_settings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    migrated_from_schema: int | None = None

    def settings_for(self, collector_id: str) -> Mapping[str, Any]:
        """Return the isolated settings declared for one collector."""
        return self.collector_settings.get(collector_id, {})

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, non-secret configuration snapshot."""
        data = asdict(self)
        data["runtime"]["output_root"] = str(self.runtime.output_root)
        data["maintenance"]["local_manifest_path"] = str(self.maintenance.local_manifest_path)
        return redact_mapping(data)

    def fingerprint(self) -> str:
        """Hash the complete validated configuration without persisting its secrets."""
        data = asdict(self)
        data["runtime"]["output_root"] = str(self.runtime.output_root.resolve())
        data["maintenance"]["local_manifest_path"] = str(self.maintenance.local_manifest_path)
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def default_config(project_root: Path) -> AppConfig:
    """Create safe defaults rooted at the checked-out project."""
    return AppConfig(schema_version=SCHEMA_VERSION,
                     runtime=RuntimeSettings(output_root=project_root / "output" / "data"))


def load_config(project_root: Path, config_path: Path | None = None) -> AppConfig:
    """Load and validate `logicytics.json`, or use the documented defaults."""
    path = config_path or project_root / "logicytics.json"
    if not path.exists():
        return default_config(project_root)
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_number,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise PlanError(f"invalid configuration file {path}: {error}") from error
    if not isinstance(raw, dict):
        raise PlanError("configuration root must be a JSON object")
    schema_version = raw.get("schema_version", SCHEMA_VERSION)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise PlanError("configuration schema_version must be an integer")
    migrated_from_schema: int | None = None
    if schema_version == 3:
        raw = _migrate_v3_configuration(raw, project_root)
        migrated_from_schema = schema_version
        schema_version = SCHEMA_VERSION
    if schema_version != SCHEMA_VERSION:
        raise PlanError(
            f"unsupported configuration schema_version {schema_version}; expected {SCHEMA_VERSION} "
            "or supported legacy schema 3"
        )
    unknown_root = sorted(set(raw) - _ROOT_FIELDS)
    if unknown_root:
        raise PlanError(f"configuration contains unsupported root settings: {', '.join(unknown_root)}")

    runtime_raw = raw.get("runtime", {})
    if not isinstance(runtime_raw, dict):
        raise PlanError("runtime configuration must be an object")
    unknown_runtime = sorted(set(runtime_raw) - _RUNTIME_FIELDS)
    if unknown_runtime:
        raise PlanError(f"runtime configuration contains unsupported settings: {', '.join(unknown_runtime)}")
    output_root_value = runtime_raw.get("output_root", project_root / "output" / "data")
    if not isinstance(output_root_value, (str, Path)) or not str(output_root_value).strip():
        raise PlanError("runtime output_root must be a non-empty path string")
    output_root = Path(output_root_value)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    default_workers = runtime_raw.get("default_max_workers", 4)
    maximum_workers = runtime_raw.get("maximum_workers", 16)
    if not _positive_integer(default_workers, minimum=1, maximum=64) or not _positive_integer(
            maximum_workers,
            minimum=1,
            maximum=64,
    ):
        raise PlanError("worker limits must be integers")
    if not 1 <= default_workers <= maximum_workers <= 64:
        raise PlanError("worker limits must satisfy 1 <= default <= maximum <= 64")
    package_completed_runs = runtime_raw.get("package_completed_runs", True)
    if not isinstance(package_completed_runs, bool):
        raise PlanError("runtime package_completed_runs must be boolean")
    maximum_run_output_bytes = runtime_raw.get(
        "maximum_run_output_bytes",
        DEFAULT_MAXIMUM_RUN_OUTPUT_BYTES,
    )
    if not _positive_integer(
            maximum_run_output_bytes,
            minimum=1,
            maximum=MAXIMUM_RUN_OUTPUT_BYTES,
    ):
        raise PlanError("runtime maximum_run_output_bytes must be an integer from 1 to 68719476736")

    interaction_raw = raw.get("interaction", {})
    if not isinstance(interaction_raw, dict):
        raise PlanError("interaction configuration must be an object")
    unknown_interaction = sorted(set(interaction_raw) - _INTERACTION_FIELDS)
    if unknown_interaction:
        raise PlanError(f"interaction configuration contains unsupported settings: {', '.join(unknown_interaction)}")
    history_enabled = interaction_raw.get("history_enabled", False)
    model_debug = interaction_raw.get("model_debug", False)
    similarity_threshold = interaction_raw.get("similarity_threshold", 0.55)
    model_name = interaction_raw.get("model_name", "stdlib-sequence-matcher")
    if not isinstance(history_enabled, bool) or not isinstance(model_debug, bool):
        raise PlanError("interaction history_enabled and model_debug must be boolean")
    if not _bounded_number(similarity_threshold, minimum=0, maximum=1):
        raise PlanError("interaction similarity_threshold must be a number from 0 to 1")
    if not isinstance(model_name, str) or not model_name.strip() or any(char in model_name for char in "\r\n"):
        raise PlanError("interaction model_name must be a non-empty single-line string")

    maintenance_raw = raw.get("maintenance", {})
    if not isinstance(maintenance_raw, dict):
        raise PlanError("maintenance configuration must be an object")
    unknown_maintenance = sorted(set(maintenance_raw) - _MAINTENANCE_FIELDS)
    if unknown_maintenance:
        raise PlanError(
            "maintenance configuration contains unsupported settings: "
            f"{', '.join(unknown_maintenance)}"
        )
    remote_url = maintenance_raw.get("remote_manifest_url")
    remote_sha256 = maintenance_raw.get("remote_manifest_sha256")
    if (remote_url is None) != (remote_sha256 is None):
        raise PlanError("remote manifest URL and SHA-256 must be configured together")
    if remote_url is not None and (
        not isinstance(remote_url, str)
        or not remote_url.startswith("https://")
        or "\n" in remote_url
    ):
        raise PlanError("remote_manifest_url must be an HTTPS URL")
    if remote_sha256 is not None and (
        not isinstance(remote_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", remote_sha256) is None
    ):
        raise PlanError("remote_manifest_sha256 must be a lowercase SHA-256 digest")
    local_manifest_value = maintenance_raw.get("local_manifest_path", "project.manifest.json")
    if not isinstance(local_manifest_value, str) or not local_manifest_value.strip():
        raise PlanError("local_manifest_path must be a non-empty relative path")
    local_manifest_path = Path(local_manifest_value)
    if (
        local_manifest_path.is_absolute()
        or local_manifest_path.drive
        or ".." in local_manifest_path.parts
    ):
        raise PlanError("local_manifest_path must remain inside the project")
    version_pattern = re.compile(r"^\d+\.\d+$")
    minimum_python = maintenance_raw.get("minimum_python", "3.11")
    recommended_python = maintenance_raw.get("recommended_python", "3.11")
    if (
        not isinstance(minimum_python, str)
        or version_pattern.fullmatch(minimum_python) is None
    ):
        raise PlanError("minimum_python must use major.minor form")
    if (
        not isinstance(recommended_python, str)
        or version_pattern.fullmatch(recommended_python) is None
    ):
        raise PlanError("recommended_python must use major.minor form")
    if tuple(map(int, recommended_python.split("."))) < tuple(
        map(int, minimum_python.split("."))
    ):
        raise PlanError("recommended_python must not be older than minimum_python")

    logging_raw = raw.get("logging", {})
    if not isinstance(logging_raw, dict):
        raise PlanError("logging configuration must be an object")
    unknown_logging = sorted(set(logging_raw) - _LOGGING_FIELDS)
    if unknown_logging:
        raise PlanError(
            f"logging configuration contains unsupported settings: {', '.join(unknown_logging)}"
        )
    logging_level = logging_raw.get("level", "INFO")
    if not isinstance(logging_level, str) or logging_level.upper() not in _LOG_LEVELS:
        raise PlanError("logging level must be DEBUG, INFO, WARNING, ERROR, CRITICAL, INTERNAL, or EXCEPTION")
    console_enabled = logging_raw.get("console_enabled", True)
    color_enabled = logging_raw.get("color_enabled", True)
    file_enabled = logging_raw.get("file_enabled", True)
    delete_previous = logging_raw.get("delete_previous", False)
    if not all(
        isinstance(value, bool)
        for value in (console_enabled, color_enabled, file_enabled, delete_previous)
    ):
        raise PlanError("logging enable, color, file, and deletion settings must be boolean")
    log_maximum_bytes = logging_raw.get("maximum_bytes", 4 * 1024 * 1024)
    if (
        not isinstance(log_maximum_bytes, int)
        or isinstance(log_maximum_bytes, bool)
        or not 1024 <= log_maximum_bytes <= 64 * 1024 * 1024
    ):
        raise PlanError("logging maximum_bytes must be an integer from 1024 to 67108864")
    retention_days = logging_raw.get("retention_days", 30)
    if (
        not isinstance(retention_days, int)
        or isinstance(retention_days, bool)
        or not 0 <= retention_days <= 3650
    ):
        raise PlanError("logging retention_days must be an integer from 0 to 3650")

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
            package_completed_runs=package_completed_runs,
            maximum_run_output_bytes=maximum_run_output_bytes,
        ),
        interaction=InteractionSettings(
            history_enabled=history_enabled,
            similarity_threshold=float(similarity_threshold),
            model_name=model_name,
            model_debug=model_debug,
        ),
        maintenance=MaintenanceSettings(
            remote_manifest_url=remote_url,
            remote_manifest_sha256=remote_sha256,
            local_manifest_path=local_manifest_path,
            minimum_python=minimum_python,
            recommended_python=recommended_python,
        ),
        logging=LoggingSettings(
            level=logging_level.upper(),
            console_enabled=console_enabled,
            color_enabled=color_enabled,
            file_enabled=file_enabled,
            maximum_bytes=log_maximum_bytes,
            delete_previous=delete_previous,
            retention_days=retention_days,
        ),
        collector_settings=collector_settings,
        migrated_from_schema=migrated_from_schema,
    )
