"""Stable v4 contracts shared by the engine, core collectors, and plugins."""

from __future__ import annotations

import re
from datetime import datetime
from math import isfinite
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

CONTRACT_VERSION = "4.0"
_CUSTOM_SPECIALTY = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_COLLECTOR_ID = re.compile(r"^(?:core|plugin|mod)\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?$")
_SEMANTIC_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_CONTRACT_VERSION = re.compile(r"^\d+\.\d+$")
_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RUN_ID = re.compile(r"^run-[0-9a-f]{32}$")
_ARTIFACT_ID = re.compile(r"^artifact\.[0-9a-f]{32}$")
_ARTIFACT_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$")


class CollectorKind(StrEnum):
    """The source tree that owns a collector."""

    CORE = "core"
    PLUGIN = "plugin"
    MOD = "mod"


class CollectorStatus(StrEnum):
    """Terminal states reported for an individual collector."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RunStatus(StrEnum):
    """Terminal and in-progress states for a complete run."""

    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutputPolicy(StrEnum):
    """How a completed run publishes its evidence."""

    PACKAGE = "package"
    MANIFEST_ONLY = "manifest_only"


class PostRunAction(StrEnum):
    """An explicitly requested host action after durable run publication."""

    NONE = "none"
    REBOOT = "reboot"
    SHUTDOWN = "shutdown"


class EvidenceKind(StrEnum):
    """The package section that owns a registered evidence artifact."""

    RAW = "raw"
    DERIVED = "derived"


class Specialty(StrEnum):
    """The closed set of supported collector specialties."""

    SYSTEM = "system"
    HARDWARE = "hardware"
    PROCESS = "process"
    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    PACKET = "packet"
    WIRELESS = "wireless"
    BLUETOOTH = "bluetooth"
    USB = "usb"
    BROWSER = "browser"
    REGISTRY = "registry"
    EVENT_LOG = "event_log"
    STORAGE = "storage"
    ENCRYPTION = "encryption"
    MEDIA = "media"
    SSH = "ssh"
    DIAGNOSTICS = "diagnostics"
    REPORTING = "reporting"
    INTEGRATION = "integration"


class Capability(StrEnum):
    """Explicitly approved platform access a collector may request."""

    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    REGISTRY_READ = "registry_read"
    SUBPROCESS = "subprocess"
    NETWORK = "network"
    PACKET_CAPTURE = "packet_capture"
    BROWSER_DATA = "browser_data"
    SENSITIVE_FILES = "sensitive_files"
    PRIVATE_KEYS = "private_keys"
    ELEVATED_PRIVILEGES = "elevated_privileges"


class ResourceClass(StrEnum):
    """Scheduling resources whose conflicting collectors cannot safely overlap."""

    GENERAL = "general"
    DISK_HEAVY = "disk_heavy"
    NETWORK_HEAVY = "network_heavy"
    REGISTRY_SENSITIVE = "registry_sensitive"
    INTERACTIVE = "interactive"


class PrivilegeLevel(StrEnum):
    """Host privilege required before a collector may launch."""

    STANDARD = "standard"
    ELEVATED = "elevated"


class NetworkAccess(StrEnum):
    """Declared network reach of a collector's implementation."""

    NONE = "none"
    LOCAL = "local"
    REMOTE = "remote"


class EstimatedCost(StrEnum):
    """Coarse scheduling and consent cost declared before execution."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class CollectorMetadata:
    """Declarative identity, limits, and permissions for one collector."""

    id: str
    name: str
    version: str
    specialty: Specialty | str
    description: str
    author: str
    privilege_level: PrivilegeLevel = PrivilegeLevel.STANDARD
    network_access: NetworkAccess = NetworkAccess.NONE
    estimated_cost: EstimatedCost = EstimatedCost.LOW
    secondary_categories: tuple[str, ...] = ()
    output_media_types: tuple[str, ...] = ("application/octet-stream",)
    supported_platforms: tuple[str, ...] = ("win32",)
    capabilities: tuple[Capability, ...] = ()
    sensitive_data_categories: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    default_profiles: tuple[str, ...] = ("standard",)
    timeout_seconds: int = 60
    maximum_memory_bytes: int = 512 * 1024 * 1024
    maximum_output_bytes: int = 100 * 1024 * 1024
    maximum_artifact_bytes: int | None = None
    maximum_artifact_files: int = 500
    maximum_retries: int = 0
    retry_delay_seconds: float = 0.0
    minimum_contract_version: str = CONTRACT_VERSION
    parallel_safe: bool = True
    resource_class: ResourceClass = ResourceClass.GENERAL

    def __post_init__(self) -> None:
        """Reject malformed metadata before it can enter planning or runtime."""
        for name in ("id", "name", "version", "description", "author", "minimum_contract_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
                raise ValueError(f"metadata {name} must be a non-empty single-line string")
        if not _COLLECTOR_ID.fullmatch(self.id):
            raise ValueError("metadata id has an invalid schema")
        if not _SEMANTIC_VERSION.fullmatch(self.version):
            raise ValueError("metadata version must use semantic versioning")
        if not _CONTRACT_VERSION.fullmatch(self.minimum_contract_version):
            raise ValueError("metadata minimum_contract_version has an invalid schema")
        if not isinstance(self.specialty, Specialty) and (
                not isinstance(self.specialty, str) or not _CUSTOM_SPECIALTY.fullmatch(self.specialty)):
            raise ValueError("metadata specialty has an invalid schema")
        self._validate_labels("supported_platforms", self.supported_platforms, require_value=True)
        self._validate_labels("sensitive_data_categories", self.sensitive_data_categories)
        self._validate_labels("secondary_categories", self.secondary_categories)
        self._validate_labels("default_profiles", self.default_profiles, require_value=True)
        primary_specialty = self.specialty.value if isinstance(self.specialty, Specialty) else self.specialty
        if primary_specialty in self.secondary_categories:
            raise ValueError("metadata secondary_categories must not repeat the primary specialty")
        if not isinstance(self.output_media_types, tuple) or not self.output_media_types or not all(
                isinstance(media_type, str) and _MEDIA_TYPE.fullmatch(media_type)
                for media_type in self.output_media_types
        ):
            raise ValueError("metadata output_media_types must be a non-empty tuple of MIME types")
        if len(set(self.output_media_types)) != len(self.output_media_types):
            raise ValueError("metadata output_media_types must not contain duplicates")
        if self.sensitive_data_categories and {"standard", "minimal"}.intersection(self.default_profiles):
            raise ValueError("sensitive collectors must not belong to standard or minimal profiles")
        if not isinstance(self.capabilities, tuple) or not all(
                isinstance(capability, Capability) for capability in self.capabilities):
            raise ValueError("metadata capabilities must be a tuple of Capability values")
        if not isinstance(self.privilege_level, PrivilegeLevel):
            raise ValueError("metadata privilege_level must be a PrivilegeLevel value")
        if not isinstance(self.network_access, NetworkAccess):
            raise ValueError("metadata network_access must be a NetworkAccess value")
        if not isinstance(self.estimated_cost, EstimatedCost):
            raise ValueError("metadata estimated_cost must be an EstimatedCost value")
        requires_elevation = Capability.ELEVATED_PRIVILEGES in self.capabilities
        if requires_elevation != (self.privilege_level is PrivilegeLevel.ELEVATED):
            raise ValueError("metadata privilege_level must match the elevated_privileges capability")
        if Capability.NETWORK in self.capabilities and self.network_access is NetworkAccess.NONE:
            raise ValueError("metadata network_access must declare local or remote access")
        if not isinstance(self.dependencies, tuple) or not all(
                isinstance(dependency, str) and _COLLECTOR_ID.fullmatch(dependency)
                for dependency in self.dependencies):
            raise ValueError("metadata dependencies must be collector IDs")
        if len(set(self.dependencies)) != len(self.dependencies) or self.id in self.dependencies:
            raise ValueError("metadata dependencies must be unique and cannot include the collector itself")
        if self.maximum_artifact_bytes is None:
            object.__setattr__(self, "maximum_artifact_bytes", self.maximum_output_bytes)
        for name in (
                "timeout_seconds",
                "maximum_memory_bytes",
                "maximum_output_bytes",
                "maximum_artifact_bytes",
                "maximum_artifact_files",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"metadata {name} must be a positive integer")
        if self.maximum_artifact_bytes > self.maximum_output_bytes:
            raise ValueError("metadata maximum_artifact_bytes must not exceed maximum_output_bytes")
        if not isinstance(self.maximum_retries, int) or isinstance(self.maximum_retries, bool) or not (
                0 <= self.maximum_retries <= 3
        ):
            raise ValueError("metadata maximum_retries must be an integer from 0 to 3")
        if not isinstance(self.retry_delay_seconds, (int, float)) or isinstance(
                self.retry_delay_seconds,
                bool,
        ) or not isfinite(self.retry_delay_seconds) or not 0 <= self.retry_delay_seconds <= 30:
            raise ValueError("metadata retry_delay_seconds must be a number from 0 to 30")
        if not isinstance(self.parallel_safe, bool):
            raise ValueError("metadata parallel_safe must be boolean")
        if not isinstance(self.resource_class, ResourceClass):
            raise ValueError("metadata resource_class must be a ResourceClass value")

    @staticmethod
    def _validate_labels(name: str, values: tuple[str, ...], *, require_value: bool = False) -> None:
        """Require unique lower-snake-case labels for selector-like metadata fields."""
        if not isinstance(values, tuple) or (require_value and not values) or not all(
                isinstance(value, str) and _LABEL.fullmatch(value) for value in values):
            raise ValueError(f"metadata {name} must be a tuple of lowercase labels")
        if len(set(values)) != len(values):
            raise ValueError(f"metadata {name} must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata."""
        data = asdict(self)
        data["specialty"] = self.specialty.value if isinstance(self.specialty, Specialty) else self.specialty
        data["capabilities"] = [capability.value for capability in self.capabilities]
        data["resource_class"] = self.resource_class.value
        data["privilege_level"] = self.privilege_level.value
        data["network_access"] = self.network_access.value
        data["estimated_cost"] = self.estimated_cost.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, allow_custom_specialty: bool = False) -> "CollectorMetadata":
        """Build metadata returned by an isolated validation worker."""
        values = dict(data)
        specialty = values["specialty"]
        try:
            values["specialty"] = Specialty(specialty)
        except ValueError:
            if not allow_custom_specialty or not isinstance(specialty, str) or not _CUSTOM_SPECIALTY.fullmatch(
                    specialty):
                raise ValueError("collector specialty is unsupported")
            values["specialty"] = specialty
        values["capabilities"] = tuple(Capability(capability) for capability in values.get("capabilities", ()))
        values["resource_class"] = ResourceClass(values.get("resource_class", ResourceClass.GENERAL))
        values["privilege_level"] = PrivilegeLevel(values.get("privilege_level", PrivilegeLevel.STANDARD))
        values["network_access"] = NetworkAccess(values.get("network_access", NetworkAccess.NONE))
        values["estimated_cost"] = EstimatedCost(values.get("estimated_cost", EstimatedCost.LOW))
        for field_name in (
                "supported_platforms",
                "sensitive_data_categories",
                "secondary_categories",
                "output_media_types",
                "dependencies",
                "default_profiles",
        ):
            values[field_name] = tuple(values.get(field_name, ()))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class Artifact:
    """An evidence artifact registered by a collector."""

    id: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str
    collector_id: str
    source_category: str
    collected_at: str
    transformations: tuple[str, ...]
    evidence_kind: EvidenceKind
    name: str
    status: str = "registered"

    def __post_init__(self) -> None:
        """Reject malformed evidence records before worker or package publication."""
        if not isinstance(self.id, str) or not _ARTIFACT_ID.fullmatch(self.id):
            raise ValueError("artifact id must be a stable artifact identifier")
        if not isinstance(self.relative_path, str) or not self.relative_path.strip():
            raise ValueError("artifact relative_path must be a non-empty string")
        if not isinstance(self.sha256, str) or not _ARTIFACT_SHA256.fullmatch(self.sha256):
            raise ValueError("artifact sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be a non-negative integer")
        if not isinstance(self.media_type, str) or not _MEDIA_TYPE.fullmatch(self.media_type):
            raise ValueError("artifact media_type must be a valid MIME type")
        if not isinstance(self.collector_id, str) or not _COLLECTOR_ID.fullmatch(self.collector_id):
            raise ValueError("artifact collector_id must identify its producing collector")
        if not isinstance(self.source_category, str) or not _LABEL.fullmatch(self.source_category):
            raise ValueError("artifact source_category must be a lowercase category label")
        if not isinstance(self.name, str) or not self.name.strip() or any(value in self.name for value in "\r\n/\\"):
            raise ValueError("artifact name must be a safe human-readable filename")
        if self.status != "registered":
            raise ValueError("artifact status must be registered")
        try:
            collected_at = datetime.fromisoformat(self.collected_at)
        except (TypeError, ValueError) as error:
            raise ValueError("artifact collected_at must be an ISO-8601 timestamp") from error
        if collected_at.tzinfo is None:
            raise ValueError("artifact collected_at must include a timezone")
        if not isinstance(self.transformations, tuple) or any(
                not isinstance(step, str) or not step.strip() for step in self.transformations
        ):
            raise ValueError("artifact transformations must be a tuple of non-empty strings")
        if not isinstance(self.evidence_kind, EvidenceKind):
            raise ValueError("artifact evidence_kind must be an EvidenceKind value")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Artifact":
        """Reconstruct a strict artifact after a JSON or worker boundary."""
        values = dict(data)
        transformations = values.get("transformations")
        if not isinstance(transformations, (list, tuple)):
            raise ValueError("artifact transformations must be an array or tuple")
        values["transformations"] = tuple(transformations)
        try:
            values["evidence_kind"] = EvidenceKind(values["evidence_kind"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("artifact evidence_kind must be raw or derived") from error
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """The outcome of a collector's side-effect-free validation phase."""

    valid: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CollectionEstimate:
    """A collector's optional, side-effect-free resource estimate."""

    estimated_seconds: float
    estimated_output_bytes: int
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject estimates that cannot describe a possible collection run."""
        if self.estimated_seconds < 0:
            raise ValueError("estimated_seconds must not be negative")
        if self.estimated_output_bytes < 0:
            raise ValueError("estimated_output_bytes must not be negative")


@dataclass(frozen=True, slots=True)
class CollectorResult:
    """The normalized result returned by an isolated collector worker."""

    status: CollectorStatus
    summary: str
    artifacts: tuple[Artifact, ...] = ()
    errors: tuple[str, ...] = ()
    metrics: Mapping[str, int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject malformed terminal results before they can cross a worker boundary."""
        if not isinstance(self.status, CollectorStatus):
            raise ValueError("collector result status must be a CollectorStatus value")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("collector result summary must be a non-empty string")
        if not isinstance(self.artifacts, tuple) or not all(isinstance(item, Artifact) for item in self.artifacts):
            raise ValueError("collector result artifacts must be a tuple of Artifact values")
        artifact_ids = tuple(item.id for item in self.artifacts)
        artifact_paths = tuple(item.relative_path for item in self.artifacts)
        if len(set(artifact_ids)) != len(artifact_ids) or len(set(artifact_paths)) != len(artifact_paths):
            raise ValueError("collector result artifacts must have unique IDs and paths")
        if not isinstance(self.errors, tuple) or not all(
                isinstance(error, str) and error.strip() for error in self.errors
        ):
            raise ValueError("collector result errors must be a tuple of non-empty strings")
        if not isinstance(self.metrics, Mapping) or not all(
                isinstance(name, str) and name.strip()
                and isinstance(value, (int, float, str))
                and not isinstance(value, bool)
                and (not isinstance(value, float) or isfinite(value))
                for name, value in self.metrics.items()
        ):
            raise ValueError("collector result metrics must contain finite scalar values")

    @classmethod
    def succeeded(cls, summary: str, artifacts: tuple[Artifact, ...] = ()) -> "CollectorResult":
        return cls(CollectorStatus.SUCCEEDED, summary, artifacts)

    @classmethod
    def partial(
            cls,
            summary: str,
            artifacts: tuple[Artifact, ...] = (),
            *,
            errors: tuple[str, ...] = (),
    ) -> "CollectorResult":
        """Return an explicitly incomplete result while preserving registered evidence."""
        return cls(CollectorStatus.PARTIAL, summary, artifacts, errors=errors)

    @classmethod
    def skipped(cls, summary: str, *, errors: tuple[str, ...] = ()) -> "CollectorResult":
        """Return an explicit prerequisite or policy skip."""
        return cls(CollectorStatus.SKIPPED, summary, errors=errors)

    @classmethod
    def cancelled(
            cls,
            summary: str,
            artifacts: tuple[Artifact, ...] = (),
    ) -> "CollectorResult":
        """Return explicit cancellation while retaining already registered evidence."""
        return cls(CollectorStatus.CANCELLED, summary, artifacts)

    @classmethod
    def failed(
            cls,
            summary: str,
            *,
            errors: tuple[str, ...] = (),
            artifacts: tuple[Artifact, ...] = (),
    ) -> "CollectorResult":
        """Return explicit failure while retaining already registered evidence."""
        return cls(CollectorStatus.FAILED, summary, artifacts, errors=errors)


@dataclass(frozen=True, slots=True)
class RunRequest:
    """An immutable user request resolved before any collector starts."""

    profile: str = "standard"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    selection_only: bool = False
    enable_plugins: bool = False
    enable_mods: bool = False
    non_python_only: bool = False
    max_workers: int = 4
    acknowledge_authorization: bool = False
    approved_capabilities: tuple[Capability, ...] = ()
    performance_check: bool = False
    rerun_from: str | None = None
    output_policy: OutputPolicy = OutputPolicy.PACKAGE
    post_run_action: PostRunAction = PostRunAction.NONE

    def __post_init__(self) -> None:
        """Reject malformed selections and execution policy before planning."""
        if not isinstance(self.profile, str) or not _LABEL.fullmatch(self.profile):
            raise ValueError("request profile must be a lowercase profile label")
        for name in ("include", "exclude"):
            selections = getattr(self, name)
            if not isinstance(selections, tuple) or not all(
                    isinstance(collector_id, str) and _COLLECTOR_ID.fullmatch(collector_id)
                    for collector_id in selections
            ):
                raise ValueError(f"request {name} must be a tuple of collector IDs")
            if len(set(selections)) != len(selections):
                raise ValueError(f"request {name} must not contain duplicate collector IDs")
        if set(self.include).intersection(self.exclude):
            raise ValueError("request include and exclude selections must not overlap")
        if self.rerun_from is not None and (
                not isinstance(self.rerun_from, str) or not _RUN_ID.fullmatch(self.rerun_from)
        ):
            raise ValueError("request rerun_from must be a valid original run ID")
        if self.rerun_from is not None and not self.include:
            raise ValueError("request rerun_from requires explicit included collector IDs")
        for name in (
                "selection_only", "enable_plugins", "enable_mods", "non_python_only",
                "acknowledge_authorization", "performance_check",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"request {name} must be boolean")
        if self.selection_only and not self.include:
            raise ValueError("request selection_only requires explicit included collector IDs")
        if not isinstance(self.max_workers, int) or isinstance(self.max_workers, bool) or not 1 <= self.max_workers <= 64:
            raise ValueError("request max_workers must be an integer from 1 to 64")
        if self.performance_check and self.max_workers != 1:
            raise ValueError("request performance_check requires max_workers=1")
        if not isinstance(self.approved_capabilities, tuple) or not all(
                isinstance(capability, Capability) for capability in self.approved_capabilities
        ):
            raise ValueError("request approved_capabilities must be a tuple of Capability values")
        if len(set(self.approved_capabilities)) != len(self.approved_capabilities):
            raise ValueError("request approved_capabilities must not contain duplicates")
        if not isinstance(self.output_policy, OutputPolicy):
            raise ValueError("request output_policy must be an OutputPolicy value")
        if not isinstance(self.post_run_action, PostRunAction):
            raise ValueError("request post_run_action must be a PostRunAction value")
        if self.post_run_action is not PostRunAction.NONE and self.output_policy is not OutputPolicy.PACKAGE:
            raise ValueError("post-run actions require packaged output")


class EventLogger(ABC):
    """A structured event sink scoped to one run or collector worker."""

    @abstractmethod
    def event(self, level: str, message: str, **fields: int | float | str) -> None:
        """Record one machine-readable event without exposing raw evidence."""


class CollectorContext:
    """The limited, per-worker interface exposed to a collector."""

    def __init__(
            self,
            run_id: str,
            collector_id: str,
            workspace: Path,
            temporary_directory: Path,
            artifacts: "ArtifactWriter",
            logger: EventLogger,
            settings: Mapping[str, Any],
            cancellation_file: Path,
    ) -> None:
        self.run_id = run_id
        self.collector_id = collector_id
        self.workspace = workspace
        self.temporary_directory = temporary_directory
        self.artifacts = artifacts
        self.logger = logger
        self.settings = settings
        self._cancellation_file = cancellation_file

    @property
    def is_cancelled(self) -> bool:
        """Whether the supervisor has requested cancellation."""
        return self._cancellation_file.exists()

    def report_progress(self, event: str, **metrics: int | float | str) -> None:
        """Write a structured progress event local to this collector workspace."""
        self.logger.event("info", event, **metrics)


class ArtifactWriter(ABC):
    """The only supported route from a collector workspace into run artifacts."""

    @abstractmethod
    def register_file(
            self,
            source: Path,
            *,
            media_type: str = "application/octet-stream",
            evidence_kind: EvidenceKind = EvidenceKind.DERIVED,
            transformations: tuple[str, ...] = (),
    ) -> Artifact:
        """Register a file created within the collector workspace."""


class Collector(ABC):
    """Base interface that all v4 collector classes must implement."""

    @classmethod
    @abstractmethod
    def metadata(cls) -> CollectorMetadata:
        """Return immutable collector metadata without performing collection."""

    @abstractmethod
    def validate(self, context: CollectorContext) -> ValidationResult:
        """Validate prerequisites without creating evidence artifacts."""

    def prepare(self, context: CollectorContext) -> ValidationResult:
        """Prepare collector-local resources after validation and authorization."""
        return ValidationResult(True)

    @abstractmethod
    def collect(self, context: CollectorContext) -> CollectorResult:
        """Collect evidence and return a normalized result."""

    def finalize(self, context: CollectorContext, result: CollectorResult) -> CollectorResult:
        """Finalize collector-local evidence and return the publishable result."""
        return result

    def estimate(self, context: CollectorContext) -> CollectionEstimate:
        """Optionally estimate time and output without collecting evidence."""
        return CollectionEstimate(estimated_seconds=0, estimated_output_bytes=0)

    @classmethod
    def dependencies(cls) -> tuple[str, ...]:
        """Optionally declare collector IDs that must complete before this collector."""
        return ()

    def cleanup(self, context: CollectorContext) -> None:
        """Release collector-local resources. The engine owns filesystem cleanup."""


class CoreCollector(Collector):
    """Marker base class for collectors shipped with Logicytics."""


class PluginCollector(Collector):
    """Marker base class for user-owned collectors discovered in plugins/."""
