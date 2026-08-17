"""Stable v4 contracts shared by the engine, core collectors, and plugins."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

CONTRACT_VERSION = "4.0"
_CUSTOM_SPECIALTY = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class CollectorKind(StrEnum):
    """The source tree that owns a collector."""

    CORE = "core"
    PLUGIN = "plugin"


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
    REGISTRY_READ = "registry_read"
    SUBPROCESS = "subprocess"
    NETWORK = "network"
    PACKET_CAPTURE = "packet_capture"
    BROWSER_DATA = "browser_data"
    SENSITIVE_FILES = "sensitive_files"
    PRIVATE_KEYS = "private_keys"
    ELEVATED_PRIVILEGES = "elevated_privileges"


@dataclass(frozen=True, slots=True)
class CollectorMetadata:
    """Declarative identity, limits, and permissions for one collector."""

    id: str
    name: str
    version: str
    specialty: Specialty | str
    description: str
    author: str
    supported_platforms: tuple[str, ...] = ("win32",)
    capabilities: tuple[Capability, ...] = ()
    sensitive_data_categories: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    default_profiles: tuple[str, ...] = ("standard",)
    timeout_seconds: int = 60
    maximum_output_bytes: int = 100 * 1024 * 1024
    minimum_contract_version: str = CONTRACT_VERSION
    parallel_safe: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata."""
        data = asdict(self)
        data["specialty"] = self.specialty.value if isinstance(self.specialty, Specialty) else self.specialty
        data["capabilities"] = [capability.value for capability in self.capabilities]
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
        for field_name in (
                "supported_platforms",
                "sensitive_data_categories",
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """The outcome of a collector's side-effect-free validation phase."""

    valid: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CollectorResult:
    """The normalized result returned by an isolated collector worker."""

    status: CollectorStatus
    summary: str
    artifacts: tuple[Artifact, ...] = ()
    errors: tuple[str, ...] = ()
    metrics: Mapping[str, int | float | str] = field(default_factory=dict)

    @classmethod
    def succeeded(cls, summary: str, artifacts: tuple[Artifact, ...] = ()) -> "CollectorResult":
        return cls(CollectorStatus.SUCCEEDED, summary, artifacts)


@dataclass(frozen=True, slots=True)
class RunRequest:
    """An immutable user request resolved before any collector starts."""

    profile: str = "standard"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    enable_plugins: bool = False
    max_workers: int = 4
    acknowledge_authorization: bool = False
    approved_capabilities: tuple[Capability, ...] = ()


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
            artifacts: "ArtifactWriter",
            logger: EventLogger,
            settings: Mapping[str, Any],
            cancellation_file: Path,
    ) -> None:
        self.run_id = run_id
        self.collector_id = collector_id
        self.workspace = workspace
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
    def register_file(self, source: Path, *, media_type: str = "application/octet-stream") -> Artifact:
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

    @abstractmethod
    def collect(self, context: CollectorContext) -> CollectorResult:
        """Collect evidence and return a normalized result."""

    def cleanup(self, context: CollectorContext) -> None:
        """Release collector-local resources. The engine owns filesystem cleanup."""


class CoreCollector(Collector):
    """Marker base class for collectors shipped with Logicytics."""


class PluginCollector(Collector):
    """Marker base class for user-owned collectors discovered in plugins/."""
