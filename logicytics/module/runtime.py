"""Per-collector process supervisor and run workspace lifecycle."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import mimetypes
import multiprocessing
import os
import queue
import re
import shlex
import shutil
import socket
import sys
import tempfile
import traceback
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from pathlib import Path
from subprocess import TimeoutExpired
from time import monotonic, sleep
from typing import Any, Protocol, TypedDict, TypeVar, cast
from uuid import uuid4

from logicytics.contracts import (
    Artifact,
    Capability,
    CollectorContext,
    CollectorMetadata,
    CollectorResult,
    CollectorStatus,
    EventLogger,
    OutputPolicy,
    PostRunAction,
    ResourceClass,
    RunStatus,
    ValidationResult,
)
from logicytics.module.artifacts import WorkspaceArtifactWriter
from logicytics.module.command_runner import parse_level_messages
from logicytics.module.configuration import AppConfig
from logicytics.module.discovery import CollectorCandidate
from logicytics.module.errors import CapabilityPolicyError, LogicyticsError
from logicytics.module.logging import FileEventLogger, get_application_logger, get_event_logger
from logicytics.module.manifest import CollectorRecord, RunManifest, utc_now, write_manifest
from logicytics.module.output_contracts import core_output_contract
from logicytics.module.output_layout import ensure_output_layout, run_fingerprint
from logicytics.module.packaging import package_manifest
from logicytics.module.planner import RunPlan
from logicytics.platform_adapters import process_adapter, windows_api_adapter

_T = TypeVar("_T")


@dataclass(slots=True)
class RunOutcome:
    """The completed run and the durable locations that describe it."""

    manifest: RunManifest
    run_directory: Path
    manifest_path: Path


class _ProcessContext(Protocol):
    """Typed multiprocessing context subset used by the supervisor."""

    def Process(
        self,
        group: None = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        *,
        daemon: bool | None = None,
    ) -> BaseProcess:
        """Create a worker process using the configured multiprocessing context."""
        ...

    def Queue(
        self,
        maxsize: int = 0,
    ) -> Queue[_T]:
        """Create a typed result queue for worker-to-supervisor messages."""
        ...


class _WorkerPayload(TypedDict):
    """Fully typed, pickle-safe payload passed to collector worker processes."""

    run_id: str
    collector_id: str
    path: str
    expected_class: str
    execution_type: str
    metadata: dict[str, object]
    blocked_capabilities: tuple[Capability, ...]
    workspace: str
    artifact_root: str
    cancellation_file: str
    settings: dict[str, Any]
    run_output_budget_bytes: int


class _SerializedResult(TypedDict):
    """Serialized CollectorResult shape transported through multiprocessing queues."""

    status: str
    summary: str
    artifacts: list[dict[str, object]]
    errors: list[str]
    metrics: dict[str, int | float | str]


class _WorkerMessage(TypedDict):
    """Terminal result message produced by a worker process."""

    collector_id: str
    result: _SerializedResult


@dataclass(slots=True)
class _ActiveWorker:
    """Track process, resource, workspace, and progress state for one active collector."""

    candidate_id: str
    process: BaseProcess
    started_at: float
    timeout_seconds: int
    maximum_memory_bytes: int
    workspace: Path
    parallel_safe: bool
    resource_class: ResourceClass
    reserved_output_bytes: int
    last_event_count: int = 0
    last_event_offset: int = 0
    last_heartbeat_at: float = 0.0
    exited_at: float | None = None


def _require_metadata(candidate: CollectorCandidate) -> CollectorMetadata:
    """Return preflight metadata or fail immediately on an invalid supervisor input."""
    metadata = candidate.metadata
    if metadata is None:
        raise RuntimeError(f"preflighted collector has no metadata: {candidate.path}")
    return metadata


class _WorkerMutationGuard:
    """Constrain audited collector mutations to its private workspace and evidence store."""

    _PATH_EVENTS = {
        "os.mkdir",
        "os.remove",
        "os.rmdir",
        "os.chmod",
        "os.chown",
        "os.utime",
        "os.truncate",
    }
    _DOUBLE_PATH_EVENTS = {"os.rename", "os.link", "os.symlink"}
    _BLOCKED_EVENTS = {"os.chdir", "os.putenv", "os.unsetenv", "os.system"}
    _FORBIDDEN_COMMANDS = {
        "choco",
        "git",
        "npm",
        "pip",
        "pip3",
        "pnpm",
        "poetry",
        "shutdown",
        "uv",
        "winget",
        "yarn",
    }
    _FORBIDDEN_TOOL_REFERENCE = re.compile(
        r"(?<![a-z0-9_.-])(?:choco|git|npm|pip(?:\d+(?:\.\d+)*)?|pnpm|poetry|shutdown|uv|winget|yarn)"
        r"(?:\.exe|\.cmd|\.bat)?(?![a-z0-9_.-])",
        re.IGNORECASE,
    )
    _FORBIDDEN_POWERSHELL = re.compile(
        r"(?<![a-z0-9_-])(?:install-module|install-package|restart-computer|stop-computer|update-module)(?![a-z0-9_-])",
        re.IGNORECASE,
    )

    def __init__(
        self,
        workspace: Path,
        artifact_root: Path,
        collector_id: str,
        capabilities: tuple[Capability, ...],
        collector_source: Path,
        blocked_capabilities: tuple[Capability, ...] = (),
    ) -> None:
        """Install audit boundaries for one collector's filesystem and capability scope."""
        self.roots = (
            workspace.resolve(),
            (artifact_root / collector_id.replace(".", "_")).resolve(),
        )
        self.runtime_roots = (Path(sys.base_prefix).resolve(), Path(__file__).resolve().parent)
        self.collector_source = collector_source.resolve()
        source_tree = next(
            (parent for parent in self.collector_source.parents if parent.name.casefold() in {"core", "plugins"}),
            None,
        )
        self.collector_roots = (
            ()
            if source_tree is None
            else (
                (source_tree.parent / "core").resolve(),
                (source_tree.parent / "plugins").resolve(),
            )
        )
        self.capabilities = frozenset(capabilities)
        self.blocked_capabilities = frozenset(blocked_capabilities)
        self.active = False
        sys.addaudithook(self._check_event)

    def _require_capability(self, capability: Capability, operation: str, detail: str = "") -> None:
        """Reject blocked access and undeclared access with distinct stable diagnostics."""
        if capability in self.blocked_capabilities:
            raise CapabilityPolicyError(
                "CAPABILITY_BLOCKED",
                capability.value,
                operation,
                detail or "blocked by the active run policy",
            )
        if capability not in self.capabilities:
            raise CapabilityPolicyError(
                "CAPABILITY_DECLARATION_MISMATCH",
                capability.value,
                operation,
                detail or "not declared in CollectorMetadata.capabilities",
            )

    def _check_path(self, value: object) -> None:
        """Reject writes that escape the collector workspace or artifact root."""
        if isinstance(value, int):
            return
        if not isinstance(value, (str, bytes, os.PathLike)):
            raise PermissionError("collector filesystem mutation has an unsupported target")
        path = Path(os.fsdecode(value)).resolve()
        if Capability.FILESYSTEM_WRITE in self.capabilities and (Capability.FILESYSTEM_WRITE not in self.blocked_capabilities):
            return
        if not any(path == root or root in path.parents for root in self.roots):
            self._require_capability(
                Capability.FILESYSTEM_WRITE,
                "filesystem_write",
                f"target={path}; collector filesystem mutation escapes its private workspace",
            )
            raise PermissionError(f"collector filesystem mutation escapes its private workspace: {path}")

    def _check_read(self, value: object) -> None:
        """Enforce declared read, browser, sensitive-file, and private-key capabilities."""
        if isinstance(value, int) or not isinstance(value, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(value)).resolve()
        if path == self.collector_source or any(path == root or root in path.parents for root in (*self.roots, *self.runtime_roots)):
            return
        self._require_capability(
            Capability.FILESYSTEM_READ,
            "filesystem_read",
            f"target={path}",
        )
        components = tuple(part.casefold() for part in path.parts)
        browser_data = any(part in {"chrome", "edge", "firefox", "opera software", "opera gx"} for part in components)
        private_key = (
            ".ssh" in components
            or path.name.casefold() in {"id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
            or path.suffix.casefold() in {".pem", ".ppk"}
        )
        sensitive = (
            private_key
            or browser_data
            or any(label in path.name.casefold() for label in ("cookie", "credential", "password", "token", "secret", "login data"))
        )
        if browser_data:
            self._require_capability(Capability.BROWSER_DATA, "browser_data", f"target={path}")
        if sensitive:
            self._require_capability(Capability.SENSITIVE_FILES, "sensitive_files", f"target={path}")
        if private_key:
            self._require_capability(Capability.PRIVATE_KEYS, "private_keys", f"target={path}")

    @staticmethod
    def _subprocess_tokens(arguments: tuple[object, ...]) -> tuple[str, ...]:
        """Normalize cross-platform subprocess audit arguments without executing a shell."""
        executable = arguments[0] if arguments else None
        command = arguments[1] if len(arguments) > 1 else None

        if isinstance(command, (list, tuple)):
            tokens = tuple(os.fsdecode(item) for item in command if isinstance(item, (str, bytes, os.PathLike)))
        elif isinstance(command, bytes):
            decoded = os.fsdecode(command)
            try:
                tokens = tuple(shlex.split(decoded, posix=os.name != "nt"))
            except ValueError:
                tokens = tuple(decoded.split())
        elif isinstance(command, str):
            try:
                tokens = tuple(shlex.split(command, posix=os.name != "nt"))
            except ValueError:
                tokens = tuple(command.split())
        else:
            tokens = ()

        if tokens:
            return tokens
        if isinstance(executable, (str, bytes, os.PathLike)):
            return (os.fsdecode(executable),)
        return ()

    @staticmethod
    def _command_stem(value: str) -> str:
        """Return a case-insensitive executable label without Windows wrapper suffixes."""
        name = Path(value.strip("\"'")).name.casefold()
        for suffix in (".exe", ".com", ".cmd", ".bat"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def _prohibited_subprocess(self, arguments: tuple[object, ...]) -> str | None:
        """Classify commands forbidden even when ordinary subprocess access was approved."""
        normalized = tuple(token.strip("\"'") for token in self._subprocess_tokens(arguments))
        stems = tuple(self._command_stem(token) for token in normalized)
        if any(stem in self._FORBIDDEN_COMMANDS for stem in stems):
            stem = next(stem for stem in stems if stem in self._FORBIDDEN_COMMANDS)
            return "system_power" if stem == "shutdown" else "repository_or_package_management"
        if any(re.fullmatch(r"pip(?:\d+(?:\.\d+)*)?", stem) for stem in stems):
            return "repository_or_package_management"
        for index, token in enumerate(normalized[:-1]):
            if token == "-m":
                module = normalized[index + 1].casefold()
                if module == "pip" or module.startswith("pip.") or module == "ensurepip":
                    return "repository_or_package_management"
                if module == "logicytics" or module.startswith("logicytics."):
                    return "main_application"
        command_text = " ".join(normalized)
        if re.search(
            r"(?<![a-z0-9_.-])logicytics\.(?:json|ya?ml)(?![a-z0-9_.-])",
            command_text,
            re.IGNORECASE,
        ):
            return "configuration_mutation"
        if re.search(
            r"(?<![a-z0-9_.\\/:-])logicytics(?:\.[a-z_][a-z0-9_]*)?(?![a-z0-9_.\\/:-])",
            command_text,
            re.IGNORECASE,
        ):
            return "main_application"
        if self._FORBIDDEN_POWERSHELL.search(command_text):
            return "system_power_or_package_management"
        if tool := self._FORBIDDEN_TOOL_REFERENCE.search(command_text):
            return "system_power" if tool.group(0).casefold().startswith("shutdown") else "repository_or_package_management"
        for token in normalized:
            candidate = Path(token)
            if candidate.suffix.casefold() != ".py":
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if any(resolved == root or root in resolved.parents for root in self.collector_roots):
                return "collector_launch"
        return None

    def _check_event(self, event: str, arguments: tuple[object, ...]) -> None:
        """Apply capability and mutation policy to one Python audit event."""
        if not self.active:
            return
        if event == "subprocess.Popen":
            self._require_capability(Capability.SUBPROCESS, "subprocess")
            if prohibited := self._prohibited_subprocess(arguments):
                raise PermissionError(f"collector subprocess command is prohibited: {prohibited}")
        if event.startswith("socket.") and event in {
            "socket.__new__",
            "socket.bind",
            "socket.connect",
            "socket.sendto",
            "socket.getaddrinfo",
        }:
            self._require_capability(Capability.NETWORK, "network")
            socket_type = arguments[2] if len(arguments) > 2 else None
            if event == "socket.__new__" and isinstance(socket_type, int) and socket_type == int(socket.SOCK_RAW):
                self._require_capability(Capability.PACKET_CAPTURE, "packet_capture")
        if event.startswith("winreg."):
            self._require_capability(Capability.REGISTRY_READ, "registry_read")
        if event in self._BLOCKED_EVENTS:
            raise PermissionError(f"collector must not modify process state: {event}")
        if event == "open":
            mode = arguments[1] if len(arguments) > 1 else None
            flags = arguments[2] if len(arguments) > 2 else 0
            writing = isinstance(mode, str) and any(flag in mode for flag in "wax+")
            writing = writing or isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                self._check_path(arguments[0])
            else:
                self._check_read(arguments[0])
        elif event in {"os.listdir", "os.scandir"} and arguments:
            self._check_read(arguments[0])
        elif event in self._PATH_EVENTS:
            self._check_path(arguments[0])
        elif event in self._DOUBLE_PATH_EVENTS:
            self._check_path(arguments[0])
            self._check_path(arguments[1])


def _load_collector(path: Path, expected_class: str):
    """Load and instantiate the preflighted collector class from its source path."""
    module_name = f"logicytics_runtime_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to load collector module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, expected_class)()


def _artifact_from_dict(data: Mapping[str, object]) -> Artifact:
    """Deserialize one artifact from an already validated mapping."""
    return Artifact.from_dict(dict(data))


def _result_from_dict(data: _SerializedResult) -> CollectorResult:
    """Deserialize the typed result payload emitted by a worker."""
    return CollectorResult(
        status=CollectorStatus(data["status"]),
        summary=data["summary"],
        artifacts=tuple(_artifact_from_dict(item) for item in data["artifacts"]),
        errors=tuple(data["errors"]),
        metrics=dict(data["metrics"]),
    )


def _mod_command(
    script: Path,
    execution_type: str,
    workspace: Path | None = None,
    collector_id: str = "mod.legacy",
    capabilities: tuple[Capability, ...] = (),
    blocked_capabilities: tuple[Capability, ...] = (),
) -> list[str]:
    """Build a shell-free command for one copied legacy MODS script."""
    if execution_type != "mod_python":
        raise ValueError("MODS supports Python (.py) scripts only")
    if workspace is None:
        raise ValueError("Python mod execution requires a private workspace")
    runner = Path(__file__).with_name("mod_runner.py")
    return [
        sys.executable,
        "-I",
        str(runner),
        str(script),
        str(workspace),
        collector_id,
        json.dumps([capability.value for capability in capabilities]),
        json.dumps([capability.value for capability in blocked_capabilities]),
    ]


def _run_mod_worker(payload: _WorkerPayload, result_queue: Queue[_WorkerMessage]) -> None:
    """Adapt a sidecar-declared legacy script to the isolated collector contract."""
    metadata = CollectorMetadata.from_dict(payload["metadata"], allow_custom_specialty=True)
    workspace = Path(payload["workspace"])
    artifact_root = Path(payload["artifact_root"])
    cancellation_file = Path(payload["cancellation_file"])
    source_directory = workspace / "source"
    source_directory.mkdir(parents=True, exist_ok=True)
    source = Path(payload["path"])
    copied_script = source_directory / source.name
    shutil.copy2(source, copied_script)
    writer = WorkspaceArtifactWriter(
        metadata.id,
        workspace,
        artifact_root,
        metadata.maximum_output_bytes,
        metadata.maximum_artifact_files,
        source_category=str(metadata.specialty),
        maximum_artifact_bytes=metadata.maximum_artifact_bytes,
        run_output_budget_bytes=payload["run_output_budget_bytes"],
        cancellation_file=cancellation_file,
    )
    logger = get_event_logger(
        workspace / "events.jsonl",
        run_id=payload["run_id"],
        collector_id=metadata.id,
    )
    stdout_path = workspace / "script_stdout.txt"
    stderr_path = workspace / "script_stderr.txt"
    guard = _WorkerMutationGuard(
        workspace,
        artifact_root,
        metadata.id,
        metadata.capabilities,
        source,
        payload["blocked_capabilities"],
    )
    result: CollectorResult
    try:
        if cancellation_file.exists():
            result = CollectorResult.cancelled("mod cancelled before execution")
        else:
            command = _mod_command(
                copied_script,
                payload["execution_type"],
                workspace,
                metadata.id,
                metadata.capabilities,
                payload["blocked_capabilities"],
            )
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "WINDIR": os.environ.get("WINDIR", ""),
                "COMSPEC": os.environ.get("COMSPEC", ""),
                "TEMP": str(workspace / "tmp"),
                "TMP": str(workspace / "tmp"),
                "LOGICYTICS_RUN_ID": payload["run_id"],
                "LOGICYTICS_COLLECTOR_ID": metadata.id,
                "LOGICYTICS_WORKSPACE": str(workspace),
            }
            guard.active = True
            try:
                completed = process_adapter.run(
                    command,
                    cwd=workspace,
                    env=environment,
                    capture_directory=workspace / "tmp",
                    capture_output=True,
                    check=False,
                    text=True,
                )
            finally:
                guard.active = False
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            for level, message in parse_level_messages(completed.stdout):
                logger.event(level.casefold(), message)
            excluded_roots = {source_directory.resolve(), (workspace / "tmp").resolve()}
            excluded_files = {
                stdout_path.resolve(),
                stderr_path.resolve(),
                (workspace / "events.jsonl").resolve(),
            }
            candidates = [stdout_path, stderr_path]
            candidates.extend(
                path
                for path in sorted(workspace.rglob("*"))
                if path.is_file()
                and path.resolve() not in excluded_files
                and not any(path.resolve().is_relative_to(root) for root in excluded_roots)
            )
            artifacts: list[Artifact] = []
            for path in candidates:
                if not path.exists() or path.stat().st_size == 0:
                    continue
                media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if media_type not in metadata.output_media_types:
                    raise ValueError(f"mod output media type was not declared: {media_type} ({path.name})")
                artifacts.append(writer.register_file(path, media_type=media_type))
            if cancellation_file.exists():
                result = CollectorResult.cancelled("mod cancelled after execution", tuple(artifacts))
            elif completed.returncode == 0:
                result = CollectorResult.succeeded("legacy mod completed", tuple(artifacts))
            else:
                result = CollectorResult.failed(
                    "legacy mod exited unsuccessfully",
                    errors=(
                        f"exit code {completed.returncode}",
                        completed.stderr.strip() or "no stderr",
                    ),
                    artifacts=tuple(artifacts),
                )
    except CapabilityPolicyError as error:
        guard.active = False
        logger.event(
            "error",
            "capability_policy_violation",
            code=error.code,
            capability=error.capability,
            operation=error.operation,
        )
        result = CollectorResult.failed(
            "legacy mod capability policy violation",
            errors=(str(error),),
            artifacts=writer.artifacts,
        )
    except BaseException as error:
        guard.active = False
        result = CollectorResult.failed(
            "legacy mod worker crashed",
            errors=(f"{type(error).__name__}: {error}", traceback.format_exc()),
            artifacts=writer.artifacts,
        )
    result_queue.put({"collector_id": metadata.id, "result": _serialize_result(result)})


def _configure_worker_temporary_directory(directory: Path) -> Callable[[], None]:
    """Route stdlib and child-process temporary files into one worker workspace."""
    location = str(directory.resolve())
    previous_tempdir = tempfile.tempdir
    previous_environment = {name: os.environ.get(name) for name in ("TEMP", "TMP")}
    tempfile.tempdir = location
    os.environ["TEMP"] = location
    os.environ["TMP"] = location

    def restore() -> None:
        """Restore the inherited process temporary-directory configuration."""
        tempfile.tempdir = previous_tempdir
        for name, previous_value in previous_environment.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value

    return restore


def _worker_entry(payload: _WorkerPayload, result_queue: Queue[_WorkerMessage]) -> None:
    """Run a single collector in an isolated child process."""
    if os.name != "nt":
        os.setsid()
    workspace = Path(payload["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    temporary_directory = workspace / "tmp"
    temporary_directory.mkdir(exist_ok=True)
    restore_temporary_directory = _configure_worker_temporary_directory(temporary_directory)
    if payload["execution_type"] != "collector":
        try:
            _run_mod_worker(payload, result_queue)
        except BaseException as error:
            result_queue.put(
                {
                    "collector_id": payload["collector_id"],
                    "result": _serialize_result(
                        CollectorResult.failed(
                            "legacy mod worker crashed",
                            errors=(f"{type(error).__name__}: {error}", traceback.format_exc()),
                        )
                    ),
                }
            )
        finally:
            restore_temporary_directory()
        return
    stdout_path = workspace / "stdout.log"
    stderr_path = workspace / "stderr.log"
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                collector = _load_collector(Path(payload["path"]), payload["expected_class"])
                metadata = collector.metadata()
                source_path = Path(payload["path"]).resolve()
                shipped_core_root = (Path(__file__).resolve().parent.parent / "core").resolve()
                try:
                    source_path.relative_to(shipped_core_root)
                except ValueError:
                    output_contract = None
                else:
                    output_contract = core_output_contract(metadata)
                writer = WorkspaceArtifactWriter(
                    metadata.id,
                    workspace,
                    Path(payload["artifact_root"]),
                    metadata.maximum_output_bytes,
                    metadata.maximum_artifact_files,
                    source_category=str(metadata.specialty),
                    maximum_artifact_bytes=metadata.maximum_artifact_bytes,
                    run_output_budget_bytes=payload["run_output_budget_bytes"],
                    cancellation_file=Path(payload["cancellation_file"]),
                    allowed_relative_paths=(output_contract.workspace_patterns if output_contract else None),
                    allowed_media_types=(output_contract.media_types if output_contract else None),
                )
                context = CollectorContext(
                    run_id=payload["run_id"],
                    collector_id=metadata.id,
                    workspace=workspace,
                    temporary_directory=temporary_directory,
                    artifacts=writer,
                    logger=get_event_logger(
                        workspace / "events.jsonl",
                        run_id=payload["run_id"],
                        collector_id=metadata.id,
                    ),
                    settings=dict(payload["settings"]),
                    cancellation_file=Path(payload["cancellation_file"]),
                )
                result: CollectorResult | None = None
                lifecycle_errors: list[str] = []
                failure_summary = "collector worker crashed"
                mutation_guard = _WorkerMutationGuard(
                    workspace,
                    Path(payload["artifact_root"]),
                    metadata.id,
                    metadata.capabilities,
                    Path(payload["path"]),
                    payload["blocked_capabilities"],
                )
                mutation_guard.active = True
                try:
                    validation = collector.validate(context)
                    if not isinstance(validation, ValidationResult):
                        raise TypeError("validate() must return ValidationResult")
                    if not validation.valid:
                        result = CollectorResult(
                            status=CollectorStatus.SKIPPED,
                            summary="collector prerequisites were not met",
                            errors=validation.reasons,
                        )
                    elif context.is_cancelled:
                        result = CollectorResult(CollectorStatus.CANCELLED, "cancelled before collection")
                    else:
                        preparation = collector.prepare(context)
                        if not isinstance(preparation, ValidationResult):
                            raise TypeError("prepare() must return ValidationResult")
                        if not preparation.valid:
                            result = CollectorResult.skipped(
                                "collector preparation was not completed",
                                errors=preparation.reasons,
                            )
                        elif context.is_cancelled:
                            result = CollectorResult.cancelled("cancelled after preparation")
                        else:
                            context.logger.event("info", "collection_started")
                            result = collector.collect(context)
                            if not isinstance(result, CollectorResult):
                                raise TypeError("collect() must return CollectorResult")
                            if result.artifacts != writer.artifacts:
                                raise TypeError("collector result artifacts must exactly match registered artifacts")
                            undeclared_media_types = sorted(
                                {
                                    artifact.media_type
                                    for artifact in result.artifacts
                                    if artifact.media_type not in metadata.output_media_types
                                }
                            )
                            if undeclared_media_types:
                                raise TypeError(f"collector registered undeclared output media types: {', '.join(undeclared_media_types)}")
                            result = collector.finalize(context, result)
                            if not isinstance(result, CollectorResult):
                                raise TypeError("finalize() must return CollectorResult")
                            if result.artifacts != writer.artifacts:
                                raise TypeError("finalized result artifacts must exactly match registered artifacts")
                except CapabilityPolicyError as error:
                    failure_summary = "collector capability policy violation"
                    context.logger.event(
                        "error",
                        "capability_policy_violation",
                        code=error.code,
                        capability=error.capability,
                        operation=error.operation,
                    )
                    lifecycle_errors.extend((str(error), traceback.format_exc()))
                except BaseException as error:
                    lifecycle_errors.extend((f"{type(error).__name__}: {error}", traceback.format_exc()))
                finally:
                    try:
                        collector.cleanup(context)
                    except BaseException as error:
                        if not lifecycle_errors:
                            failure_summary = "collector cleanup failed"
                        lifecycle_errors.extend(
                            (
                                f"collector cleanup failed: {type(error).__name__}: {error}",
                                traceback.format_exc(),
                            )
                        )
                    finally:
                        mutation_guard.active = False
                if lifecycle_errors:
                    result = CollectorResult(
                        CollectorStatus.FAILED,
                        failure_summary,
                        artifacts=writer.artifacts,
                        errors=(*(result.errors if result is not None else ()), *lifecycle_errors),
                    )
                assert result is not None
                context.logger.event(
                    "error" if result.status == CollectorStatus.FAILED else "info",
                    "collection_finished",
                    status=result.status.value,
                )
        result_queue.put({"collector_id": metadata.id, "result": _serialize_result(result)})
    except BaseException as error:  # child processes must always report a terminal result
        result_queue.put(
            {
                "collector_id": payload["collector_id"],
                "result": _serialize_result(
                    CollectorResult(
                        CollectorStatus.FAILED,
                        "collector worker crashed",
                        errors=(f"{type(error).__name__}: {error}", traceback.format_exc()),
                    )
                ),
            }
        )
    finally:
        restore_temporary_directory()


def _serialize_result(result: CollectorResult) -> _SerializedResult:
    """Convert a typed collector result into a queue-safe primitive payload."""
    return {
        "status": result.status.value,
        "summary": result.summary,
        "artifacts": [dict(cast(Mapping[str, object], artifact.to_dict())) for artifact in result.artifacts],
        "errors": list(result.errors),
        "metrics": {
            key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str)) and not isinstance(value, bool)
        },
    }


class RunSupervisor:
    """Plans process isolation, timeouts, manifests, and collector failure containment."""

    def __init__(self, project_root: Path, configuration: AppConfig) -> None:
        """Initialize a supervisor rooted at the project and configured output policy."""
        self.project_root = project_root.resolve()
        self.configuration = configuration
        self._active_workers: dict[str, _ActiveWorker] = {}

    def run(self, plan: RunPlan) -> RunOutcome:
        """Execute a preflighted plan and persist the manifest throughout the run."""
        configured_blocks = set(self.configuration.runtime.blocked_capabilities)
        request_blocks = set(plan.request.blocked_capabilities)
        blocked = configured_blocks.union(request_blocks)
        policy_violations = {
            metadata.id: sorted(capability.value for capability in set(metadata.capabilities).intersection(blocked))
            for metadata in (_require_metadata(candidate) for candidate in plan.collectors)
        }
        policy_violations = {collector_id: capabilities for collector_id, capabilities in policy_violations.items() if capabilities}
        if policy_violations:
            details = "; ".join(
                f"{collector_id}={', '.join(capabilities)}" for collector_id, capabilities in sorted(policy_violations.items())
            )
            raise CapabilityPolicyError(
                "CAPABILITY_BLOCKED",
                "multiple" if len(policy_violations) > 1 else next(iter(policy_violations.values()))[0],
                "run",
                f"collector policy prevents execution: {details}",
            )
        if plan.collectors and not plan.request.acknowledge_authorization:
            selected_metadata = tuple(_require_metadata(candidate) for candidate in plan.collectors)
            categories = sorted(str(metadata.specialty) for metadata in selected_metadata)
            sensitive_outputs = sorted({category for metadata in selected_metadata for category in metadata.sensitive_data_categories})
            raise PermissionError(
                "collection requires --acknowledge-authorization "
                "(acknowledge_authorization=True); "
                f"selected categories: {', '.join(dict.fromkeys(categories))}; "
                f"sensitive outputs: {', '.join(sensitive_outputs) or 'none'}"
            )
        if plan.request.max_workers > self.configuration.runtime.maximum_workers:
            raise ValueError("requested workers exceed configured maximum_workers")
        output_layout = ensure_output_layout(self.configuration.runtime.output_root)
        application_logger = get_application_logger(
            output_layout.application_log,
            self.configuration.logging,
        )
        debug_logging = self.configuration.logging.level.upper() == "DEBUG"
        run_id = f"run-{uuid4().hex}"
        run_directory = output_layout.runs / run_fingerprint(run_id)
        workspace_root = run_directory / "collectors"
        artifact_root = run_directory / "artifacts"
        cancellation_file = run_directory / ".cancelled"
        workspace_root.mkdir(parents=True, exist_ok=False)
        artifact_root.mkdir(parents=True, exist_ok=False)
        run_logger = get_event_logger(run_directory / "logs" / "engine.jsonl", run_id=run_id)
        manifest_path = run_directory / "manifest.json"
        manifest = RunManifest.create(
            run_id,
            asdict(plan.request),
            self.configuration.to_manifest_dict(),
            [(_require_metadata(candidate).id, candidate.path) for candidate in plan.collectors],
            parent_run_id=plan.request.rerun_from,
            plan_fingerprint=plan.fingerprint,
        )
        write_manifest(manifest_path, manifest)
        manifest.status = RunStatus.RUNNING
        write_manifest(manifest_path, manifest)
        run_started_at = monotonic()
        run_logger.event(
            "info",
            "run_started",
            collectors=len(plan.collectors),
            profile=plan.request.profile,
            max_workers=plan.request.max_workers,
            plan_fingerprint=plan.fingerprint,
            output_policy=plan.request.output_policy.value,
        )
        application_logger.event(
            "INFO",
            "run_started",
            source="logicytics.module.runtime",
            collectors=len(plan.collectors),
            profile=plan.request.profile,
        )
        if debug_logging:
            application_logger.event(
                "DEBUG",
                "run_launch_details",
                source="logicytics.module.runtime",
                run_id=run_id,
                max_workers=plan.request.max_workers,
                plan_fingerprint=plan.fingerprint,
                output_policy=plan.request.output_policy.value,
            )

        records = {record.id: record for record in manifest.collectors}
        try:
            self._supervise(
                plan,
                run_id,
                workspace_root,
                artifact_root,
                cancellation_file,
                manifest,
                manifest_path,
                records,
                run_logger,
                application_logger,
            )
        except KeyboardInterrupt:
            cancellation_file.touch()
            run_logger.event("warning", "run_cancellation_requested")
            application_logger.event(
                "WARNING",
                "run_cancellation_requested",
                source="logicytics.module.runtime",
                run_id=run_id,
            )
            self._cancel_records(records, manifest, manifest_path, "run cancelled by user")

        manifest.finalize_status()
        if plan.request.performance_check:
            self._write_performance_report(run_directory, output_layout.performance_logs, manifest)
        should_package = self.configuration.runtime.package_completed_runs and plan.request.output_policy is OutputPolicy.PACKAGE
        if should_package:
            run_logger.event("info", "run_packaging_started")
            application_logger.event(
                "INFO",
                "run_packaging_started",
                source="logicytics.module.runtime",
            )
            try:
                package_path, hash_path = package_manifest(
                    run_directory,
                    manifest,
                    manifest_path,
                    package_directory=output_layout.packages,
                    hash_directory=output_layout.hashes,
                )
                run_logger.event(
                    "info",
                    "run_packaged",
                    package_path=str(package_path),
                    hash_path=str(hash_path),
                )
                application_logger.event(
                    "INFO",
                    "run_packaged",
                    source="logicytics.module.runtime",
                )
                if debug_logging:
                    application_logger.event(
                        "DEBUG",
                        "run_package_details",
                        source="logicytics.module.runtime",
                        run_id=run_id,
                        package_path=str(package_path),
                        hash_path=str(hash_path),
                    )
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                manifest.status = RunStatus.FAILED
                manifest.package = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                run_logger.event("error", "run_packaging_failed", error_type=type(error).__name__)
                application_logger.event(
                    "CRITICAL",
                    "run_packaging_failed",
                    source="logicytics.module.runtime",
                    run_id=run_id,
                    error_type=type(error).__name__,
                )
        else:
            packaging_skip_reason = (
                "manifest_only_output" if plan.request.output_policy is OutputPolicy.MANIFEST_ONLY else "configuration_disabled"
            )
            run_logger.event(
                "info",
                "run_packaging_skipped",
                reason=packaging_skip_reason,
            )
            application_logger.event(
                "INFO",
                "run_packaging_skipped",
                source="logicytics.module.runtime",
                reason=packaging_skip_reason,
            )
        write_manifest(manifest_path, manifest)
        status_counts = {
            status: sum(1 for record in manifest.collectors if record.status == status)
            for status in sorted({record.status for record in manifest.collectors})
        }
        run_logger.event(
            "info",
            "run_finished",
            status=manifest.status.value,
            artifacts=manifest.total_artifact_bytes,
            duration_seconds=round(monotonic() - run_started_at, 3),
            **{f"{status}_count": count for status, count in status_counts.items()},
        )
        application_level = {
            RunStatus.SUCCEEDED: "INFO",
            RunStatus.PARTIAL: "WARNING",
            RunStatus.CANCELLED: "WARNING",
            RunStatus.FAILED: "ERROR",
        }.get(manifest.status, "CRITICAL")
        application_logger.event(
            application_level,
            "run_finished",
            source="logicytics.module.runtime",
            status=manifest.status.value,
            duration_seconds=round(monotonic() - run_started_at, 3),
            **{f"{status}_count": count for status, count in status_counts.items()},
        )
        if debug_logging:
            application_logger.event(
                "DEBUG",
                "run_completion_details",
                source="logicytics.module.runtime",
                run_id=run_id,
                artifacts=manifest.total_artifact_bytes,
                manifest_path=str(manifest_path),
            )
        if plan.request.post_run_action is not PostRunAction.NONE:
            self._execute_post_run_action(plan.request.post_run_action, manifest, run_logger)
            application_logger.event(
                "WARNING",
                "post_run_action_scheduled",
                source="logicytics.module.runtime",
                run_id=run_id,
                action=plan.request.post_run_action.value,
                delay_seconds=60,
            )
        return RunOutcome(manifest=manifest, run_directory=run_directory, manifest_path=manifest_path)

    @staticmethod
    def _execute_post_run_action(
        action: PostRunAction,
        manifest: RunManifest,
        run_logger: FileEventLogger,
    ) -> None:
        """Schedule an explicit Windows power action only after verified packaging."""
        package = manifest.package or {}
        if (
            manifest.status is not RunStatus.SUCCEEDED
            or not isinstance(package.get("path"), str)
            or not isinstance(package.get("sha256"), str)
        ):
            raise LogicyticsError("post-run action requires a successful run and verified package")
        flag = "/r" if action is PostRunAction.REBOOT else "/s"
        try:
            completed = process_adapter.run(
                ["shutdown", flag, "/t", "60", "/d", "p:0:0", "/c", "Logicytics run completed"],
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError as error:
            raise LogicyticsError(f"unable to schedule {action.value}: {error}") from error
        if completed.returncode != 0:
            error_detail = completed.stderr.strip() or completed.stdout.strip()
            raise LogicyticsError(f"unable to schedule {action.value}: {error_detail}")
        run_logger.event("warning", "post_run_action_scheduled", action=action.value, delay_seconds=60)

    @staticmethod
    def _write_performance_report(run_directory: Path, performance_logs: Path, manifest: RunManifest) -> None:
        """Publish packaged timing evidence and the global readable performance log."""
        performance_path = run_directory / "logs" / "performance.json"
        payload = {
            "run_id": manifest.run_id,
            "collectors": [
                {
                    "id": record.id,
                    "status": record.status,
                    "duration_seconds": record.duration_seconds,
                    "peak_memory_bytes": record.peak_memory_bytes,
                    "progress": record.progress,
                }
                for record in manifest.collectors
            ],
        }
        performance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fingerprint = run_fingerprint(manifest.run_id)
        log_path = performance_logs / f"{fingerprint[:8]}.log"
        rows = [
            "Performance report",
            f"Run id: {manifest.run_id}",
            f"Run fingerprint: {fingerprint}",
            f"Run status: {manifest.status.value}",
            "",
            "Collector timings",
            "-----------------",
        ]
        for record in manifest.collectors:
            duration = "unavailable" if record.duration_seconds is None else f"{record.duration_seconds:.3f}s"
            memory = "unavailable" if record.peak_memory_bytes is None else str(record.peak_memory_bytes)
            rows.append(f"{record.id}: {duration}; status={record.status}; peak_memory_bytes={memory}")
        log_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _supervise(
        self,
        plan: RunPlan,
        run_id: str,
        workspace_root: Path,
        artifact_root: Path,
        cancellation_file: Path,
        manifest: RunManifest,
        manifest_path: Path,
        records: dict[str, CollectorRecord],
        run_logger: FileEventLogger,
        application_logger: EventLogger,
    ) -> None:
        """Schedule bounded isolated workers and contain each terminal failure."""
        pending = list(plan.collectors)
        candidates = {_require_metadata(candidate).id: candidate for candidate in plan.collectors}
        active: dict[str, _ActiveWorker] = {}
        retry_not_before: dict[str, float] = {}
        self._active_workers = active

        spawn_context = cast(
            _ProcessContext,
            cast(object, multiprocessing.get_context("spawn")),
        )

        result_queue: Queue[_WorkerMessage] = spawn_context.Queue()
        worker_limit = min(plan.request.max_workers, self.configuration.runtime.maximum_workers)
        committed_output_bytes = 0
        while pending or active:
            if cancellation_file.exists():
                self._cancel_active(active, records, manifest, manifest_path, "run cancellation requested")
                for candidate in pending:
                    metadata = _require_metadata(candidate)
                    records[metadata.id].apply_result(CollectorResult(CollectorStatus.CANCELLED, "not started because run was cancelled"))
                return
            while pending and len(active) < worker_limit:
                if any(not worker.parallel_safe for worker in active.values()):
                    break
                candidate = pending[0]
                metadata = _require_metadata(candidate)
                if monotonic() < retry_not_before.get(metadata.id, 0):
                    break
                dependency_states = {dependency: records[dependency].status for dependency in metadata.dependencies}
                failed_dependencies = {
                    dependency: status
                    for dependency, status in dependency_states.items()
                    if status in {"partial", "skipped", "cancelled", "failed"}
                }
                if failed_dependencies:
                    pending.pop(0)
                    details = ", ".join(f"{dependency}={status}" for dependency, status in sorted(failed_dependencies.items()))
                    records[metadata.id].apply_result(
                        CollectorResult(
                            CollectorStatus.SKIPPED,
                            "collector dependency was not satisfied",
                            errors=(f"dependency did not succeed: {details}",),
                        )
                    )
                    run_logger.event(
                        "warning",
                        "collector_dependency_unsatisfied",
                        collector_id=metadata.id,
                    )
                    write_manifest(manifest_path, manifest)
                    continue

                if any(status != "succeeded" for status in dependency_states.values()):
                    break

                if active and not metadata.parallel_safe:
                    break

                if active:
                    candidate_class = metadata.resource_class

                    has_interactive_worker = any(worker.resource_class is ResourceClass.INTERACTIVE for worker in active.values())

                    has_same_resource_class = any(worker.resource_class is candidate_class for worker in active.values())

                    conflicts_with_active_worker = (
                        candidate_class is ResourceClass.INTERACTIVE
                        or has_interactive_worker
                        or (candidate_class is not ResourceClass.GENERAL and has_same_resource_class)
                    )

                    if conflicts_with_active_worker:
                        break

                reserved_output_bytes = sum(worker.reserved_output_bytes for worker in active.values())
                remaining_output_bytes = (
                    self.configuration.runtime.maximum_run_output_bytes - committed_output_bytes - reserved_output_bytes
                )
                if remaining_output_bytes < 1:
                    if active:
                        break
                    pending.pop(0)
                    records[metadata.id].apply_result(
                        CollectorResult(
                            CollectorStatus.SKIPPED,
                            "run output limit reached before collector could start",
                            errors=("configured maximum_run_output_bytes was exhausted",),
                        )
                    )
                    run_logger.event(
                        "warning",
                        "collector_run_output_limit_reached",
                        collector_id=metadata.id,
                    )
                    write_manifest(manifest_path, manifest)
                    continue
                output_budget = min(metadata.maximum_output_bytes, remaining_output_bytes)
                pending.pop(0)
                retry_not_before.pop(metadata.id, None)
                workspace = workspace_root / metadata.id.replace(".", "_")
                payload: _WorkerPayload = {
                    "run_id": run_id,
                    "collector_id": metadata.id,
                    "path": str(candidate.path),
                    "expected_class": candidate.expected_class,
                    "execution_type": candidate.execution_type,
                    "metadata": dict(cast(Mapping[str, object], metadata.to_dict())),
                    "blocked_capabilities": plan.request.blocked_capabilities,
                    "workspace": str(workspace),
                    "artifact_root": str(artifact_root),
                    "cancellation_file": str(cancellation_file),
                    "settings": dict(self.configuration.settings_for(metadata.id)),
                    "run_output_budget_bytes": output_budget,
                }
                process = spawn_context.Process(
                    target=_worker_entry,
                    args=(payload, result_queue),
                    name=f"Logicytics-{metadata.id}",
                )
                records[metadata.id].status = "running"
                records[metadata.id].started_at = utc_now()
                records[metadata.id].heartbeat_at = utc_now()
                records[metadata.id].attempt_count += 1
                process.start()
                records[metadata.id].worker_pid = process.pid
                active[metadata.id] = _ActiveWorker(
                    metadata.id,
                    process,
                    monotonic(),
                    metadata.timeout_seconds,
                    metadata.maximum_memory_bytes,
                    workspace,
                    metadata.parallel_safe,
                    metadata.resource_class,
                    output_budget,
                )
                start_fields = {
                    "collector_id": metadata.id,
                    "attempt": records[metadata.id].attempt_count,
                    "worker_pid": process.pid or 0,
                    "timeout_seconds": metadata.timeout_seconds,
                    "maximum_memory_bytes": metadata.maximum_memory_bytes,
                    "output_budget_bytes": output_budget,
                    "resource_class": metadata.resource_class.value,
                    "parallel_safe": metadata.parallel_safe,
                    "declared_capabilities": ",".join(sorted(capability.value for capability in metadata.capabilities)) or "none",
                }
                run_logger.event("info", "collector_started", **start_fields)
                application_logger.event(
                    "INFO",
                    "collector_started",
                    source="logicytics.module.runtime",
                    collector_id=metadata.id,
                    attempt=records[metadata.id].attempt_count,
                )
                if self.configuration.logging.level.upper() == "DEBUG":
                    application_logger.event(
                        "DEBUG",
                        "collector_launch_details",
                        source="logicytics.module.runtime",
                        run_id=run_id,
                        **start_fields,
                    )
                write_manifest(manifest_path, manifest)
                if not metadata.parallel_safe:
                    break

            try:
                message = result_queue.get(timeout=0.1)
            except queue.Empty:
                message = None
            if message is not None:
                collector_id = message["collector_id"]
                worker = active.pop(collector_id, None)
                if worker is not None:
                    worker.process.join(timeout=1)
                    record = records[collector_id]
                    self._apply_worker_result(record, _result_from_dict(message["result"]), worker)
                    artifact_bytes = self._collector_artifact_bytes(artifact_root, collector_id)
                    committed_output_bytes += artifact_bytes
                    self._cleanup_worker_temporary_directory(worker)
                    capability_error = next(
                        (error for error in record.errors if error.startswith(("CAPABILITY_BLOCKED:", "CAPABILITY_DECLARATION_MISMATCH:"))),
                        None,
                    )
                    if capability_error is not None:
                        application_logger.event(
                            "ERROR",
                            "capability_policy_violation",
                            source="logicytics.module.runtime",
                            run_id=run_id,
                            collector_id=collector_id,
                            code=capability_error.split(":", 1)[0],
                            detail=capability_error,
                        )
                    attempt_status = record.status
                    attempt_duration = record.duration_seconds or 0.0
                    retry_scheduled = self._schedule_retry(
                        candidates[collector_id],
                        records[collector_id],
                        artifact_bytes,
                        pending,
                        retry_not_before,
                        cancellation_file,
                        run_logger,
                    )
                    finish_level = "warning" if retry_scheduled else "info"
                    finish_fields = {
                        "collector_id": collector_id,
                        "status": record.status,
                        "attempt_status": attempt_status,
                        "attempt": record.attempt_count,
                        "duration_seconds": record.duration_seconds or 0.0,
                        "attempt_duration_seconds": attempt_duration,
                        "artifact_count": len(record.artifacts),
                        "artifact_bytes": artifact_bytes,
                        "event_count": record.event_count,
                        "peak_memory_bytes": record.peak_memory_bytes,
                        "worker_exit_code": worker.process.exitcode or 0,
                        "termination_reason": record.termination_reason or "unknown",
                        "retry_scheduled": retry_scheduled,
                    }
                    run_logger.event(finish_level, "collector_finished", **finish_fields)
                    application_level = (
                        "ERROR"
                        if record.status == CollectorStatus.FAILED
                        else "WARNING"
                        if retry_scheduled or record.status in {CollectorStatus.CANCELLED, CollectorStatus.SKIPPED}
                        else "INFO"
                    )
                    application_logger.event(
                        application_level,
                        "collector_finished",
                        source="logicytics.module.runtime",
                        collector_id=collector_id,
                        status=record.status,
                        duration_seconds=record.duration_seconds or 0.0,
                        summary=record.summary or "not-finished",
                    )
                    if self.configuration.logging.level.upper() == "DEBUG":
                        application_logger.event(
                            "DEBUG",
                            "collector_completion_details",
                            source="logicytics.module.runtime",
                            run_id=run_id,
                            **finish_fields,
                        )
                    write_manifest(manifest_path, manifest)

            for collector_id, worker in tuple(active.items()):
                if self._refresh_worker_progress(records[collector_id], worker):
                    write_manifest(manifest_path, manifest)
                memory_bytes = self._worker_memory_bytes(worker.process)
                if memory_bytes is not None:
                    records[collector_id].peak_memory_bytes = max(
                        records[collector_id].peak_memory_bytes,
                        memory_bytes,
                    )
                if memory_bytes is not None and memory_bytes > worker.maximum_memory_bytes:
                    self._terminate_process_tree(worker.process)
                    active.pop(collector_id)
                    records[collector_id].termination_reason = "memory_limit_exceeded"
                    self._apply_worker_result(
                        records[collector_id],
                        CollectorResult(
                            CollectorStatus.FAILED,
                            "collector exceeded its declared memory limit",
                            errors=(f"collector working set {memory_bytes} exceeded maximum_memory_bytes={worker.maximum_memory_bytes}",),
                        ),
                        worker,
                    )
                    artifact_bytes = self._collector_artifact_bytes(artifact_root, collector_id)
                    committed_output_bytes += artifact_bytes
                    self._cleanup_worker_temporary_directory(worker)
                    if not self._schedule_retry(
                        candidates[collector_id],
                        records[collector_id],
                        artifact_bytes,
                        pending,
                        retry_not_before,
                        cancellation_file,
                        run_logger,
                    ):
                        run_logger.event("error", "collector_memory_limit_exceeded", collector_id=collector_id)
                    write_manifest(manifest_path, manifest)
                elif monotonic() - worker.started_at > worker.timeout_seconds:
                    self._terminate_process_tree(worker.process)
                    active.pop(collector_id)
                    records[collector_id].termination_reason = "timeout_exceeded"
                    self._apply_worker_result(
                        records[collector_id],
                        CollectorResult(
                            CollectorStatus.FAILED,
                            "collector exceeded its declared timeout",
                            errors=(f"collector exceeded its {worker.timeout_seconds}-second timeout",),
                        ),
                        worker,
                    )
                    artifact_bytes = self._collector_artifact_bytes(artifact_root, collector_id)
                    committed_output_bytes += artifact_bytes
                    self._cleanup_worker_temporary_directory(worker)
                    if not self._schedule_retry(
                        candidates[collector_id],
                        records[collector_id],
                        artifact_bytes,
                        pending,
                        retry_not_before,
                        cancellation_file,
                        run_logger,
                    ):
                        run_logger.event("error", "collector_timed_out", collector_id=collector_id)
                    write_manifest(manifest_path, manifest)
                elif not worker.process.is_alive():
                    worker.process.join(timeout=1)
                    if worker.exited_at is None:
                        worker.exited_at = monotonic()
                        continue
                    if monotonic() - worker.exited_at < 1.0:
                        continue
                    active.pop(collector_id)
                    records[collector_id].termination_reason = "exited_without_result"
                    self._apply_worker_result(
                        records[collector_id],
                        CollectorResult(CollectorStatus.FAILED, "collector exited without a result"),
                        worker,
                    )
                    artifact_bytes = self._collector_artifact_bytes(artifact_root, collector_id)
                    committed_output_bytes += artifact_bytes
                    self._cleanup_worker_temporary_directory(worker)
                    if not self._schedule_retry(
                        candidates[collector_id],
                        records[collector_id],
                        artifact_bytes,
                        pending,
                        retry_not_before,
                        cancellation_file,
                        run_logger,
                    ):
                        run_logger.event("error", "collector_exited_without_result", collector_id=collector_id)
                    write_manifest(manifest_path, manifest)
            sleep(0.01)

    @staticmethod
    def _worker_memory_bytes(process: BaseProcess) -> int | None:
        """Return one worker's resident working set using local OS facilities."""
        pid = process.pid
        if pid is None:
            return None
        return process_adapter.memory_bytes(pid)

    @staticmethod
    def _schedule_retry(
        candidate: CollectorCandidate,
        record: CollectorRecord,
        artifact_bytes: int,
        pending: list[CollectorCandidate],
        retry_not_before: dict[str, float],
        cancellation_file: Path,
        run_logger: FileEventLogger,
    ) -> bool:
        """Retry only explicitly permitted failed attempts that produced no evidence."""
        metadata = _require_metadata(candidate)
        if (
            record.status != CollectorStatus.FAILED.value
            or record.attempt_count > metadata.maximum_retries
            or artifact_bytes != 0
            or cancellation_file.exists()
        ):
            return False
        record.retry_history.append(
            {
                "attempt": record.attempt_count,
                "status": record.status,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "summary": record.summary,
                "errors": list(record.errors),
                "failure": dict(record.failure) if isinstance(record.failure, Mapping) else None,
                "duration_seconds": record.duration_seconds,
                "worker_pid": record.worker_pid,
                "worker_exit_code": record.worker_exit_code,
                "termination_reason": record.termination_reason,
            }
        )
        record.status = "retry_pending"
        record.summary = "collector retry scheduled"
        record.errors = []
        record.failure = None
        record.artifacts = []
        record.started_at = None
        record.finished_at = None
        record.duration_seconds = None
        record.worker_pid = None
        record.worker_exit_code = None
        record.termination_reason = None
        retry_not_before[record.id] = monotonic() + metadata.retry_delay_seconds
        pending.insert(0, candidate)
        run_logger.event(
            "warning",
            "collector_retry_scheduled",
            collector_id=record.id,
            attempt=record.attempt_count + 1,
        )
        return True

    @staticmethod
    def _collector_artifact_bytes(artifact_root: Path, collector_id: str) -> int:
        """Count run-owned evidence, including files left by a failed collector."""
        directory = artifact_root / collector_id.replace(".", "_")
        if not directory.is_dir():
            return 0
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

    @staticmethod
    def _cleanup_worker_temporary_directory(worker: _ActiveWorker) -> None:
        """Remove only a terminal worker's private scratch directory, never its logs."""
        temporary_directory = worker.workspace / "tmp"
        try:
            shutil.rmtree(temporary_directory, ignore_errors=True)
        except OSError:
            pass

    @staticmethod
    def _windows_descendants(parent_pid: int) -> tuple[int, ...]:
        """Snapshot only descendants belonging to one isolated Windows worker."""
        return windows_api_adapter.process_descendants(parent_pid)

    @staticmethod
    def _terminate_process_tree(process: BaseProcess) -> None:
        """Terminate one worker-owned process tree without touching peer workers."""
        pid = process.pid
        if pid is not None:
            try:
                if os.name == "nt":
                    descendants = RunSupervisor._windows_descendants(pid)
                    process_adapter.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=5,
                    )
                    for child_pid in descendants:
                        windows_api_adapter.terminate_process(child_pid)
                else:
                    process_adapter.terminate_process_group(pid)
            except (OSError, TimeoutExpired):
                pass
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)

    @staticmethod
    def _refresh_worker_progress(record: CollectorRecord, worker: _ActiveWorker) -> bool:
        """Persist worker liveness and any newly written structured progress events."""
        changed = False
        now = monotonic()
        elapsed = round(now - worker.started_at, 3)
        if elapsed > record.progress["elapsed_seconds"]:
            record.progress["elapsed_seconds"] = elapsed
            changed = True
        if now - worker.last_heartbeat_at >= 1:
            record.heartbeat_at = utc_now()
            worker.last_heartbeat_at = now
            changed = True
        events_path = worker.workspace / "events.jsonl"
        if not events_path.exists():
            return changed
        try:
            with events_path.open("r", encoding="utf-8") as stream:
                stream.seek(worker.last_event_offset)
                events: list[str] = []
                while line := stream.readline():
                    if not line.endswith("\n"):
                        break
                    events.append(line)
                    worker.last_event_offset = stream.tell()
        except OSError:
            return changed
        if not events:
            return changed
        worker.last_event_count += len(events)
        record.event_count = worker.last_event_count
        aliases = {
            "files_scanned": ("files_scanned", "scanned_files", "entry_count"),
            "files_copied": ("files_copied", "copied_files"),
            "bytes_written": ("bytes_written",),
            "packets_observed": ("packets_observed", "observation_count", "packet_count"),
            "events_processed": ("events_processed", "event_count"),
        }
        for line in events:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = event.get("at")
            if isinstance(timestamp, str):
                record.last_progress_at = timestamp
            fields = event.get("fields", {})
            if not isinstance(fields, dict):
                continue
            for name, alternatives in aliases.items():
                for alternative in alternatives:
                    value = fields.get(alternative)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                        record.progress[name] = max(record.progress[name], value)
            changed = True
        return changed

    @staticmethod
    def _apply_worker_result(record: CollectorRecord, result: CollectorResult, worker: _ActiveWorker) -> None:
        """Attach result, elapsed time, and progress accounting to one record."""
        record.apply_result(result, duration_seconds=round(monotonic() - worker.started_at, 3))
        record.progress["bytes_written"] = max(
            record.progress["bytes_written"],
            sum(artifact.size_bytes for artifact in result.artifacts),
        )
        record.progress["files_copied"] = max(record.progress["files_copied"], len(result.artifacts))
        record.progress["elapsed_seconds"] = max(record.progress["elapsed_seconds"], record.duration_seconds or 0.0)
        record.worker_pid = worker.process.pid
        record.worker_exit_code = worker.process.exitcode
        if record.termination_reason is None:
            record.termination_reason = "completed" if result.status == CollectorStatus.SUCCEEDED else result.status.value
        RunSupervisor._refresh_worker_progress(record, worker)

    def _cancel_active(
        self,
        active: dict[str, _ActiveWorker],
        records: dict[str, CollectorRecord],
        manifest: RunManifest,
        manifest_path: Path,
        reason: str,
    ) -> None:
        """Terminate active workers, mark their records cancelled, and persist the manifest."""
        for collector_id, worker in tuple(active.items()):
            self._terminate_process_tree(worker.process)
            active.pop(collector_id)
            records[collector_id].termination_reason = "cancelled"
            self._apply_worker_result(
                records[collector_id],
                CollectorResult(CollectorStatus.CANCELLED, reason),
                worker,
            )
            self._cleanup_worker_temporary_directory(worker)
        write_manifest(manifest_path, manifest)

    def _cancel_records(
        self,
        records: dict[str, CollectorRecord],
        manifest: RunManifest,
        manifest_path: Path,
        reason: str,
    ) -> None:
        """Cancel active and still-planned records, then write the final cancellation state."""
        self._cancel_active(self._active_workers, records, manifest, manifest_path, reason)
        for record in records.values():
            if record.status in {"planned", "running"}:
                record.apply_result(CollectorResult(CollectorStatus.CANCELLED, reason))
        write_manifest(manifest_path, manifest)
