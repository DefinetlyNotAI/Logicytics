"""Per-collector process supervisor and run workspace lifecycle."""

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import json
import multiprocessing
import os
import queue
import signal
import shutil
import socket
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

from logicytics.artifacts import WorkspaceArtifactWriter
from logicytics.configuration import AppConfig
from logicytics.contracts import (
    Artifact,
    Capability,
    CollectorContext,
    CollectorResult,
    CollectorStatus,
    ResourceClass,
    RunStatus,
    ValidationResult,
)
from logicytics.discovery import CollectorCandidate
from logicytics.logging import FileEventLogger
from logicytics.manifest import CollectorRecord, RunManifest, write_manifest, utc_now
from logicytics.packaging import package_manifest
from logicytics.planner import RunPlan


@dataclass(slots=True)
class RunOutcome:
    """The completed run and the durable locations that describe it."""

    manifest: RunManifest
    run_directory: Path
    manifest_path: Path


@dataclass(slots=True)
class _ActiveWorker:
    candidate_id: str
    process: multiprocessing.Process
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


class _WorkerMutationGuard:
    """Constrain audited collector mutations to its private workspace and evidence store."""

    _PATH_EVENTS = {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown", "os.utime", "os.truncate"}
    _DOUBLE_PATH_EVENTS = {"os.rename", "os.link", "os.symlink"}
    _BLOCKED_EVENTS = {"os.chdir", "os.putenv", "os.unsetenv", "os.system"}

    def __init__(
            self,
            workspace: Path,
            artifact_root: Path,
            collector_id: str,
            capabilities: tuple[Capability, ...],
            collector_source: Path,
    ) -> None:
        self.roots = (workspace.resolve(), (artifact_root / collector_id.replace(".", "_")).resolve())
        self.runtime_roots = (Path(sys.base_prefix).resolve(), Path(__file__).resolve().parent)
        self.collector_source = collector_source.resolve()
        self.capabilities = frozenset(capabilities)
        self.active = False
        sys.addaudithook(self._check_event)

    def _check_path(self, value: object) -> None:
        if isinstance(value, int):
            return
        if not isinstance(value, (str, bytes, os.PathLike)):
            raise PermissionError("collector filesystem mutation has an unsupported target")
        path = Path(os.fsdecode(value)).resolve()
        if not any(path == root or root in path.parents for root in self.roots):
            raise PermissionError(f"collector filesystem mutation escapes its private workspace: {path}")

    def _check_read(self, value: object) -> None:
        if isinstance(value, int) or not isinstance(value, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(value)).resolve()
        if path == self.collector_source or any(
                path == root or root in path.parents for root in (*self.roots, *self.runtime_roots)
        ):
            return
        if Capability.FILESYSTEM_READ not in self.capabilities:
            raise PermissionError("collector requires declared filesystem_read capability")
        components = tuple(part.casefold() for part in path.parts)
        browser_data = any(part in {"chrome", "edge", "firefox", "opera software", "opera gx"} for part in components)
        private_key = ".ssh" in components or path.name.casefold() in {
            "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"
        } or path.suffix.casefold() in {".pem", ".ppk"}
        sensitive = private_key or browser_data or any(
            label in path.name.casefold() for label in ("cookie", "credential", "password", "token", "secret", "login data")
        )
        if browser_data and Capability.BROWSER_DATA not in self.capabilities:
            raise PermissionError("collector requires declared browser_data capability")
        if sensitive and Capability.SENSITIVE_FILES not in self.capabilities:
            raise PermissionError("collector requires declared sensitive_files capability")
        if private_key and Capability.PRIVATE_KEYS not in self.capabilities:
            raise PermissionError("collector requires declared private_keys capability")

    def _check_event(self, event: str, arguments: tuple[object, ...]) -> None:
        if not self.active:
            return
        if event == "subprocess.Popen" and Capability.SUBPROCESS not in self.capabilities:
            raise PermissionError("collector requires declared subprocess capability")
        if event.startswith("socket.") and event in {
            "socket.__new__", "socket.bind", "socket.connect", "socket.sendto", "socket.getaddrinfo"
        }:
            if Capability.NETWORK not in self.capabilities:
                raise PermissionError("collector requires declared network capability")
            if event == "socket.__new__" and len(arguments) > 2 and int(arguments[2]) == int(socket.SOCK_RAW):
                if Capability.PACKET_CAPTURE not in self.capabilities:
                    raise PermissionError("collector requires declared packet_capture capability")
        if event.startswith("winreg.") and Capability.REGISTRY_READ not in self.capabilities:
            raise PermissionError("collector requires declared registry_read capability")
        if event in self._BLOCKED_EVENTS:
            raise PermissionError(f"collector must not modify process state: {event}")
        if event == "open":
            mode = arguments[1] if len(arguments) > 1 else None
            flags = arguments[2] if len(arguments) > 2 else 0
            writing = isinstance(mode, str) and any(flag in mode for flag in "wax+")
            writing = writing or isinstance(flags, int) and bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
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
    module_name = f"logicytics_runtime_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to load collector module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, expected_class)()


def _artifact_from_dict(data: dict[str, object]) -> Artifact:
    return Artifact(
        id=str(data["id"]),
        relative_path=str(data["relative_path"]),
        sha256=str(data["sha256"]),
        size_bytes=int(data["size_bytes"]),
        media_type=str(data["media_type"]),
        collector_id=str(data["collector_id"]),
        source_category=str(data["source_category"]),
        collected_at=str(data["collected_at"]),
        transformations=tuple(str(step) for step in data["transformations"]),
    )


def _result_from_dict(data: dict[str, object]) -> CollectorResult:
    return CollectorResult(
        status=CollectorStatus(str(data["status"])),
        summary=str(data["summary"]),
        artifacts=tuple(_artifact_from_dict(item) for item in data.get("artifacts", [])),
        errors=tuple(str(error) for error in data.get("errors", [])),
        metrics=dict(data.get("metrics", {})),
    )


def _worker_entry(payload: dict[str, object], result_queue: multiprocessing.Queue) -> None:
    """Run a single collector in an isolated child process."""
    if os.name != "nt":
        os.setsid()
    workspace = Path(str(payload["workspace"]))
    workspace.mkdir(parents=True, exist_ok=True)
    temporary_directory = workspace / "tmp"
    temporary_directory.mkdir(exist_ok=True)
    stdout_path = workspace / "stdout.log"
    stderr_path = workspace / "stderr.log"
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                collector = _load_collector(Path(str(payload["path"])), str(payload["expected_class"]))
                metadata = collector.metadata()
                writer = WorkspaceArtifactWriter(
                    metadata.id,
                    workspace,
                    Path(str(payload["artifact_root"])),
                    metadata.maximum_output_bytes,
                    metadata.maximum_artifact_files,
                    source_category=(
                        metadata.specialty.value
                        if hasattr(metadata.specialty, "value")
                        else metadata.specialty
                    ),
                    maximum_artifact_bytes=metadata.maximum_artifact_bytes,
                    run_output_budget_bytes=int(payload["run_output_budget_bytes"]),
                    cancellation_file=Path(str(payload["cancellation_file"])),
                )
                context = CollectorContext(
                    run_id=str(payload["run_id"]),
                    collector_id=metadata.id,
                    workspace=workspace,
                    temporary_directory=temporary_directory,
                    artifacts=writer,
                    logger=FileEventLogger(
                        workspace / "events.jsonl",
                        run_id=str(payload["run_id"]),
                        collector_id=metadata.id,
                    ),
                    settings=dict(payload["settings"]),
                    cancellation_file=Path(str(payload["cancellation_file"])),
                )
                result: CollectorResult | None = None
                lifecycle_errors: list[str] = []
                failure_summary = "collector worker crashed"
                mutation_guard = _WorkerMutationGuard(
                    workspace,
                    Path(str(payload["artifact_root"])),
                    metadata.id,
                    metadata.capabilities,
                    Path(str(payload["path"])),
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
                        context.logger.event("info", "collection_started")
                        result = collector.collect(context)
                        if not isinstance(result, CollectorResult):
                            raise TypeError("collect() must return CollectorResult")
                        if result.artifacts != writer.artifacts:
                            raise TypeError("collector result artifacts must exactly match registered artifacts")
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
                "collector_id": str(payload["collector_id"]),
                "result": _serialize_result(
                    CollectorResult(
                        CollectorStatus.FAILED,
                        "collector worker crashed",
                        errors=(f"{type(error).__name__}: {error}", traceback.format_exc()),
                    )
                ),
            }
        )


def _serialize_result(result: CollectorResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "summary": result.summary,
        "artifacts": [artifact.to_dict() for artifact in result.artifacts],
        "errors": list(result.errors),
        "metrics": dict(result.metrics),
    }


class RunSupervisor:
    """Plans process isolation, timeouts, manifests, and collector failure containment."""

    def __init__(self, project_root: Path, configuration: AppConfig) -> None:
        self.project_root = project_root.resolve()
        self.configuration = configuration
        self._active_workers: dict[str, _ActiveWorker] = {}

    def run(self, plan: RunPlan) -> RunOutcome:
        """Execute a preflighted plan and persist the manifest throughout the run."""
        if plan.collectors and not plan.request.acknowledge_authorization:
            categories = sorted(
                candidate.metadata.specialty.value
                if hasattr(candidate.metadata.specialty, "value")
                else candidate.metadata.specialty
                for candidate in plan.collectors
                if candidate.metadata is not None
            )
            sensitive_outputs = sorted(
                {
                    category
                    for candidate in plan.collectors
                    if candidate.metadata is not None
                    for category in candidate.metadata.sensitive_data_categories
                }
            )
            raise PermissionError(
                "collection requires --acknowledge-authorization "
                "(acknowledge_authorization=True); "
                f"selected categories: {', '.join(dict.fromkeys(categories))}; "
                f"sensitive outputs: {', '.join(sensitive_outputs) or 'none'}"
            )
        if plan.request.max_workers > self.configuration.runtime.maximum_workers:
            raise ValueError("requested workers exceed configured maximum_workers")
        run_id = f"run-{uuid4().hex}"
        run_directory = self.configuration.runtime.output_root / run_id
        workspace_root = run_directory / "collectors"
        artifact_root = run_directory / "artifacts"
        cancellation_file = run_directory / ".cancelled"
        workspace_root.mkdir(parents=True, exist_ok=False)
        artifact_root.mkdir(parents=True, exist_ok=False)
        run_logger = FileEventLogger(run_directory / "logs" / "engine.jsonl", run_id=run_id)
        manifest_path = run_directory / "manifest.json"
        manifest = RunManifest.create(
            run_id,
            asdict(plan.request),
            self.configuration.to_manifest_dict(),
            [(candidate.metadata.id, candidate.path) for candidate in plan.collectors if candidate.metadata],
            parent_run_id=plan.request.rerun_from,
        )
        write_manifest(manifest_path, manifest)
        manifest.status = RunStatus.RUNNING
        write_manifest(manifest_path, manifest)
        run_logger.event("info", "run_started", collectors=len(plan.collectors))

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
            )
        except KeyboardInterrupt:
            cancellation_file.touch()
            run_logger.event("warning", "run_cancellation_requested")
            self._cancel_records(records, manifest, manifest_path, "run cancelled by user")

        manifest.finalize_status()
        if plan.request.performance_check:
            self._write_performance_report(run_directory, manifest)
        if self.configuration.runtime.package_completed_runs:
            try:
                package_path, hash_path = package_manifest(run_directory, manifest, manifest_path)
                run_logger.event(
                    "info",
                    "run_packaged",
                    package_path=str(package_path),
                    hash_path=str(hash_path),
                )
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                manifest.status = RunStatus.FAILED
                manifest.package = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                run_logger.event("error", "run_packaging_failed", error_type=type(error).__name__)
        write_manifest(manifest_path, manifest)
        run_logger.event("info", "run_finished", status=manifest.status.value, artifacts=manifest.total_artifact_bytes)
        return RunOutcome(manifest=manifest, run_directory=run_directory, manifest_path=manifest_path)

    @staticmethod
    def _write_performance_report(run_directory: Path, manifest: RunManifest) -> None:
        """Finalize run-owned timing diagnostics before evidence packaging begins."""
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
    ) -> None:
        """Schedule bounded isolated workers and contain each terminal failure."""
        pending = list(plan.collectors)
        candidates = {
            candidate.metadata.id: candidate
            for candidate in plan.collectors
            if candidate.metadata is not None
        }
        active: dict[str, _ActiveWorker] = {}
        retry_not_before: dict[str, float] = {}
        self._active_workers = active
        result_queue: multiprocessing.Queue = multiprocessing.get_context("spawn").Queue()
        worker_limit = min(plan.request.max_workers, self.configuration.runtime.maximum_workers)
        committed_output_bytes = 0
        while pending or active:
            if cancellation_file.exists():
                self._cancel_active(active, records, manifest, manifest_path, "run cancellation requested")
                for candidate in pending:
                    assert candidate.metadata is not None
                    records[candidate.metadata.id].apply_result(
                        CollectorResult(CollectorStatus.CANCELLED, "not started because run was cancelled")
                    )
                return
            while pending and len(active) < worker_limit:
                if any(not worker.parallel_safe for worker in active.values()):
                    break
                candidate = pending[0]
                assert candidate.metadata is not None
                if monotonic() < retry_not_before.get(candidate.metadata.id, 0):
                    break
                dependency_states = {
                    dependency: records[dependency].status
                    for dependency in candidate.metadata.dependencies
                }
                failed_dependencies = {
                    dependency: status
                    for dependency, status in dependency_states.items()
                    if status in {"partial", "skipped", "cancelled", "failed"}
                }
                if failed_dependencies:
                    pending.pop(0)
                    details = ", ".join(
                        f"{dependency}={status}"
                        for dependency, status in sorted(failed_dependencies.items())
                    )
                    records[candidate.metadata.id].apply_result(
                        CollectorResult(
                            CollectorStatus.SKIPPED,
                            "collector dependency was not satisfied",
                            errors=(f"dependency did not succeed: {details}",),
                        )
                    )
                    run_logger.event(
                        "warning",
                        "collector_dependency_unsatisfied",
                        collector_id=candidate.metadata.id,
                    )
                    write_manifest(manifest_path, manifest)
                    continue
                if any(status != "succeeded" for status in dependency_states.values()):
                    break
                if active and not candidate.metadata.parallel_safe:
                    break
                if active and (
                        candidate.metadata.resource_class is ResourceClass.INTERACTIVE
                        or any(worker.resource_class is ResourceClass.INTERACTIVE for worker in active.values())
                        or (
                            candidate.metadata.resource_class is not ResourceClass.GENERAL
                            and any(
                                worker.resource_class is candidate.metadata.resource_class
                                for worker in active.values()
                            )
                        )
                ):
                    break
                reserved_output_bytes = sum(worker.reserved_output_bytes for worker in active.values())
                remaining_output_bytes = (
                    self.configuration.runtime.maximum_run_output_bytes
                    - committed_output_bytes
                    - reserved_output_bytes
                )
                if remaining_output_bytes < 1:
                    if active:
                        break
                    pending.pop(0)
                    records[candidate.metadata.id].apply_result(
                        CollectorResult(
                            CollectorStatus.SKIPPED,
                            "run output limit reached before collector could start",
                            errors=("configured maximum_run_output_bytes was exhausted",),
                        )
                    )
                    run_logger.event(
                        "warning",
                        "collector_run_output_limit_reached",
                        collector_id=candidate.metadata.id,
                    )
                    write_manifest(manifest_path, manifest)
                    continue
                output_budget = min(candidate.metadata.maximum_output_bytes, remaining_output_bytes)
                pending.pop(0)
                retry_not_before.pop(candidate.metadata.id, None)
                workspace = workspace_root / candidate.metadata.id.replace(".", "_")
                payload: dict[str, object] = {
                    "run_id": run_id,
                    "collector_id": candidate.metadata.id,
                    "path": str(candidate.path),
                    "expected_class": candidate.expected_class,
                    "workspace": str(workspace),
                    "artifact_root": str(artifact_root),
                    "cancellation_file": str(cancellation_file),
                    "settings": dict(self.configuration.settings_for(candidate.metadata.id)),
                    "run_output_budget_bytes": output_budget,
                }
                process = multiprocessing.get_context("spawn").Process(
                    target=_worker_entry,
                    args=(payload, result_queue),
                    name=f"Logicytics-{candidate.metadata.id}",
                )
                records[candidate.metadata.id].status = "running"
                records[candidate.metadata.id].started_at = utc_now()
                records[candidate.metadata.id].heartbeat_at = utc_now()
                records[candidate.metadata.id].attempt_count += 1
                process.start()
                records[candidate.metadata.id].worker_pid = process.pid
                active[candidate.metadata.id] = _ActiveWorker(
                    candidate.metadata.id,
                    process,
                    monotonic(),
                    candidate.metadata.timeout_seconds,
                    candidate.metadata.maximum_memory_bytes,
                    workspace,
                    candidate.metadata.parallel_safe,
                    candidate.metadata.resource_class,
                    output_budget,
                )
                run_logger.event("info", "collector_started", collector_id=candidate.metadata.id)
                write_manifest(manifest_path, manifest)
                if not candidate.metadata.parallel_safe:
                    break

            try:
                message = result_queue.get(timeout=0.1)
            except queue.Empty:
                message = None
            if message is not None:
                collector_id = str(message["collector_id"])
                worker = active.pop(collector_id, None)
                if worker is not None:
                    worker.process.join(timeout=1)
                    self._apply_worker_result(records[collector_id], _result_from_dict(message["result"]), worker)
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
                        run_logger.event("info", "collector_finished", collector_id=collector_id)
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
                            errors=(
                                f"collector working set {memory_bytes} exceeded "
                                f"maximum_memory_bytes={worker.maximum_memory_bytes}",
                            ),
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
    def _worker_memory_bytes(process: multiprocessing.Process) -> int | None:
        """Return one worker's resident working set using local OS facilities."""
        if process.pid is None:
            return None
        if os.name == "nt":
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            handle = ctypes.windll.kernel32.OpenProcess(0x0410, False, process.pid)
            if not handle:
                return None
            try:
                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                if not ctypes.windll.psapi.GetProcessMemoryInfo(
                        handle,
                        ctypes.byref(counters),
                        counters.cb,
                ):
                    return None
                return int(counters.WorkingSetSize)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        status_path = Path(f"/proc/{process.pid}/status")
        try:
            for line in status_path.read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None

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
        assert candidate.metadata is not None
        if (
                record.status != CollectorStatus.FAILED.value
                or record.attempt_count > candidate.metadata.maximum_retries
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
                "failure": dict(record.failure) if record.failure is not None else None,
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
        retry_not_before[record.id] = monotonic() + candidate.metadata.retry_delay_seconds
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
        class ProcessEntry(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_ulong),
                ("cntUsage", ctypes.c_ulong),
                ("th32ProcessID", ctypes.c_ulong),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", ctypes.c_ulong),
                ("cntThreads", ctypes.c_ulong),
                ("th32ParentProcessID", ctypes.c_ulong),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.c_ulong),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        kernel = ctypes.windll.kernel32
        snapshot = kernel.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot in (0, -1, ctypes.c_void_p(-1).value):
            return ()
        try:
            entry = ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            relationships: dict[int, list[int]] = {}
            valid = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while valid:
                relationships.setdefault(int(entry.th32ParentProcessID), []).append(int(entry.th32ProcessID))
                valid = kernel.Process32NextW(snapshot, ctypes.byref(entry))
            descendants: list[int] = []
            pending = [parent_pid]
            while pending:
                children = relationships.get(pending.pop(), [])
                descendants.extend(children)
                pending.extend(children)
            return tuple(reversed(descendants))
        finally:
            kernel.CloseHandle(snapshot)

    @staticmethod
    def _terminate_process_tree(process: multiprocessing.Process) -> None:
        """Terminate one worker-owned process tree without touching peer workers."""
        if process.pid is not None:
            try:
                if os.name == "nt":
                    descendants = RunSupervisor._windows_descendants(process.pid)
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=5,
                    )
                    for child_pid in descendants:
                        handle = ctypes.windll.kernel32.OpenProcess(0x0001 | 0x100000, False, child_pid)
                        if handle:
                            try:
                                ctypes.windll.kernel32.TerminateProcess(handle, 1)
                                ctypes.windll.kernel32.WaitForSingleObject(handle, 2000)
                            finally:
                                ctypes.windll.kernel32.CloseHandle(handle)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)

    @staticmethod
    def _refresh_worker_progress(record, worker: _ActiveWorker) -> bool:
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
    def _apply_worker_result(record, result: CollectorResult, worker: _ActiveWorker) -> None:
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
            record.termination_reason = (
                "completed" if result.status == CollectorStatus.SUCCEEDED else result.status.value
            )
        RunSupervisor._refresh_worker_progress(record, worker)

    def _cancel_active(self, active, records, manifest, manifest_path, reason: str) -> None:
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

    def _cancel_records(self, records, manifest, manifest_path, reason: str) -> None:
        self._cancel_active(self._active_workers, records, manifest, manifest_path, reason)
        for record in records.values():
            if record.status in {"planned", "running"}:
                record.apply_result(CollectorResult(CollectorStatus.CANCELLED, reason))
        write_manifest(manifest_path, manifest)
