"""Per-collector process supervisor and run workspace lifecycle."""

from __future__ import annotations

import contextlib
import importlib.util
import multiprocessing
import os
import queue
import subprocess
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

from logicytics.artifacts import WorkspaceArtifactWriter
from logicytics.configuration import AppConfig
from logicytics.contracts import (
    Artifact,
    CollectorContext,
    CollectorResult,
    CollectorStatus,
    RunStatus,
    ValidationResult,
)
from logicytics.logging import FileEventLogger
from logicytics.manifest import RunManifest, write_manifest, utc_now
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
    workspace: Path


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
                collector.cleanup(context)
                context.logger.event("info", "collection_finished", status=result.status.value)
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
            raise PermissionError("collection requires acknowledge_authorization=True")
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
        )
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
        write_manifest(manifest_path, manifest)
        run_logger.event("info", "run_finished", status=manifest.status.value, artifacts=manifest.total_artifact_bytes)
        return RunOutcome(manifest=manifest, run_directory=run_directory, manifest_path=manifest_path)

    def _supervise(
            self,
            plan: RunPlan,
            run_id: str,
            workspace_root: Path,
            artifact_root: Path,
            cancellation_file: Path,
            manifest: RunManifest,
            manifest_path: Path,
            records: dict[str, object],
            run_logger: FileEventLogger,
    ) -> None:
        """Schedule bounded isolated workers and contain each terminal failure."""
        pending = list(plan.collectors)
        active: dict[str, _ActiveWorker] = {}
        self._active_workers = active
        result_queue: multiprocessing.Queue = multiprocessing.get_context("spawn").Queue()
        worker_limit = min(plan.request.max_workers, self.configuration.runtime.maximum_workers)
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
                candidate = pending.pop(0)
                assert candidate.metadata is not None
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
                }
                process = multiprocessing.get_context("spawn").Process(
                    target=_worker_entry,
                    args=(payload, result_queue),
                    name=f"Logicytics-{candidate.metadata.id}",
                )
                records[candidate.metadata.id].status = "running"
                records[candidate.metadata.id].started_at = utc_now()
                process.start()
                active[candidate.metadata.id] = _ActiveWorker(
                    candidate.metadata.id,
                    process,
                    monotonic(),
                    candidate.metadata.timeout_seconds,
                    workspace,
                )
                run_logger.event("info", "collector_started", collector_id=candidate.metadata.id)
                write_manifest(manifest_path, manifest)

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
                    run_logger.event("info", "collector_finished", collector_id=collector_id)
                    write_manifest(manifest_path, manifest)

            for collector_id, worker in tuple(active.items()):
                if monotonic() - worker.started_at > worker.timeout_seconds:
                    self._terminate_process_tree(worker.process)
                    active.pop(collector_id)
                    self._apply_worker_result(
                        records[collector_id],
                        CollectorResult(CollectorStatus.FAILED, "collector exceeded its declared timeout"),
                        worker,
                    )
                    run_logger.event("error", "collector_timed_out", collector_id=collector_id)
                    write_manifest(manifest_path, manifest)
                elif not worker.process.is_alive() and worker.process.exitcode not in (None, 0):
                    worker.process.join(timeout=1)
                    active.pop(collector_id)
                    self._apply_worker_result(
                        records[collector_id],
                        CollectorResult(CollectorStatus.FAILED, "collector exited without a result"),
                        worker,
                    )
                    run_logger.event("error", "collector_exited_without_result", collector_id=collector_id)
                    write_manifest(manifest_path, manifest)
            sleep(0.01)

    @staticmethod
    def _terminate_process_tree(process: multiprocessing.Process) -> None:
        """Terminate a worker and, on Windows, any subprocesses it created."""
        if process.pid is not None and os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                text=True,
            )
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)

    @staticmethod
    def _apply_worker_result(record, result: CollectorResult, worker: _ActiveWorker) -> None:
        """Attach result, elapsed time, and progress accounting to one record."""
        record.apply_result(result, duration_seconds=round(monotonic() - worker.started_at, 3))
        events_path = worker.workspace / "events.jsonl"
        if events_path.exists():
            events = events_path.read_text(encoding="utf-8").splitlines()
            record.event_count = len(events)
            if events:
                import json

                record.last_progress_at = json.loads(events[-1]).get("at")

    def _cancel_active(self, active, records, manifest, manifest_path, reason: str) -> None:
        for collector_id, worker in tuple(active.items()):
            self._terminate_process_tree(worker.process)
            active.pop(collector_id)
            self._apply_worker_result(
                records[collector_id],
                CollectorResult(CollectorStatus.CANCELLED, reason),
                worker,
            )
        write_manifest(manifest_path, manifest)

    def _cancel_records(self, records, manifest, manifest_path, reason: str) -> None:
        self._cancel_active(self._active_workers, records, manifest, manifest_path, reason)
        for record in records.values():
            if record.status in {"planned", "running"}:
                record.apply_result(CollectorResult(CollectorStatus.CANCELLED, reason))
        write_manifest(manifest_path, manifest)
