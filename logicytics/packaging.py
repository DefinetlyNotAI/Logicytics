"""Creation of manifest-led evidence packages and their SHA-256 sidecars."""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO

from logicytics.artifacts import sha256_file
from logicytics.contracts import Artifact
from logicytics.manifest import RunManifest, write_manifest

if TYPE_CHECKING:
    from logicytics.runtime import RunOutcome

_STREAM_BLOCK_BYTES = 1024 * 1024


def _summary(manifest: RunManifest) -> str:
    lines = [
        f"Logicytics run: {manifest.run_id}",
        f"Status: {manifest.status.value}",
        f"Requested: {manifest.requested_at}",
        f"Finished: {manifest.finished_at or 'not finalized'}",
        "",
        "Collectors:",
    ]
    for record in manifest.collectors:
        lines.extend(
            [
                f"- {record.id}",
                f"  Status: {record.status}",
                f"  Isolation: {record.isolation_mode}",
                f"  Worker PID: {record.worker_pid if record.worker_pid is not None else 'not started'}",
                f"  Worker exit code: {record.worker_exit_code if record.worker_exit_code is not None else 'unknown'}",
                f"  Termination: {record.termination_reason or 'not started'}",
                f"  Attempts: {record.attempt_count}",
                f"  Peak memory bytes: {record.peak_memory_bytes}",
                f"  Files scanned: {record.progress['files_scanned']}",
                f"  Files copied: {record.progress['files_copied']}",
                f"  Bytes written: {record.progress['bytes_written']}",
                f"  Packets observed: {record.progress['packets_observed']}",
                f"  Events processed: {record.progress['events_processed']}",
                f"  Elapsed seconds: {record.progress['elapsed_seconds']}",
                f"  Started: {record.started_at or 'not started'}",
                f"  Finished: {record.finished_at or 'not finished'}",
                f"  Summary: {record.summary or 'none'}",
            ]
        )
        if record.failure is not None:
            lines.extend(
                [
                    f"  Failed operation: {record.failure['operation']}",
                    f"  Platform error: {record.failure['platform_error']}",
                    f"  Remediation: {record.failure['remediation']}",
                    f"  Retry safe: {str(record.failure['retry_safe']).lower()}",
                ]
            )
        if record.errors:
            lines.append("  Reasons:")
            lines.extend(f"    - {' '.join(error.splitlines())}" for error in record.errors)
    lines.extend(["", f"Artifacts: {len(manifest.artifact_list())}"])
    return "\n".join(lines) + "\n"


def _artifact_sources(run_directory: Path, manifest: RunManifest) -> list[tuple[Path, str]]:
    """Resolve and verify every manifest-declared artifact before packaging it."""
    sources: list[tuple[Path, str]] = []
    archive_names: set[str] = set()
    artifact_root = (run_directory / "artifacts").resolve()
    for record in manifest.collectors:
        for item in record.artifacts:
            artifact = Artifact(**item)
            if artifact.collector_id != record.id:
                raise ValueError(f"manifest artifact collector ownership is invalid: {artifact.relative_path}")
            relative = PurePosixPath(artifact.relative_path)
            owner = record.id.replace(".", "_")
            if (
                    not artifact.relative_path
                    or "\\" in artifact.relative_path
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or relative.as_posix() != artifact.relative_path
                    or len(relative.parts) < 2
                    or relative.parts[0] != owner
            ):
                raise ValueError(f"manifest artifact escapes its collector-owned store: {artifact.relative_path}")
            source = artifact_root.joinpath(*relative.parts)
            if not source.is_file():
                raise FileNotFoundError(f"manifest artifact is missing: {artifact.relative_path}")
            try:
                source.resolve(strict=True).relative_to(artifact_root / owner)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"manifest artifact escapes its collector-owned store: {artifact.relative_path}"
                ) from error
            archive_name = f"artifacts/{relative.as_posix()}"
            if archive_name in archive_names:
                raise ValueError(f"manifest contains duplicate artifact path: {artifact.relative_path}")
            if source.stat().st_size != artifact.size_bytes or sha256_file(source) != artifact.sha256:
                raise ValueError(f"manifest artifact verification failed: {artifact.relative_path}")
            archive_names.add(archive_name)
            sources.append((source, archive_name))
    return sources


def _log_sources(run_directory: Path, manifest: RunManifest) -> list[tuple[Path, str]]:
    """Return only the engine log and manifest-owned collector event channels."""
    sources: list[tuple[Path, str]] = []
    root = run_directory.resolve()
    candidates = [root / "logs" / "engine.jsonl"]
    candidates.extend(
        root / "collectors" / record.id.replace(".", "_") / "events.jsonl"
        for record in manifest.collectors
    )
    for path in candidates:
        if not path.exists() and not path.is_symlink():
            continue
        if not path.is_file() or path.resolve(strict=True) != path:
            raise ValueError(f"diagnostic log escapes its run-owned event channel: {path.name}")
        sources.append((path, path.relative_to(root).as_posix()))
    if manifest.request.get("performance_check") is True:
        performance_path = root / "logs" / "performance.json"
        if not performance_path.is_file():
            raise FileNotFoundError("requested performance report is missing")
        if performance_path.resolve(strict=True) != performance_path:
            raise ValueError("performance report escapes its run-owned log directory")
        sources.append((performance_path, "logs/performance.json"))
    return sources


def _stream_archive_member(archive: zipfile.ZipFile, source: Path, archive_name: str) -> None:
    """Copy one approved member into the archive using bounded evidence blocks."""
    with source.open("rb") as reader, archive.open(archive_name, "w", force_zip64=True) as writer:
        while block := reader.read(_STREAM_BLOCK_BYTES):
            writer.write(block)


def _sha256_stream(stream: BinaryIO) -> str:
    """Hash an open binary stream without loading the evidence into memory."""
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(_STREAM_BLOCK_BYTES), b""):
        digest.update(block)
    return digest.hexdigest()


def _verify_archive(package_path: Path, manifest: RunManifest, expected_names: set[str]) -> None:
    """Confirm packaged artifact bytes match the finalized manifest exactly."""
    with zipfile.ZipFile(package_path) as archive:
        name_list = archive.namelist()
        if len(name_list) != len(set(name_list)) or set(name_list) != expected_names:
            raise ValueError("package contents do not match its manifest-led input set")
        if archive.testzip() is not None:
            raise ValueError("package integrity verification failed")
        for artifact in manifest.artifact_list():
            archive_name = f"artifacts/{artifact.relative_path}"
            member = archive.getinfo(archive_name)
            with archive.open(member) as stream:
                digest = _sha256_stream(stream)
            if member.file_size != artifact.size_bytes or digest != artifact.sha256:
                raise ValueError(f"packaged artifact verification failed: {artifact.relative_path}")


def package_manifest(run_directory: Path, manifest: RunManifest, manifest_path: Path) -> tuple[Path, Path]:
    """Package registered artifacts, manifest, and summary without scanning arbitrary files."""
    package_directory = run_directory.parent.parent / "PACKAGES"
    package_directory.mkdir(parents=True, exist_ok=True)
    package_path = package_directory / f"{manifest.run_id}.zip"
    hash_path = package_path.with_suffix(".zip.sha256")

    summary_path = run_directory / "summary.txt"
    artifact_sources = _artifact_sources(run_directory, manifest)
    log_sources = _log_sources(run_directory, manifest)
    manifest.package = {"path": str(package_path)}
    write_manifest(manifest_path, manifest)
    summary_path.write_text(_summary(manifest), encoding="utf-8")
    expected_names = {"manifest.json", "summary.txt"}
    expected_names.update(archive_name for _, archive_name in artifact_sources)
    expected_names.update(archive_name for _, archive_name in log_sources)
    temporary_package = package_path.with_suffix(".zip.tmp")
    temporary_hash = hash_path.with_suffix(".sha256.tmp")
    try:
        with zipfile.ZipFile(temporary_package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            _stream_archive_member(archive, manifest_path, "manifest.json")
            _stream_archive_member(archive, summary_path, "summary.txt")
            for source, archive_name in (*artifact_sources, *log_sources):
                _stream_archive_member(archive, source, archive_name)
        _verify_archive(temporary_package, manifest, expected_names)
        package_sha256 = sha256_file(temporary_package)
        temporary_hash.write_text(f"{package_sha256}  {package_path.name}\n", encoding="ascii")
        package_backup = package_path.with_suffix(".zip.backup")
        hash_backup = hash_path.with_suffix(".sha256.backup")
        try:
            if package_path.exists():
                os.replace(package_path, package_backup)
            if hash_path.exists():
                os.replace(hash_path, hash_backup)
            os.replace(temporary_package, package_path)
            os.replace(temporary_hash, hash_path)
        except BaseException:
            if package_path.exists():
                package_path.unlink()
            if hash_path.exists() and hash_backup.exists():
                hash_path.unlink()
            if package_backup.exists():
                os.replace(package_backup, package_path)
            if hash_backup.exists():
                os.replace(hash_backup, hash_path)
            raise
        if package_backup.exists():
            package_backup.unlink()
        if hash_backup.exists():
            hash_backup.unlink()
    finally:
        if temporary_package.exists():
            temporary_package.unlink()
        if temporary_hash.exists():
            temporary_hash.unlink()
    manifest.package = {
        "path": str(package_path),
        "sha256_path": str(hash_path),
        "sha256": package_sha256,
    }
    write_manifest(manifest_path, manifest)
    return package_path, hash_path


def package_run(outcome: "RunOutcome") -> tuple[Path, Path]:
    """Compatibility wrapper for callers holding a complete runtime outcome."""
    return package_manifest(outcome.run_directory, outcome.manifest, outcome.manifest_path)
