"""Creation of manifest-led evidence packages and their SHA-256 sidecars."""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from logicytics.artifacts import sha256_file
from logicytics.manifest import RunManifest, write_manifest

if TYPE_CHECKING:
    from logicytics.runtime import RunOutcome


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
                f"  Started: {record.started_at or 'not started'}",
                f"  Finished: {record.finished_at or 'not finished'}",
                f"  Summary: {record.summary or 'none'}",
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
    for artifact in manifest.artifact_list():
        source = run_directory / "artifacts" / artifact.relative_path
        archive_name = f"artifacts/{artifact.relative_path}"
        if archive_name in archive_names:
            raise ValueError(f"manifest contains duplicate artifact path: {artifact.relative_path}")
        if not source.is_file():
            raise FileNotFoundError(f"manifest artifact is missing: {artifact.relative_path}")
        if source.stat().st_size != artifact.size_bytes or sha256_file(source) != artifact.sha256:
            raise ValueError(f"manifest artifact verification failed: {artifact.relative_path}")
        archive_names.add(archive_name)
        sources.append((source, archive_name))
    return sources


def _log_sources(run_directory: Path, manifest: RunManifest) -> list[tuple[Path, str]]:
    """Return only run-owned structured logs, preserving their relative paths."""
    sources: list[tuple[Path, str]] = []
    for directory in (run_directory / "logs", run_directory / "collectors"):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.jsonl")):
            sources.append((path, path.relative_to(run_directory).as_posix()))
    if manifest.request.get("performance_check") is True:
        performance_path = run_directory / "logs" / "performance.json"
        if not performance_path.is_file():
            raise FileNotFoundError("requested performance report is missing")
        sources.append((performance_path, "logs/performance.json"))
    return sources


def _sha256_stream(stream: BinaryIO) -> str:
    """Hash an open binary stream without loading the evidence into memory."""
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
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
    try:
        with zipfile.ZipFile(temporary_package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(manifest_path, "manifest.json")
            archive.write(summary_path, "summary.txt")
            for source, archive_name in (*artifact_sources, *log_sources):
                archive.write(source, archive_name)
        _verify_archive(temporary_package, manifest, expected_names)
        os.replace(temporary_package, package_path)
    finally:
        if temporary_package.exists():
            temporary_package.unlink()
    temporary_hash = hash_path.with_suffix(".sha256.tmp")
    package_sha256 = sha256_file(package_path)
    temporary_hash.write_text(f"{package_sha256}  {package_path.name}\n", encoding="ascii")
    os.replace(temporary_hash, hash_path)
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
