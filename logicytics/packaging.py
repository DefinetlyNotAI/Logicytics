"""Creation of manifest-led evidence packages and their SHA-256 sidecars."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from logicytics.artifacts import sha256_file
from logicytics.manifest import RunManifest, write_manifest
from logicytics.runtime import RunOutcome


def _summary(manifest: RunManifest) -> str:
    lines = [f"Logicytics run: {manifest.run_id}", f"Status: {manifest.status.value}", "", "Collectors:"]
    for record in manifest.collectors:
        lines.append(f"- {record.id}: {record.status} — {record.summary or ''}".rstrip())
    lines.extend(["", f"Artifacts: {len(manifest.artifact_list())}"])
    return "\n".join(lines) + "\n"


def _artifact_sources(outcome: RunOutcome) -> list[tuple[Path, str]]:
    """Resolve and verify every manifest-declared artifact before packaging it."""
    sources: list[tuple[Path, str]] = []
    archive_names: set[str] = set()
    for artifact in outcome.manifest.artifact_list():
        source = outcome.run_directory / "artifacts" / artifact.relative_path
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


def _log_sources(run_directory: Path) -> list[tuple[Path, str]]:
    """Return only run-owned structured logs, preserving their relative paths."""
    sources: list[tuple[Path, str]] = []
    for directory in (run_directory / "logs", run_directory / "collectors"):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.jsonl")):
            sources.append((path, path.relative_to(run_directory).as_posix()))
    return sources


def _verify_archive(package_path: Path, expected_names: set[str]) -> None:
    """Confirm an atomically written package contains exactly its declared inputs."""
    with zipfile.ZipFile(package_path) as archive:
        names = set(archive.namelist())
        if names != expected_names:
            raise ValueError("package contents do not match its manifest-led input set")
        if archive.testzip() is not None:
            raise ValueError("package integrity verification failed")


def package_run(outcome: RunOutcome) -> tuple[Path, Path]:
    """Package registered artifacts, manifest, and summary without scanning arbitrary files."""
    package_directory = outcome.run_directory.parent.parent / "PACKAGES"
    package_directory.mkdir(parents=True, exist_ok=True)
    package_path = package_directory / f"{outcome.manifest.run_id}.zip"
    hash_path = package_path.with_suffix(".zip.sha256")
    outcome.manifest.package = {"path": str(package_path)}
    write_manifest(outcome.manifest_path, outcome.manifest)

    summary_path = outcome.run_directory / "summary.txt"
    summary_path.write_text(_summary(outcome.manifest), encoding="utf-8")
    artifact_sources = _artifact_sources(outcome)
    log_sources = _log_sources(outcome.run_directory)
    expected_names = {"manifest.json", "summary.txt"}
    expected_names.update(archive_name for _, archive_name in artifact_sources)
    expected_names.update(archive_name for _, archive_name in log_sources)
    temporary_package = package_path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary_package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(outcome.manifest_path, "manifest.json")
            archive.write(summary_path, "summary.txt")
            for source, archive_name in (*artifact_sources, *log_sources):
                archive.write(source, archive_name)
        _verify_archive(temporary_package, expected_names)
        os.replace(temporary_package, package_path)
    finally:
        if temporary_package.exists():
            temporary_package.unlink()
    temporary_hash = hash_path.with_suffix(".sha256.tmp")
    temporary_hash.write_text(f"{sha256_file(package_path)}  {package_path.name}\n", encoding="ascii")
    os.replace(temporary_hash, hash_path)
    return package_path, hash_path
