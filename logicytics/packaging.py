"""Creation of manifest-led evidence packages and their SHA-256 sidecars."""

from __future__ import annotations

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
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(outcome.manifest_path, "manifest.json")
        archive.write(summary_path, "summary.txt")
        for artifact in outcome.manifest.artifact_list():
            source = outcome.run_directory / "artifacts" / artifact.relative_path
            if not source.is_file():
                raise FileNotFoundError(f"manifest artifact is missing: {artifact.relative_path}")
            archive.write(source, f"artifacts/{artifact.relative_path}")
        for log_path in sorted((outcome.run_directory / "logs").rglob("*.jsonl")):
            archive.write(log_path, log_path.relative_to(outcome.run_directory).as_posix())
        for collector_path in sorted((outcome.run_directory / "collectors").rglob("*.jsonl")):
            archive.write(collector_path, collector_path.relative_to(outcome.run_directory).as_posix())
    hash_path.write_text(f"{sha256_file(package_path)}  {package_path.name}\n", encoding="ascii")
    return package_path, hash_path
