"""Authenticated manifests, file integrity, version checks, and developer audits."""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
import re
import sys
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, TypedDict

from logicytics.configuration import MaintenanceSettings

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:(?:-([0-9A-Za-z.-]+))|((?:a|b|rc|\.dev)\d+))?$"
)
_MAXIMUM_MANIFEST_BYTES = 2 * 1024 * 1024
_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "output",
        "venv",
    }
)


@dataclass(frozen=True, slots=True)
class IntegrityManifest:
    """A validated version and required-file digest map."""

    version: str
    files: Mapping[str, str]
    source: str


def sha256_path(path: Path) -> str:
    """Hash a regular file with bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently accepting the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"integrity manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json(payload: bytes | str, source: str) -> object:
    """Decode strict UTF-8 JSON with duplicate-key detection."""
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} integrity manifest is not valid UTF-8 JSON: {error}") from error


def _parse_manifest(payload: object, source: str) -> IntegrityManifest:
    """Reject ambiguous or execution-bearing remote/local manifest fields."""
    allowed_fields = {"schema_version", "version", "files"}
    if not isinstance(payload, dict) or set(payload) != allowed_fields:
        raise ValueError("integrity manifest must contain only schema_version, version, and files")
    if payload["schema_version"] != 1:
        raise ValueError("integrity manifest schema_version must be 1")
    version = payload["version"]
    files = payload["files"]
    if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
        raise ValueError("integrity manifest version must use semantic versioning")
    if not isinstance(files, dict):
        raise ValueError("integrity manifest files must be an object")
    validated: dict[str, str] = {}
    for name, digest in files.items():
        if (
                not isinstance(name, str)
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("integrity manifest file entries require path and SHA-256 strings")
        path = PurePosixPath(name)
        if (
                name == "."
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != name
                or "\\" in name
        ):
            raise ValueError(f"integrity manifest path is unsafe: {name!r}")
        if any(
                part in _EXCLUDED_PARTS or part.startswith(".") and part != ".github"
                for part in path.parts
        ):
            raise ValueError(f"integrity manifest path is excluded: {name!r}")
        validated[name] = digest
    return IntegrityManifest(version, dict(sorted(validated.items())), source)


def fetch_remote_manifest(settings: MaintenanceSettings) -> IntegrityManifest | None:
    """Fetch an optional HTTPS manifest and verify exact pinned bytes before parsing."""
    if settings.remote_manifest_url is None:
        return None
    assert settings.remote_manifest_sha256 is not None
    request = urllib.request.Request(
        settings.remote_manifest_url,
        headers={"User-Agent": "Logicytics/4 integrity-check"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = response.read(_MAXIMUM_MANIFEST_BYTES + 1)
    if len(payload) > _MAXIMUM_MANIFEST_BYTES:
        raise ValueError("remote integrity manifest exceeds the 2 MiB limit")
    digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, settings.remote_manifest_sha256):
        raise ValueError("remote integrity manifest SHA-256 does not match its configured pin")
    raw = _decode_json(payload, "remote")
    return _parse_manifest(raw, settings.remote_manifest_url)


def load_local_manifest(
        project_root: Path, settings: MaintenanceSettings
) -> IntegrityManifest | None:
    """Load the configured project-owned manifest without leaving the repository."""
    path = (project_root / settings.local_manifest_path).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError("local integrity manifest escapes the project") from error
    if not path.exists():
        return None
    try:
        raw = _decode_json(path.read_text(encoding="utf-8"), "local")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"local integrity manifest is invalid: {error}") from error
    return _parse_manifest(raw, str(path))


def project_files(project_root: Path, settings: MaintenanceSettings) -> tuple[Path, ...]:
    """List manifest-eligible files while excluding outputs, caches, history, and tools."""
    root = project_root.resolve()
    manifest_path = (root / settings.local_manifest_path).resolve()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.resolve() == manifest_path:
            continue
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        lowered_parts = tuple(part.casefold() for part in relative.parts)
        if len(lowered_parts) >= 2 and lowered_parts[:2] == ("tools", "sysinternals"):
            continue
        if path.name in {"interaction_history.json.gz", "flag_usage.svg"}:
            continue
        files.append(path)
    return tuple(sorted(files))


def build_manifest(
        project_root: Path, settings: MaintenanceSettings, version: str
) -> IntegrityManifest:
    """Create a deterministic required-file manifest from eligible project files."""
    if _VERSION.fullmatch(version) is None:
        raise ValueError("next version must use semantic versioning")
    root = project_root.resolve()
    files = {
        path.relative_to(root).as_posix(): sha256_path(path)
        for path in project_files(root, settings)
    }
    return IntegrityManifest(version, files, "generated")


def write_local_manifest(
        project_root: Path,
        settings: MaintenanceSettings,
        manifest: IntegrityManifest,
) -> Path:
    """Atomically publish an explicitly approved local manifest update."""
    path = (project_root / settings.local_manifest_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"schema_version": 1, "version": manifest.version, "files": manifest.files},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def write_legacy_ini_manifest(
        project_root: Path,
        manifest: IntegrityManifest,
) -> Path:
    """Atomically update only version and file membership in historical CODE/config.ini."""
    if _VERSION.fullmatch(manifest.version) is None:
        raise ValueError("next version must use semantic versioning")
    path = (project_root / "CODE" / "config.ini").resolve()
    try:
        path.relative_to(project_root.resolve())
        payload = path.read_bytes()
    except (ValueError, OSError) as error:
        raise ValueError(f"legacy config.ini is unavailable: {error}") from error
    if len(payload) > _MAXIMUM_MANIFEST_BYTES:
        raise ValueError("legacy config.ini exceeds the 2 MiB limit")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"legacy config.ini is not valid UTF-8: {error}") from error
    header = re.search(r"(?m)^\[System Settings\][ \t]*\r?$", text)
    if header is None:
        raise ValueError("legacy config.ini is missing [System Settings]")
    following = re.search(r"(?m)^\[[^]\r\n]+\][ \t]*\r?$", text[header.end():])
    section_end = header.end() + following.start() if following is not None else len(text)
    replacement = text[header.start():section_end]
    files = ", ".join(sorted(manifest.files))
    updates = {"version": manifest.version, "files": f'"{files}"'}
    for key, value in updates.items():
        pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
        if pattern.search(replacement) is None:
            raise ValueError(f"legacy config.ini [System Settings] is missing {key}")
        replacement = pattern.sub(f"{key} = {value}", replacement, count=1)
    updated = text[:header.start()] + replacement + text[section_end:]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8", newline="")
    temporary.replace(path)
    return path


def compare_files(
        project_root: Path,
        settings: MaintenanceSettings,
        manifest: IntegrityManifest,
) -> dict[str, list[str]]:
    """Report exact missing, modified, extra, and unchanged project files."""
    root = project_root.resolve()
    current = {
        path.relative_to(root).as_posix(): sha256_path(path)
        for path in project_files(root, settings)
    }
    expected = dict(manifest.files)
    return {
        "missing": sorted(set(expected) - set(current)),
        "modified": sorted(
            name
            for name in set(expected).intersection(current)
            if expected[name] != current[name]
        ),
        "extra": sorted(set(current) - set(expected)),
        "unchanged": sorted(
            name
            for name in set(expected).intersection(current)
            if expected[name] == current[name]
        ),
    }


def local_version(project_root: Path) -> str:
    """Read the authoritative package version from pyproject.toml."""
    with (project_root / "pyproject.toml").open("rb") as stream:
        payload = tomllib.load(stream)
    version = payload.get("project", {}).get("version")
    if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
        raise ValueError("pyproject.toml does not contain a semantic project version")
    return version


def compare_versions(local: str, remote: str) -> str:
    """Compare semantic versions while treating snapshots as older than stable peers."""

    def parsed(value: str) -> tuple[int, int, int, int, tuple[tuple[int, int | str], ...]]:
        match = _VERSION.fullmatch(value)
        if match is None:
            raise ValueError("version comparison requires semantic versions")
        major, minor, patch, prerelease, pep_prerelease = match.groups()
        identifiers: tuple[tuple[int, int | str], ...] = ()
        if pep_prerelease is not None:
            pep_match = re.fullmatch(r"(a|b|rc|\.dev)(\d+)", pep_prerelease)
            assert pep_match is not None
            label, number = pep_match.groups()
            identifiers = ((0, {".dev": 0, "a": 1, "b": 2, "rc": 3}[label]), (0, int(number)))
        elif prerelease is not None:
            identifiers = tuple(
                (0, int(identifier)) if identifier.isdigit() else (1, identifier)
                for identifier in prerelease.split(".")
            )
        is_stable = prerelease is None and pep_prerelease is None
        return int(major), int(minor), int(patch), 1 if is_stable else 0, identifiers

    left, right = parsed(local), parsed(remote)
    return "current" if left == right else "behind" if left < right else "ahead"


def python_support(settings: MaintenanceSettings) -> dict[str, str]:
    """Classify the running Python as recommended, supported, or incompatible."""
    running = (sys.version_info.major, sys.version_info.minor)
    minimum = tuple(map(int, settings.minimum_python.split(".")))
    recommended = tuple(map(int, settings.recommended_python.split(".")))
    status = (
        "incompatible"
        if running < minimum
        else "recommended"
        if running == recommended
        else "supported"
    )
    return {
        "running": f"{running[0]}.{running[1]}",
        "minimum": settings.minimum_python,
        "recommended": settings.recommended_python,
        "status": status,
    }


def maintenance_diagnostics(
        project_root: Path, settings: MaintenanceSettings
) -> dict[str, object]:
    """Resolve optional remote/local integrity evidence without changing collection state."""
    remote = fetch_remote_manifest(settings)
    local_manifest = load_local_manifest(project_root, settings)
    selected = remote or local_manifest
    return {
        "python_support": python_support(settings),
        "local_version": local_version(project_root),
        "manifest_source": selected.source if selected else None,
        "manifest_version": selected.version if selected else None,
        "version_status": (
            compare_versions(local_version(project_root), selected.version)
            if selected
            else "unconfigured"
        ),
        "files": (
            compare_files(project_root, settings, selected)
            if selected
            else {"missing": [], "modified": [], "extra": [], "unchanged": []}
        ),
        "remote_enabled": remote is not None,
    }


class DeveloperChecks(TypedDict):
    naming_violations: list[str]
    misplaced_python: list[str]
    missing_module_docstrings: list[str]
    crowded_modules: list[str]
    eligible_file_count: int


def developer_checks(project_root: Path, settings: MaintenanceSettings) -> DeveloperChecks:
    """Inspect repository organization without importing or executing project modules."""
    naming: list[str] = []
    missing_docstrings: list[str] = []
    crowded_modules: list[str] = []
    misplaced_python: list[str] = []
    root = project_root.resolve()
    for path in project_files(root, settings):
        relative = path.relative_to(root)
        if path.suffix.casefold() != ".py":
            continue
        if path.stem != "__init__" and re.fullmatch(r"[a-z][a-z0-9_]*", path.stem) is None:
            naming.append(relative.as_posix())
        if relative.parts[0] not in {"CODE", "core", "logicytics", "plugins", "tests"}:
            misplaced_python.append(relative.as_posix())
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (OSError, UnicodeDecodeError, SyntaxError):
            missing_docstrings.append(relative.as_posix())
            continue
        if ast.get_docstring(tree) is None:
            missing_docstrings.append(relative.as_posix())

        public_features = [
            node.name for node in tree.body
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_")
        ]

        if len(public_features) > 12:
            crowded_modules.append(relative.as_posix())

    return {
        "naming_violations": sorted(naming),
        "misplaced_python": sorted(misplaced_python),
        "missing_module_docstrings": sorted(missing_docstrings),
        "crowded_modules": sorted(crowded_modules),
        "eligible_file_count": len(project_files(root, settings)),
    }
