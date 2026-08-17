"""Safe recursive file discovery utilities for engine and collector use."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def list_files(
        root: Path,
        *,
        extensions: Iterable[str] = (),
        excluded_names: Iterable[str] = (),
        excluded_directories: Iterable[str] = (),
        append_to: Iterable[Path] = (),
) -> tuple[Path, ...]:
    """Return sorted normalized files under root, applying explicit filters safely."""
    resolved_root = root.resolve()
    normalized_extensions = {extension.casefold() if extension.startswith(".") else f".{extension.casefold()}" for
                             extension in extensions}
    names = {name.casefold() for name in excluded_names}
    directories = {name.casefold() for name in excluded_directories}
    results = {Path(path).resolve() for path in append_to}
    if not resolved_root.is_dir():
        return tuple(sorted(results))
    for path in resolved_root.rglob("*"):
        try:
            relative = path.resolve().relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        if any(part.casefold() in directories for part in relative.parts[:-1]):
            continue
        if not path.is_file() or path.name.casefold() in names:
            continue
        if normalized_extensions and path.suffix.casefold() not in normalized_extensions:
            continue
        results.add(path.resolve())
    return tuple(sorted(results))
