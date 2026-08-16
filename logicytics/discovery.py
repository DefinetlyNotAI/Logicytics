"""Strict, side-effect-free discovery and preflight for collector modules."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from logicytics.contracts import CONTRACT_VERSION, CollectorKind, CollectorMetadata

_FILENAME = re.compile(r"^[a-z][a-z0-9_]*\.py$")
_VAGUE_NAMES = {"main.py", "misc.py", "stuff.py", "utils.py"}
_FORBIDDEN_TOP_LEVEL_CALLS = {"exit", "quit", "input", "open", "print"}


@dataclass(slots=True)
class CollectorCandidate:
    """A file that may be a runnable collector."""

    path: Path
    kind: CollectorKind
    expected_class: str
    static_errors: list[str] = field(default_factory=list)
    metadata: CollectorMetadata | None = None
    runtime_error: str | None = None

    @property
    def valid(self) -> bool:
        """Whether static and runtime validation both passed."""
        return not self.static_errors and self.metadata is not None and self.runtime_error is None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Complete validation outcome for all discovered collector candidates."""

    candidates: tuple[CollectorCandidate, ...]

    @property
    def valid(self) -> tuple[CollectorCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.valid)

    @property
    def invalid(self) -> tuple[CollectorCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.valid)


def _pascal_case(filename: str) -> str:
    return "".join(part.capitalize() for part in Path(filename).stem.split("_")) + "Collector"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _iter_candidates(project_root: Path, kind: CollectorKind) -> list[Path]:
    root = project_root / ("core" if kind is CollectorKind.CORE else "plugins")
    if not root.exists():
        return []
    paths: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part.startswith("_") or part in {"tests", "examples", "venv", ".venv"} for part in relative.parts):
            continue
        if not _is_within(path, root):
            continue
        if kind is CollectorKind.CORE and len(relative.parts) != 2:
            continue
        if kind is CollectorKind.PLUGIN and not (len(relative.parts) == 1 or relative.name == "main.py"):
            continue
        paths.append(path)
    return paths


def _top_level_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return f"{node.func.value.id}.{node.func.attr}"
    return None


def _validate_static(path: Path, kind: CollectorKind) -> CollectorCandidate:
    expected_class = _pascal_case(path.name)
    candidate = CollectorCandidate(path=path, kind=kind, expected_class=expected_class)
    if path.name in _VAGUE_NAMES and not (kind is CollectorKind.PLUGIN and path.name == "main.py"):
        candidate.static_errors.append("collector filename is reserved or too vague")
    if not _FILENAME.fullmatch(path.name):
        candidate.static_errors.append("filename must be lowercase snake_case.py")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        candidate.static_errors.append(f"cannot parse UTF-8 Python module: {error}")
        return candidate
    if ast.get_docstring(tree) is None:
        candidate.static_errors.append("module requires a docstring")

    public_classes = [item for item in tree.body if isinstance(item, ast.ClassDef) and not item.name.startswith("_")]
    if len(public_classes) != 1:
        candidate.static_errors.append("module must define exactly one public collector class")
    elif public_classes[0].name != expected_class:
        candidate.static_errors.append(f"collector class must be named {expected_class}")
    elif not any(
        (isinstance(base, ast.Name) and base.id == ("CoreCollector" if kind is CollectorKind.CORE else "PluginCollector"))
        or (isinstance(base, ast.Attribute) and base.attr == ("CoreCollector" if kind is CollectorKind.CORE else "PluginCollector"))
        for base in public_classes[0].bases
    ):
        candidate.static_errors.append("collector class must inherit from its required base class")
    elif public_classes:
        _validate_class_shape(public_classes[0], candidate)

    for item in tree.body:
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            continue
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            if any(isinstance(node, ast.Call) for node in ast.walk(item.value)):
                candidate.static_errors.append("forbidden call in top-level assignment")
            continue
        if isinstance(item, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
            call_name = _top_level_call_name(item.value)
            if call_name in _FORBIDDEN_TOP_LEVEL_CALLS or call_name:
                candidate.static_errors.append(f"forbidden top-level call: {call_name or 'unknown'}")
                continue
        candidate.static_errors.append(f"forbidden top-level statement: {type(item).__name__}")
    return candidate


def _validate_class_shape(class_node: ast.ClassDef, candidate: CollectorCandidate) -> None:
    """Enforce the small, documented public API allowed on a collector class."""
    if ast.get_docstring(class_node) is None:
        candidate.static_errors.append("collector class requires a docstring")
    methods = {
        item.name: item
        for item in class_node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_")
    }
    allowed = {"metadata", "validate", "collect", "cleanup", "estimate", "dependencies"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        candidate.static_errors.append(f"unsupported public collector methods: {', '.join(unknown)}")
    required = {"metadata", "validate", "collect", "cleanup"}
    expected_returns = {
        "metadata": "CollectorMetadata",
        "validate": "ValidationResult",
        "collect": "CollectorResult",
        "cleanup": "None",
    }
    missing = sorted(required - set(methods))
    if missing:
        candidate.static_errors.append(f"missing required collector methods: {', '.join(missing)}")
    for name, method in methods.items():
        if ast.get_docstring(method) is None:
            candidate.static_errors.append(f"{name} requires a docstring")
        if method.returns is None:
            candidate.static_errors.append(f"{name} requires a return type annotation")
        elif name in expected_returns and ast.unparse(method.returns).rsplit(".", 1)[-1] != expected_returns[name]:
            candidate.static_errors.append(f"{name} must return {expected_returns[name]}")
        parameters = method.args.args
        expected_parameter_count = 1 if name == "metadata" else 2
        if len(parameters) != expected_parameter_count:
            candidate.static_errors.append(f"{name} has an invalid parameter count")
            continue
        expected_first = "cls" if name == "metadata" else "self"
        if parameters[0].arg != expected_first:
            candidate.static_errors.append(f"{name} must begin with {expected_first}")
        if name == "metadata":
            if not any(isinstance(decorator, ast.Name) and decorator.id == "classmethod" for decorator in method.decorator_list):
                candidate.static_errors.append("metadata must be a classmethod")
        else:
            context_parameter = parameters[1]
            annotation = ast.unparse(context_parameter.annotation).rsplit(".", 1)[-1] if context_parameter.annotation else None
            if context_parameter.arg != "context" or annotation != "CollectorContext":
                candidate.static_errors.append(f"{name} must accept an annotated context parameter")


def discover(project_root: Path) -> tuple[CollectorCandidate, ...]:
    """Discover candidates without importing their modules."""
    candidates = [
        _validate_static(path, CollectorKind.CORE)
        for path in _iter_candidates(project_root, CollectorKind.CORE)
    ]
    candidates.extend(
        _validate_static(path, CollectorKind.PLUGIN)
        for path in _iter_candidates(project_root, CollectorKind.PLUGIN)
    )
    return tuple(candidates)


def _runtime_probe(project_root: Path, candidate: CollectorCandidate) -> None:
    engine_root = Path(__file__).resolve().parent.parent
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(
        value for value in (str(engine_root), str(project_root), inherited_pythonpath) if value
    )
    command = [
        sys.executable,
        "-m",
        "logicytics.validation_worker",
        str(candidate.path),
        candidate.kind.value,
        candidate.expected_class,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={**os.environ, "PYTHONPATH": pythonpath, "LOGICYTICS_VALIDATION": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        candidate.runtime_error = f"validation worker failed: {error}"
        return
    if completed.returncode != 0:
        candidate.runtime_error = completed.stderr.strip() or "validation worker rejected collector"
        return
    try:
        payload = json.loads(completed.stdout)
        metadata = CollectorMetadata.from_dict(payload["metadata"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        candidate.runtime_error = f"invalid validation response: {error}"
        return
    if metadata.minimum_contract_version != CONTRACT_VERSION:
        candidate.runtime_error = "collector contract version is unsupported"
    elif metadata.timeout_seconds < 1 or metadata.maximum_output_bytes < 1:
        candidate.runtime_error = "collector must declare positive timeout and output limits"
    elif not metadata.id.startswith(f"{candidate.kind.value}."):
        candidate.runtime_error = "collector ID must start with its owner kind"
    elif candidate.kind is CollectorKind.CORE and metadata.specialty.value != candidate.path.parent.name:
        candidate.runtime_error = "core collector specialty must match its parent folder"
    elif candidate.kind is CollectorKind.CORE and metadata.id != (
        f"core.{candidate.path.parent.name}.{candidate.path.stem}"
    ):
        candidate.runtime_error = "core collector ID must match core/<specialty>/<filename>.py"
    elif candidate.kind is CollectorKind.PLUGIN and metadata.id != (
        f"plugin.{candidate.path.parent.name if candidate.path.name == 'main.py' else candidate.path.stem}"
    ):
        candidate.runtime_error = "plugin collector ID must match its plugin folder or filename"
    else:
        candidate.metadata = metadata


def preflight(project_root: Path) -> PreflightReport:
    """Perform static checks then a short-lived isolated metadata probe."""
    candidates = list(discover(project_root))
    for candidate in candidates:
        if not candidate.static_errors:
            _runtime_probe(project_root, candidate)
    seen_ids: set[str] = set()
    for candidate in candidates:
        if candidate.metadata is None:
            continue
        if candidate.metadata.id in seen_ids:
            candidate.runtime_error = f"duplicate collector ID: {candidate.metadata.id}"
        seen_ids.add(candidate.metadata.id)
    return PreflightReport(tuple(candidates))
