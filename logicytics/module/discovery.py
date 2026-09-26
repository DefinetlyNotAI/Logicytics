"""Strict, side-effect-free discovery and preflight for collector modules."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from subprocess import TimeoutExpired

from logicytics.module.contracts import (
    CONTRACT_VERSION,
    Capability,
    CollectorKind,
    CollectorMetadata,
    Specialty,
)
from logicytics.module.platform_adapters import process_adapter

_FILENAME = re.compile(r"^[a-z][a-z0-9_]*\.py$")
_VAGUE_NAMES = {"main.py", "misc.py", "stuff.py", "utils.py"}
_APPLICATION_IMPORTS = {
    "CollectorSnapshot",
    "RunSnapshot",
    "api",
    "artifacts",
    "cli",
    "configuration",
    "discovery",
    "environment",
    "load_configuration",
    "manifest",
    "packaging",
    "plan_run",
    "planner",
    "query_run",
    "read_artifact",
    "runtime",
    "run_collection",
    "validation_worker",
}
_CACHE_SCHEMA_VERSION = 2
_COLLECTOR_SERVICE_MODULES = {
    "logicytics.contracts",
    "logicytics.module.contracts",
    "logicytics.global.ctypes_collector",
    "logicytics.platform_adapters",
    "logicytics.module.platform_adapters",
}

PreflightProgress = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class ValidationDiagnostic:
    """One stable, source-addressable collector validation failure."""

    path: str
    line: int
    rule: str
    message: str


@dataclass(slots=True)
class CollectorCandidate:
    """A file that may be a runnable collector."""

    path: Path
    kind: CollectorKind
    expected_class: str
    static_errors: list[str] = field(default_factory=list)
    metadata: CollectorMetadata | None = None
    runtime_error: str | None = None
    execution_type: str = "collector"

    @property
    def valid(self) -> bool:
        """Whether static and runtime validation both passed."""
        return not self.static_errors and self.metadata is not None and self.runtime_error is None

    @property
    def selection_id(self) -> str:
        """Return the ID used to explicitly select this path even when metadata is invalid."""
        if self.metadata is not None:
            return self.metadata.id
        owner = self.path.parent.name if self.path.name == "main.py" else self.path.stem
        return f"{self.kind.value}.{owner}"

    @property
    def diagnostics(self) -> tuple[ValidationDiagnostic, ...]:
        """Normalize free-form validator details into exact source diagnostics."""
        messages = [*self.static_errors]
        if self.runtime_error:
            messages.append(self.runtime_error)
        return tuple(
            ValidationDiagnostic(
                path=str(self.path),
                line=_diagnostic_line(self.path, message),
                rule=_diagnostic_rule(message, runtime=index >= len(self.static_errors)),
                message=message,
            )
            for index, message in enumerate(messages)
        )


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Complete validation outcome for all discovered collector candidates."""

    candidates: tuple[CollectorCandidate, ...]

    @property
    def valid(self) -> tuple[CollectorCandidate, ...]:
        """Return candidates that passed both static and runtime validation."""
        return tuple(candidate for candidate in self.candidates if candidate.valid)

    @property
    def invalid(self) -> tuple[CollectorCandidate, ...]:
        """Return candidates that must be blocked or quarantined."""
        return tuple(candidate for candidate in self.candidates if not candidate.valid)

    def to_dict(
            self,
            *,
            selected_plugins: tuple[str, ...] = (),
            enable_plugins: bool = False,
    ) -> dict[str, list[dict[str, object]]]:
        """Classify invalid plugins as quarantined unless the request selects them."""
        valid = [
            {"id": candidate.metadata.id, "kind": candidate.kind.value, "path": str(candidate.path)}
            for candidate in self.valid
            if candidate.metadata is not None
        ]
        invalid: list[dict[str, object]] = []
        quarantined: list[dict[str, object]] = []
        selected = set(selected_plugins)
        for candidate in self.invalid:
            item = {
                "id": candidate.selection_id,
                "kind": candidate.kind.value,
                "path": str(candidate.path),
                "diagnostics": [asdict(diagnostic) for diagnostic in candidate.diagnostics],
            }
            if (
                    candidate.kind is CollectorKind.CORE
                    or (enable_plugins and candidate.kind is CollectorKind.PLUGIN)
                    or candidate.selection_id in selected
            ):
                invalid.append(item)
            else:
                quarantined.append(item)
        return {"valid": valid, "quarantined": quarantined, "invalid": invalid}


def _diagnostic_rule(message: str, *, runtime: bool) -> str:
    """Return a stable machine-readable rule for one validator message."""
    if runtime:
        return "runtime.contract"
    normalized = message.casefold()
    rules = (
        ("filename", "static.filename"),
        ("docstring", "static.docstring"),
        ("print", "static.console_output"),
        ("import-time", "static.import_time_side_effect"),
        ("application import", "static.engine_boundary"),
        ("top-level", "static.top_level_statement"),
        ("inherit", "static.inheritance"),
        ("collector class", "static.class_name"),
        ("return", "static.return_annotation"),
        ("parameter", "static.method_signature"),
        ("classmethod", "static.method_signature"),
        ("method", "static.method_contract"),
    )
    return next((rule for marker, rule in rules if marker in normalized), "static.module_contract")


def _diagnostic_line(path: Path, message: str) -> int:
    """Locate the exact source line implicated by a normalized validation message."""
    explicit = re.search(r"\(line (\d+)\)", message)
    if explicit:
        return int(explicit.group(1))
    traceback_lines = re.findall(r'File "([^"]+)", line (\d+)', message)
    resolved = path.resolve()
    for source, line in reversed(traceback_lines):
        try:
            if Path(source).resolve() == resolved:
                return int(line)
        except OSError:
            continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        return max(1, int(getattr(error, "lineno", 1) or 1))
    method_match = re.match(r"(metadata|validate|collect|cleanup|estimate|dependencies)\b", message)
    if method_match:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_match.group(1):
                return node.lineno
    if "print" in message:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _top_level_call_name(node) == "print":
                return node.lineno
    if "class" in message or "inherit" in message:
        collector_class = next((node for node in tree.body if isinstance(node, ast.ClassDef)), None)
        if collector_class is not None:
            return collector_class.lineno
    return 1


def _pascal_case(filename: str) -> str:
    """Convert a snake-case collector filename to its required class name."""
    return "".join(part.capitalize() for part in Path(filename).stem.split("_")) + "Collector"


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a candidate path resolves inside its discovery root."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _iter_candidates(project_root: Path, kind: CollectorKind) -> list[Path]:
    """Enumerate eligible collector files for one trusted or opt-in source tree."""
    root = project_root / ("core" if kind is CollectorKind.CORE else "plugins")
    if not root.exists():
        return []
    paths: list[Path] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if any(
                part.startswith(("_", ".")) or part in {"tests", "examples", "venv", ".venv"}
                for part in relative.parts
        ):
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
    """Return a stable dotted name for a simple AST call expression."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return f"{node.func.value.id}.{node.func.attr}"
    return None


class _ImportTimeCallFinder(ast.NodeVisitor):
    """Find calls whose expressions execute while Python imports a module."""

    def __init__(self) -> None:
        """Initialize an empty call collector for import-time AST expressions."""
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Record a call and avoid duplicate nested call diagnostics."""
        self.calls.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Inspect decorators, defaults, and annotations but not a function body."""
        self._visit_callable_header(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Inspect async callable headers without treating deferred bodies as imports."""
        self._visit_callable_header(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Inspect lambda defaults without treating its deferred body as import-time work."""
        self._visit_arguments(node.args)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Inspect class headers and class-body expressions evaluated at definition time."""
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for item in node.body:
            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                continue
            self.visit(item)

    def _visit_callable_header(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit decorators, annotations, and defaults evaluated at definition time."""
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    def _visit_arguments(self, arguments: ast.arguments) -> None:
        """Visit argument defaults and annotations that execute during module import."""
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if arguments.vararg is not None and arguments.vararg.annotation is not None:
            self.visit(arguments.vararg.annotation)
        if arguments.kwarg is not None and arguments.kwarg.annotation is not None:
            self.visit(arguments.kwarg.annotation)


def _validate_import_time_expressions(tree: ast.Module, candidate: CollectorCandidate) -> None:
    """Reject calls that can collect, mutate, or exit before worker isolation exists."""
    finder = _ImportTimeCallFinder()
    finder.visit(tree)
    for call in finder.calls:
        name = _top_level_call_name(call) or ast.unparse(call.func)
        candidate.static_errors.append(f"forbidden import-time call: {name} (line {call.lineno})")


def _validate_engine_boundary_imports(tree: ast.Module, candidate: CollectorCandidate) -> None:
    """Keep collector modules on contracts only, never application or orchestration services."""
    application_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("logicytics.") and alias.name not in _COLLECTOR_SERVICE_MODULES:
                    candidate.static_errors.append(f"forbidden application import: {alias.name} (line {node.lineno})")
                elif alias.name == "logicytics":
                    application_aliases.add(alias.asname or "logicytics")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("logicytics.") and module not in _COLLECTOR_SERVICE_MODULES:
                candidate.static_errors.append(f"forbidden application import: {module} (line {node.lineno})")
            elif module == "logicytics":
                forbidden = sorted(alias.name for alias in node.names if alias.name in _APPLICATION_IMPORTS)
                if forbidden:
                    candidate.static_errors.append(
                        f"forbidden application import: {', '.join(forbidden)} (line {node.lineno})")
    for node in ast.walk(tree):
        if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in application_aliases
                and node.attr in _APPLICATION_IMPORTS
        ):
            candidate.static_errors.append(
                f"forbidden application import: {node.value.id}.{node.attr} (line {node.lineno})")


def _validate_static(path: Path, kind: CollectorKind) -> CollectorCandidate:
    """Apply filename, AST shape, import-boundary, and side-effect rules to a candidate."""
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
    if any(isinstance(node, ast.Call) and _top_level_call_name(node) == "print" for node in ast.walk(tree)):
        candidate.static_errors.append("collectors must not print; use structured progress or logging")

    public_classes = [item for item in tree.body if isinstance(item, ast.ClassDef) and not item.name.startswith("_")]
    if len(public_classes) != 1:
        candidate.static_errors.append("module must define exactly one public collector class")
    elif public_classes[0].name != expected_class:
        candidate.static_errors.append(f"collector class must be named {expected_class}")
    elif not any(
            (isinstance(base, ast.Name) and base.id == (
                    "CoreCollector" if kind is CollectorKind.CORE else "PluginCollector"))
            or (isinstance(base, ast.Attribute) and base.attr == (
                    "CoreCollector" if kind is CollectorKind.CORE else "PluginCollector"))
            for base in public_classes[0].bases
    ):
        candidate.static_errors.append("collector class must inherit from its required base class")
    elif public_classes:
        _validate_class_shape(public_classes[0], candidate)

    _validate_import_time_expressions(tree, candidate)
    _validate_engine_boundary_imports(tree, candidate)
    for item in tree.body:
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            continue
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(item, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        candidate.static_errors.append(f"forbidden top-level statement: {type(item).__name__}")
    return candidate



def _validate_class_shape(
        class_node: ast.ClassDef,
        candidate: CollectorCandidate,
) -> None:
    """Enforce the small, documented public API allowed on a collector class."""
    if ast.get_docstring(class_node) is None:
        candidate.static_errors.append("collector class requires a docstring")

    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not item.name.startswith("_"):
                methods[item.name] = item

    allowed = {
        "metadata",
        "validate",
        "prepare",
        "collect",
        "finalize",
        "cleanup",
        "estimate",
        "dependencies",
    }

    metadata_calls: list[ast.Call] = []

    metadata_method = methods.get("metadata")
    if metadata_method is not None:
        for node in ast.walk(metadata_method):
            if not isinstance(node, ast.Call):
                continue

            is_metadata_constructor = (isinstance(node.func, ast.Name) and node.func.id == "CollectorMetadata") or (
                    isinstance(node.func, ast.Attribute) and node.func.attr == "CollectorMetadata"
            )

            if is_metadata_constructor:
                metadata_calls.append(node)

    if metadata_method is not None and len(metadata_calls) != 1:
        candidate.static_errors.append("metadata must construct one CollectorMetadata object directly")

    if candidate.kind is CollectorKind.PLUGIN and metadata_method is not None:
        required_plugin_fields = {
            "capabilities",
            "privilege_level",
            "sensitive_data_categories",
            "network_access",
            "estimated_cost",
            "timeout_seconds",
            "maximum_output_bytes",
            "output_media_types",
            "minimum_contract_version",
        }

        if len(metadata_calls) == 1:
            metadata_call = metadata_calls[0]

            declared_fields = {keyword.arg for keyword in metadata_call.keywords if keyword.arg is not None}

            missing_fields = sorted(required_plugin_fields - declared_fields)

            if missing_fields:
                candidate.static_errors.append(f"plugin metadata must explicitly declare: {', '.join(missing_fields)}")

    if candidate.kind is CollectorKind.CORE and len(metadata_calls) == 1:
        if not any(keyword.arg == "capabilities" for keyword in metadata_calls[0].keywords):
            candidate.static_errors.append(
                "CAPABILITY_METADATA_MISSING: core metadata must explicitly declare capabilities")

    collect_method = methods.get("collect")

    if collect_method is not None and len(metadata_calls) == 1:
        metadata_call = metadata_calls[0]

        declared_keyword = next(
            (keyword for keyword in metadata_call.keywords if keyword.arg == "output_media_types"),
            None,
        )

        if declared_keyword is None:
            candidate.static_errors.append("metadata must explicitly declare output_media_types")
        else:
            try:
                evaluated_media_types = ast.literal_eval(declared_keyword.value)
                declared_media_types = set(evaluated_media_types)
            except (TypeError, ValueError):
                candidate.static_errors.append("metadata output_media_types must be a literal tuple")
            else:
                registered_media_types: set[str] = set()

                lifecycle_methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = [collect_method]

                finalize_method = methods.get("finalize")
                if finalize_method is not None:
                    lifecycle_methods.append(finalize_method)

                for method in lifecycle_methods:
                    for call in ast.walk(method):
                        if not isinstance(call, ast.Call):
                            continue

                        if not (isinstance(call.func, ast.Attribute) and call.func.attr == "register_file"):
                            continue

                        media_keyword = next(
                            (keyword for keyword in call.keywords if keyword.arg == "media_type"),
                            None,
                        )

                        if media_keyword is None:
                            registered_media_types.add("application/octet-stream")
                        elif isinstance(media_keyword.value, ast.Constant) and isinstance(media_keyword.value.value,
                                                                                          str):
                            registered_media_types.add(media_keyword.value.value)
                        else:
                            candidate.static_errors.append(
                                f"register_file media_type must be a literal string (line {call.lineno})")

                if not registered_media_types.issubset(declared_media_types):
                    candidate.static_errors.append(
                        "metadata output_media_types must include every registered artifact type")

    unknown = sorted(set(methods) - allowed)

    if unknown:
        candidate.static_errors.append(f"unsupported public collector methods: {', '.join(unknown)}")

    required = {
        "metadata",
        "validate",
        "collect",
        "cleanup",
    }

    expected_returns = {
        "metadata": "CollectorMetadata",
        "validate": "ValidationResult",
        "prepare": "ValidationResult",
        "collect": "CollectorResult",
        "finalize": "CollectorResult",
        "estimate": "CollectionEstimate",
        "dependencies": "tuple[str,...]",
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
        elif name in expected_returns:
            actual_return = ast.unparse(method.returns).replace(" ", "")
            expected_return = expected_returns[name]

            if actual_return != expected_return:
                candidate.static_errors.append(f"{name} must return {expected_return}")

        parameters = method.args.args

        expected_parameter_count = 1 if name in {"metadata", "dependencies"} else 3 if name == "finalize" else 2

        if len(parameters) != expected_parameter_count:
            candidate.static_errors.append(f"{name} has an invalid parameter count")
            continue

        expected_first = "cls" if name in {"metadata", "dependencies"} else "self"

        if parameters[0].arg != expected_first:
            candidate.static_errors.append(f"{name} must begin with {expected_first}")

        if name in {"metadata", "dependencies"}:
            is_classmethod = any(isinstance(decorator, ast.Name) and decorator.id == "classmethod" for decorator in
                                 method.decorator_list)

            if not is_classmethod:
                candidate.static_errors.append(f"{name} must be a classmethod")

            continue

        context_parameter = parameters[1]

        if context_parameter.annotation is None:
            context_annotation = None
        else:
            context_annotation = ast.unparse(context_parameter.annotation).rsplit(".", 1)[-1]

        if context_parameter.arg != "context" or context_annotation != "CollectorContext":
            candidate.static_errors.append(f"{name} must accept an annotated context parameter")

        if name == "finalize":
            result_parameter = parameters[2]

            if result_parameter.annotation is None:
                result_annotation = None
            else:
                result_annotation = ast.unparse(result_parameter.annotation).rsplit(".", 1)[-1]

            if result_parameter.arg != "result" or result_annotation != "CollectorResult":
                candidate.static_errors.append("finalize must accept an annotated result parameter")


def discover(project_root: Path) -> tuple[CollectorCandidate, ...]:
    """Discover candidates without importing their modules."""
    candidates = [_validate_static(path, CollectorKind.CORE) for path in
                  _iter_candidates(project_root, CollectorKind.CORE)]
    candidates.extend(
        _validate_static(path, CollectorKind.PLUGIN) for path in _iter_candidates(project_root, CollectorKind.PLUGIN))
    return tuple(candidates)


def _accept_runtime_metadata(candidate: CollectorCandidate, payload: object) -> None:
    """Validate cached or freshly probed metadata through the same strict path."""
    try:
        if not isinstance(payload, dict):
            raise ValueError("metadata payload must be an object")
        metadata = CollectorMetadata.from_dict(
            payload,
            allow_custom_specialty=candidate.kind is not CollectorKind.CORE,
        )
    except (KeyError, TypeError, ValueError) as error:
        candidate.runtime_error = f"invalid validation response: {error}"
        return

    specialty = metadata.specialty.value if isinstance(metadata.specialty, Specialty) else metadata.specialty

    if metadata.minimum_contract_version != CONTRACT_VERSION:
        candidate.runtime_error = "collector contract version is unsupported"
    elif metadata.timeout_seconds < 1 or metadata.maximum_output_bytes < 1:
        candidate.runtime_error = "collector must declare positive timeout and output limits"
    elif not metadata.id.startswith(f"{candidate.kind.value}."):
        candidate.runtime_error = "collector ID must start with its owner kind"
    elif candidate.kind is CollectorKind.CORE and specialty != candidate.path.parent.name:
        candidate.runtime_error = "core collector specialty must match its parent folder"
    elif candidate.kind is CollectorKind.CORE and metadata.id != (
            f"core.{candidate.path.parent.name}.{candidate.path.stem}"):
        candidate.runtime_error = "core collector ID must match core/<specialty>/<filename>.py"
    elif candidate.kind is CollectorKind.PLUGIN and metadata.id != (
            f"plugin.{candidate.path.parent.name if candidate.path.name == 'main.py' else candidate.path.stem}"
    ):
        candidate.runtime_error = "plugin collector ID must match its plugin folder or filename"
    else:
        candidate.metadata = metadata


def _runtime_probe(project_root: Path, candidate: CollectorCandidate, temporary_root: Path | None = None) -> None:
    """Probe metadata in a short-lived restricted worker after static validation."""
    engine_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join((str(engine_root), str(project_root)))
    command = [
        sys.executable,
        "-m",
        "logicytics.module.validation_worker",
        str(candidate.path),
        candidate.kind.value,
        candidate.expected_class,
    ]
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="logicytics-preflight-", dir=temporary_root) as temporary:
        probe_directory = Path(temporary)
        environment = {
            "PYTHONPATH": pythonpath,
            "LOGICYTICS_VALIDATION": "1",
            "PATH": os.environ.get("PATH", ""),
            "TEMP": str(probe_directory),
            "TMP": str(probe_directory),
            "PYTHONUTF8": "1",
        }
        for name in ("SYSTEMROOT", "WINDIR", "COMSPEC"):
            if value := os.environ.get(name):
                environment[name] = value
        try:
            completed = process_adapter.run(
                command,
                cwd=probe_directory,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env=environment,
            )
        except (OSError, TimeoutExpired) as error:
            candidate.runtime_error = f"validation worker failed: {error}"
            return
    if completed.returncode != 0:
        candidate.runtime_error = completed.stderr.strip() or "validation worker rejected collector"
        return
    try:
        payload = json.loads(completed.stdout)
        metadata_payload = payload["metadata"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        candidate.runtime_error = f"invalid validation response: {error}"
        return
    _accept_runtime_metadata(candidate, metadata_payload)


def _cache_path(project_root: Path, cache_directory: Path | None = None) -> Path:
    """Keep disposable validation state outside source and evidence directories."""
    project_key = hashlib.sha256(str(project_root.resolve()).encode("utf-8")).hexdigest()
    root = cache_directory if cache_directory is not None else Path(
        tempfile.gettempdir()) / "logicytics-preflight-cache"
    return root / f"{project_key}.json"


def _source_hash(path: Path) -> str:
    """Hash exact collector bytes so any source edit invalidates its probe result."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cache(project_root: Path, configuration_hash: str, cache_directory: Path | None = None) -> dict[str, object]:
    """Load only a cache created for this interpreter, contract, and configuration."""
    path = _cache_path(project_root, cache_directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    expected = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "interpreter": sys.version,
        "contract_version": CONTRACT_VERSION,
        "configuration_hash": configuration_hash,
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        return {}
    entries = payload.get("entries")
    return entries if isinstance(entries, dict) else {}


def _write_cache(
        project_root: Path,
        configuration_hash: str,
        candidates: list[CollectorCandidate],
        cache_directory: Path | None = None,
) -> None:
    """Atomically persist successful probes; invalid candidates are always reprobed."""
    entries: dict[str, object] = {}
    for candidate in candidates:
        if candidate.metadata is None or candidate.runtime_error or candidate.static_errors:
            continue
        relative_path = candidate.path.resolve().relative_to(project_root.resolve()).as_posix()
        entries[relative_path] = {
            "kind": candidate.kind.value,
            "source_hash": _source_hash(candidate.path),
            "metadata": candidate.metadata.to_dict(),
        }
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "interpreter": sys.version,
        "contract_version": CONTRACT_VERSION,
        "configuration_hash": configuration_hash,
        "entries": entries,
    }
    path = _cache_path(project_root, cache_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(
        project_root: Path,
        *,
        configuration_hash: str = "unconfigured",
        progress: PreflightProgress | None = None,
        invalidate_cache: bool = False,
        cache_directory: Path | None = None,
        temporary_directory: Path | None = None,
) -> PreflightReport:
    """Perform static checks then a short-lived isolated metadata probe."""
    if invalidate_cache:
        cache_path = _cache_path(project_root, cache_directory)
        try:
            cache_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise OSError(f"unable to invalidate preflight cache: {error}") from error
    candidates = list(discover(project_root))
    cached = _load_cache(project_root, configuration_hash, cache_directory)
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        if progress is not None:
            progress("checking", index - 1, total, candidate.selection_id)
        if candidate.static_errors:
            if progress is not None:
                progress("checked", index, total, candidate.selection_id)
            continue
        relative_path = candidate.path.resolve().relative_to(project_root.resolve()).as_posix()
        entry = cached.get(relative_path)
        if (
                isinstance(entry, dict)
                and entry.get("kind") == candidate.kind.value
                and entry.get("source_hash") == _source_hash(candidate.path)
        ):
            _accept_runtime_metadata(candidate, entry.get("metadata"))
        else:
            _runtime_probe(project_root, candidate, temporary_directory)
        if progress is not None:
            progress("checked", index, total, candidate.selection_id)
    seen_ids: set[str] = set()
    for candidate in candidates:
        if candidate.metadata is None:
            continue
        if candidate.metadata.id in seen_ids:
            candidate.runtime_error = f"duplicate collector ID: {candidate.metadata.id}"
        seen_ids.add(candidate.metadata.id)
    _write_cache(project_root, configuration_hash, candidates, cache_directory)
    return PreflightReport(tuple(candidates))
