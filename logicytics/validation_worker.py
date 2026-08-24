"""Short-lived metadata probe used only after AST-based collector validation."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
from pathlib import Path

from logicytics.contracts import (
    Collector,
    CollectorContext,
    CollectorKind,
    CollectorMetadata,
    CoreCollector,
    PluginCollector,
    ValidationResult,
)


class _ProbeLogger:
    """Discard structured events emitted during the side-effect-free validation probe."""

    def event(self, level: str, message: str, **fields: int | float | str) -> None:
        """Accept probe events without exposing them through the JSON-only worker output."""


class _ProbeArtifactWriter:
    """Reject evidence registration during preflight validation."""

    def register_file(self, source: Path, *, media_type: str = "application/octet-stream"):
        """Prevent a validation method from registering collection artifacts."""
        raise RuntimeError("validate() must not register artifacts")


def _load_module(path: Path):
    module_name = f"logicytics_probe_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to create module specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_contract(collector_type: type[Collector], kind: CollectorKind) -> CollectorMetadata:
    required_base = CoreCollector if kind is CollectorKind.CORE else PluginCollector
    if not issubclass(collector_type, required_base):
        raise ValueError(f"collector must inherit from {required_base.__name__}")
    for method_name, expected_parameters in {
        "metadata": 0,
        "validate": 1,
        "collect": 1,
        "cleanup": 1,
    }.items():
        method = getattr(collector_type, method_name, None)
        if method is None:
            raise ValueError(f"missing required method: {method_name}")
        parameters = list(inspect.signature(method).parameters.values())
        if method_name == "metadata":
            parameters = parameters[1:] if parameters and parameters[0].name == "cls" else parameters
        elif parameters and parameters[0].name == "self":
            parameters = parameters[1:]
        if len(parameters) != expected_parameters:
            raise ValueError(f"{method_name} has an invalid signature")
    dependencies = getattr(collector_type, "dependencies", None)
    if dependencies is None or list(inspect.signature(dependencies).parameters.values()):
        raise ValueError("dependencies has an invalid signature")
    declared_dependencies = dependencies()
    if not isinstance(declared_dependencies, tuple) or not all(isinstance(item, str) for item in declared_dependencies):
        raise ValueError("dependencies() must return tuple[str, ...]")
    metadata = collector_type.metadata()
    if not isinstance(metadata, CollectorMetadata):
        raise ValueError("metadata() must return CollectorMetadata")
    collector = collector_type()
    with tempfile.TemporaryDirectory(prefix="logicytics-validation-") as temporary:
        workspace = Path(temporary)
        context = CollectorContext(
            run_id="validation",
            collector_id=metadata.id,
            workspace=workspace,
            artifacts=_ProbeArtifactWriter(),
            logger=_ProbeLogger(),
            settings={},
            cancellation_file=workspace / "cancelled",
        )
        try:
            validation = collector.validate(context)
        except Exception as error:
            raise ValueError(f"validate() probe failed: {error}") from error
    if not isinstance(validation, ValidationResult):
        raise ValueError("validate() must return ValidationResult")
    return metadata


def main() -> int:
    """Validate a collector module and emit only JSON to stdout."""
    try:
        path = Path(sys.argv[1]).resolve()
        kind = CollectorKind(sys.argv[2])
        expected_class = sys.argv[3]
        sys.path.insert(0, str(path.parent))
        module = _load_module(path)
        collector_type = getattr(module, expected_class, None)
        if not inspect.isclass(collector_type):
            raise ValueError(f"missing collector class: {expected_class}")
        metadata = _validate_contract(collector_type, kind)
        print(json.dumps({"metadata": metadata.to_dict()}, sort_keys=True))
        return 0
    except (IndexError, TypeError, ValueError, ImportError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
