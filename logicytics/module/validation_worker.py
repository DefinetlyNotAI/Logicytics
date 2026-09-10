"""Short-lived metadata probe used only after AST-based collector validation."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

from logicytics.contracts import (
    Artifact,
    ArtifactWriter,
    Collector,
    CollectorContext,
    CollectorKind,
    CollectorMetadata,
    CoreCollector,
    EventLogger,
    EvidenceKind,
    PluginCollector,
    ValidationResult,
)
from logicytics.module.logging import ApplicationLogger


class _ProbeLogger(EventLogger):
    """Discard structured events emitted during the side-effect-free validation probe."""

    def event(self, level: str, message: str, **fields: float | str) -> None:
        """Accept probe events without exposing them through the JSON-only worker output."""


class _ProbeArtifactWriter(ArtifactWriter):
    """Reject evidence registration during preflight validation."""

    def register_file(
            self,
            source: Path,
            *,
            media_type: str = "application/octet-stream",
            evidence_kind: EvidenceKind = EvidenceKind.DERIVED,
            transformations: tuple[str, ...] = (),
    ) -> Artifact:
        """Prevent a validation method from registering collection artifacts."""
        raise RuntimeError("validate() must not register artifacts")


class _ValidationSideEffectGuard:
    """Reject audited mutations only while a collector validation method is active."""

    _BLOCKED_EVENTS = {
        "os.chmod",
        "os.chown",
        "os.link",
        "os.mkdir",
        "os.putenv",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.symlink",
        "os.truncate",
        "os.unsetenv",
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.sendto",
        "subprocess.Popen",
        "winreg.CreateKey",
        "winreg.DeleteKey",
        "winreg.DeleteValue",
        "winreg.SetValue",
    }

    def __init__(self) -> None:
        """Initialize an inactive audit guard for the validation context."""
        self.active = False

    def __enter__(self) -> _ValidationSideEffectGuard:
        """Install the audit hook and begin rejecting validation side effects."""
        sys.addaudithook(self._reject_side_effect)
        self.active = True
        return self

    def __exit__(self, *_: object) -> None:
        """Stop enforcing validation side-effect checks after the probe exits."""
        self.active = False

    def _reject_side_effect(self, event: str, arguments: tuple[object, ...]) -> None:
        """Raise when an active validation probe attempts a blocked mutation."""
        if not self.active:
            return
        if event == "open":
            mode = arguments[1] if len(arguments) > 1 else None
            flags = arguments[2] if len(arguments) > 2 else 0
            writing = isinstance(mode, str) and any(flag in mode for flag in "wax+")
            writing = writing or isinstance(flags, int) and bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if not writing:
                return
        elif event not in self._BLOCKED_EVENTS:
            return
        raise RuntimeError(f"validate() must not perform side effects: {event}")


def _load_module(path: Path):
    """Load one collector source module under an isolated probe module name."""
    module_name = f"logicytics_probe_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to create module specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_contract(
        collector_type: type[Collector],
        kind: CollectorKind,
) -> CollectorMetadata:
    """Validate collector signatures, metadata consistency, and probe purity."""
    required_base = CoreCollector if kind is CollectorKind.CORE else PluginCollector

    if not issubclass(collector_type, required_base):
        raise ValueError(f"collector must inherit from {required_base.__name__}")

    for method_name, expected_parameters in {
        "metadata": 0,
        "validate": 1,
        "prepare": 1,
        "collect": 1,
        "finalize": 2,
        "cleanup": 1,
    }.items():
        method = getattr(collector_type, method_name, None)

        if method is None or not callable(method):
            raise ValueError(f"missing required method: {method_name}")

        parameters = list(inspect.signature(method).parameters.values())

        if method_name == "metadata":
            parameters = parameters[1:] if parameters and parameters[0].name == "cls" else parameters
        elif parameters and parameters[0].name == "self":
            parameters = parameters[1:]

        if len(parameters) != expected_parameters:
            raise ValueError(f"{method_name} has an invalid signature")

    raw_dependencies = getattr(collector_type, "dependencies", None)

    if raw_dependencies is None or not callable(raw_dependencies):
        raise ValueError("dependencies has an invalid signature")

    dependencies = cast(
        Callable[[], tuple[str, ...]],
        raw_dependencies,
    )

    if inspect.signature(dependencies).parameters:
        raise ValueError("dependencies has an invalid signature")

    declared_dependencies = dependencies()

    if not isinstance(declared_dependencies, tuple) or not all(isinstance(item, str) for item in declared_dependencies):
        raise ValueError("dependencies() must return tuple[str, ...]")

    metadata = collector_type.metadata()

    if not isinstance(metadata, CollectorMetadata):
        raise ValueError("metadata() must return CollectorMetadata")

    if declared_dependencies != metadata.dependencies:
        raise ValueError("dependencies() must match metadata.dependencies")

    collector = collector_type()

    with tempfile.TemporaryDirectory(prefix="logicytics-validation-") as temporary:
        workspace = Path(temporary)
        temporary_directory = workspace / "tmp"
        temporary_directory.mkdir()

        context = CollectorContext(
            run_id="validation",
            collector_id=metadata.id,
            workspace=workspace,
            temporary_directory=temporary_directory,
            artifacts=_ProbeArtifactWriter(),
            logger=_ProbeLogger(),
            settings={},
            cancellation_file=workspace / "cancelled",
        )

        try:
            with _ValidationSideEffectGuard():
                validation = collector.validate(context)
        except Exception as error:
            raise ValueError(f"validate() probe failed: {error}") from error

    if not isinstance(validation, ValidationResult):
        raise ValueError("validate() must return ValidationResult")

    return metadata


def main() -> int:
    """Validate a collector module and keep JSON stdout private to the parent worker."""
    try:
        path = Path(sys.argv[1]).resolve()
        kind = CollectorKind(sys.argv[2])
        expected_class = sys.argv[3]
        sys.path.insert(0, str(path.parent))
        module = _load_module(path)
        collector_object = getattr(module, expected_class, None)

        if not inspect.isclass(collector_object):
            raise ValueError(f"missing collector class: {expected_class}")

        collector_type = cast(type[Collector], collector_object)

        if not issubclass(collector_type, Collector):
            raise ValueError(f"{expected_class} is not a Collector")

        metadata = _validate_contract(collector_type, kind)
        protocol_message = json.dumps({"metadata": metadata.to_dict()}, sort_keys=True)
        if sys.stdout.isatty():
            ApplicationLogger.render_section(
                sys.stdout,
                "Collector validation",
                ("Metadata validated successfully.",),
            )
        else:
            sys.stdout.write(protocol_message + "\n")
        return 0
    except (IndexError, TypeError, ValueError, ImportError, OSError) as error:
        ApplicationLogger.render_section(
            sys.stderr,
            "Collector validation error",
            (str(error),),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
