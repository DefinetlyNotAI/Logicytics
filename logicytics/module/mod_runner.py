"""Audit-confined bootstrap for legacy Python MODS scripts."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run one copied Python MOD with workspace mutation and capability enforcement."""
    arguments = sys.argv[1:] if argv is None else argv
    engine_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(engine_root))
    from logicytics.module.logging import ApplicationLogger

    if len(arguments) != 5:
        ApplicationLogger.render_section(
            sys.stderr,
            "Python MOD runner error",
            (
                "The runner requires script, workspace, collector ID, "
                "capabilities, and blocked capabilities.",
            ),
        )
        return 2
    script = Path(arguments[0]).resolve()
    workspace = Path(arguments[1]).resolve()
    collector_id = arguments[2]
    try:
        capability_values = json.loads(arguments[3])
        if not isinstance(capability_values, list):
            raise ValueError("capabilities must be a JSON list")
        blocked_capability_values = json.loads(arguments[4])
        if not isinstance(blocked_capability_values, list):
            raise ValueError("blocked capabilities must be a JSON list")
    except (json.JSONDecodeError, ValueError) as error:
        ApplicationLogger.render_section(
            sys.stderr,
            "Python MOD runner error",
            (f"Invalid capabilities: {error}",),
        )
        return 2

    from logicytics.contracts import Capability
    from logicytics.module.runtime import _WorkerMutationGuard

    try:
        capabilities = tuple(Capability(value) for value in capability_values)
        blocked_capabilities = tuple(Capability(value) for value in blocked_capability_values)
    except ValueError as error:
        ApplicationLogger.render_section(
            sys.stderr,
            "Python MOD runner error",
            (f"Invalid capability: {error}",),
        )
        return 2
    try:
        script.relative_to(workspace)
    except ValueError:
        ApplicationLogger.render_section(
            sys.stderr,
            "Python MOD runner error",
            ("The script must be copied inside its private workspace.",),
        )
        return 2

    guard = _WorkerMutationGuard(
        workspace,
        workspace / "published",
        collector_id,
        capabilities,
        script,
        blocked_capabilities,
    )
    guard.active = True
    previous_argv = sys.argv
    try:
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        guard.active = False
        sys.argv = previous_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
