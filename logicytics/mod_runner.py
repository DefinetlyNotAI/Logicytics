"""Audit-confined bootstrap for legacy Python MODS scripts."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run one copied Python MOD with workspace mutation and capability enforcement."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 4:
        print("Python MOD runner requires script, workspace, collector ID, and capabilities", file=sys.stderr)
        return 2
    script = Path(arguments[0]).resolve()
    workspace = Path(arguments[1]).resolve()
    collector_id = arguments[2]
    try:
        capability_values = json.loads(arguments[3])
        if not isinstance(capability_values, list):
            raise ValueError("capabilities must be a JSON list")
    except (json.JSONDecodeError, ValueError) as error:
        print(f"Invalid Python MOD capabilities: {error}", file=sys.stderr)
        return 2

    engine_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(engine_root))
    from logicytics.contracts import Capability
    from logicytics.runtime import _WorkerMutationGuard

    try:
        capabilities = tuple(Capability(value) for value in capability_values)
    except ValueError as error:
        print(f"Invalid Python MOD capability: {error}", file=sys.stderr)
        return 2
    try:
        script.relative_to(workspace)
    except ValueError:
        print("Python MOD script must be copied inside its private workspace", file=sys.stderr)
        return 2

    guard = _WorkerMutationGuard(
        workspace,
        workspace / "published",
        collector_id,
        capabilities,
        script,
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
