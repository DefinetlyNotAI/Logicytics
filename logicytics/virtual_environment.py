"""Virtual-environment guardrails shared by runnable Logicytics commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


_LOCAL_ENVIRONMENT = Path(".venv")
_ACTIVATION_SCRIPT = Path("Scripts") / "Activate.ps1"


def is_running_in_virtual_environment() -> bool:
    """Return whether the active interpreter belongs to a virtual environment."""
    return sys.prefix != sys.base_prefix


def virtual_environment_error(root: Path) -> tuple[str, str]:
    """Give the actionable next step for a command started outside a virtual environment."""
    activation_script = root / _LOCAL_ENVIRONMENT / _ACTIVATION_SCRIPT
    if activation_script.is_file():
        return (
            "Logicytics must run inside a virtual environment.",
            r"Next step: .\.venv\Scripts\Activate.ps1",
        )
    return (
        "Logicytics must run inside a virtual environment.",
        "Next step: run python -m logicytics.cli.installer first.",
    )


def render_virtual_environment_error(stream: TextIO, root: Path) -> None:
    """Write the startup guard error without importing the normal CLI stack."""
    stream.write("Logicytics startup error\n")
    for detail in virtual_environment_error(root):
        stream.write(f"  {detail}\n")
