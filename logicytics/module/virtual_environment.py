"""Virtual-environment guardrails shared by runnable Logicytics commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from logicytics.module.presentation import render_alert

_LOCAL_ENVIRONMENT = Path(".venv")
_ACTIVATION_SCRIPT = Path("Scripts") / "Activate.ps1"


def is_running_in_virtual_environment() -> bool:
    """Return whether the active interpreter is backed by a real virtual environment."""
    if sys.prefix == sys.base_prefix and not hasattr(sys, "real_prefix"):
        return False
    return (Path(sys.prefix) / "pyvenv.cfg").is_file()


def virtual_environment_details() -> dict[str, str | bool | None]:
    """Describe the exact interpreter evidence used by the venv guard."""
    environment_root = Path(sys.prefix).resolve()
    configuration = environment_root / "pyvenv.cfg"
    return {
        "active": is_running_in_virtual_environment(),
        "executable": str(Path(sys.executable).resolve()),
        "prefix": str(environment_root),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "configuration": str(configuration) if configuration.is_file() else None,
    }


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
        "Next step: python -m logicytics.cli.installer",
    )


def render_virtual_environment_error(stream: TextIO, root: Path) -> None:
    """Render the startup guard through the shared logging presentation format."""
    render_alert(stream, "Logicytics startup error", virtual_environment_error(root))
