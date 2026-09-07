"""Bootstrap the supported virtual environment and root YAML configuration."""

from __future__ import annotations

import argparse
import venv
from pathlib import Path

from logicytics.module.configuration import write_default_configuration


def project_root() -> Path:
    """Return the repository root without importing the normal application CLI."""
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Create the local virtual environment and initialize YAML configuration."""
    parser = argparse.ArgumentParser(description="Prepare a Logicytics installation.")
    parser.add_argument("--environment", type=Path, default=Path(".venv"), help="Virtual environment directory.")
    parser.add_argument("--overwrite-config", action="store_true", help="Replace the root YAML template.")
    arguments = parser.parse_args(argv)
    root = project_root()
    environment = arguments.environment if arguments.environment.is_absolute() else root / arguments.environment
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
        print(f"Created virtual environment: {environment}")
    else:
        print(f"Using existing virtual environment: {environment}")
    configuration = write_default_configuration(root, overwrite=arguments.overwrite_config)
    print(f"Configuration ready: {configuration}")
    print(f"Run: {environment / 'Scripts' / 'python.exe'} -m logicytics preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
