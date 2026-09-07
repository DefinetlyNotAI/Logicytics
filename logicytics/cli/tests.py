"""Discover and execute the complete Logicytics test suite."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


def _require_virtual_environment() -> None:
    """Reject normal tooling execution outside the environment prepared by installer."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Logicytics tests require a virtual environment. Run logicytics.cli.installer first.")


def main(argv: list[str] | None = None) -> int:
    """Dynamically discover tests and return a CI-appropriate result code."""
    parser = argparse.ArgumentParser(description="Run all discovered Logicytics tests.")
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=2)
    arguments = parser.parse_args(argv)
    try:
        _require_virtual_environment()
    except RuntimeError as error:
        print(f"ERROR | {error}", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[2]
    result = unittest.TextTestRunner(verbosity=arguments.verbosity).run(
        unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
    )
    print(
        f"Result: failures={len(result.failures)} errors={len(result.errors)} skipped={len(result.skipped)} "
        f"tests={result.testsRun}"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
