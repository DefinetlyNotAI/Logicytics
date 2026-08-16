"""Minimal command-line interface for v4 planning and supervised execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from logicytics.configuration import load_config
from logicytics.contracts import Capability, RunRequest
from logicytics.discovery import preflight
from logicytics.errors import LogicyticsError
from logicytics.environment import inspect_environment
from logicytics.packaging import package_run
from logicytics.planner import build_plan
from logicytics.runtime import RunSupervisor
from logicytics.sysinternals import ensure_sysinternals


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _request(arguments: argparse.Namespace, default_workers: int) -> RunRequest:
    profile = "minimal" if getattr(arguments, "minimal", False) else "deep" if getattr(arguments, "depth", False) else "standard" if getattr(arguments, "default_mode", False) or getattr(arguments, "threaded", False) else arguments.profile
    return RunRequest(
        profile=profile,
        include=tuple(arguments.include),
        exclude=tuple(arguments.exclude),
        enable_plugins=arguments.plugins,
        max_workers=1 if getattr(arguments, "performance_check", False) or getattr(arguments, "default_mode", False) else arguments.workers or default_workers,
        acknowledge_authorization=getattr(arguments, "acknowledge_authorization", False),
        approved_capabilities=tuple(Capability(value) for value in arguments.allow_capability),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Logicytics v4 run-oriented evidence framework")
    parser.add_argument("--config", type=Path, help="Path to a v4 JSON configuration file")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "debug", "update", "plan", "run"):
        subparser = subcommands.add_parser(command)
        subparser.add_argument("--profile", default="standard")
        subparser.add_argument("--include", action="append", default=[])
        subparser.add_argument("--exclude", action="append", default=[])
        subparser.add_argument("--plugins", action="store_true")
        subparser.add_argument("--workers", type=int)
        subparser.add_argument(
            "--allow-capability",
            action="append",
            default=[],
            choices=[capability.value for capability in Capability],
            help="Approve an access capability requested by the selected collectors.",
        )
        if command == "run":
            mode = subparser.add_mutually_exclusive_group()
            mode.add_argument("--default", dest="default_mode", action="store_true", help="Run the standard built-in profile.")
            mode.add_argument("--threaded", action="store_true", help="Run the standard built-in profile with configured parallel workers.")
            mode.add_argument("--minimal", action="store_true", help="Run the minimal built-in profile.")
            mode.add_argument("--depth", action="store_true", help="Run the deep built-in profile.")
            subparser.add_argument(
                "--acknowledge-authorization",
                action="store_true",
                help="Confirm you are authorized to collect the selected evidence.",
            )
        if command == "update":
            subparser.add_argument("--apply", action="store_true", help="Explicitly run git pull after repository checks.")
            subparser.add_argument(
                "--performance-check",
                action="store_true",
                help="Run collectors sequentially and write a per-collector duration report.",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected preflight, planning, or supervised execution command."""
    arguments = _parser().parse_args(argv)
    root = _project_root()
    try:
        configuration = load_config(root, arguments.config)
        report = preflight(root)
        if arguments.command == "preflight":
            payload = {
                "environment": inspect_environment().to_dict(),
                "sysinternals": ensure_sysinternals(root).to_dict(),
                "valid": [candidate.metadata.id for candidate in report.valid if candidate.metadata],
                "invalid": [
                    {"path": str(candidate.path), "errors": candidate.static_errors or [candidate.runtime_error]}
                    for candidate in report.invalid
                ],
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not report.invalid else 2
        if arguments.command == "debug":
            payload = {
                "configuration": configuration.to_manifest_dict(),
                "environment": inspect_environment().to_dict(),
                "python": {"executable": sys.executable, "implementation": platform.python_implementation(), "version": platform.python_version(), "prefix": sys.prefix, "virtual_environment": sys.prefix != sys.base_prefix, "psutil_available": importlib.util.find_spec("psutil") is not None, "cpu_count": os.cpu_count()},
                "sysinternals": ensure_sysinternals(root).to_dict(),
                "preflight": {"valid_collectors": len(report.valid), "invalid_collectors": len(report.invalid)},
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not report.invalid else 2
        if arguments.command == "update":
            git = subprocess.run(["git", "--version"], capture_output=True, check=False, text=True)
            is_repository = (root / ".git").exists()
            payload = {"git_available": git.returncode == 0, "git_version": git.stdout.strip() or None, "is_repository": is_repository, "applied": False}
            if arguments.apply:
                if git.returncode != 0 or not is_repository:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                    return 2
                pulled = subprocess.run(["git", "pull"], cwd=root, capture_output=True, check=False, text=True)
                payload.update({"applied": True, "returncode": pulled.returncode, "stdout": pulled.stdout, "stderr": pulled.stderr})
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not arguments.apply or payload.get("returncode") == 0 else 1
        plan = build_plan(report, _request(arguments, configuration.runtime.default_max_workers))
        if arguments.command == "plan":
            print("\n".join(candidate.metadata.id for candidate in plan.collectors if candidate.metadata))
            return 0
        outcome = RunSupervisor(root, configuration).run(plan)
        if arguments.performance_check:
            performance_path = outcome.run_directory / "logs" / "performance.json"
            performance_path.write_text(
                json.dumps(
                    {
                        "run_id": outcome.manifest.run_id,
                        "collectors": [
                            {"id": record.id, "status": record.status, "duration_seconds": record.duration_seconds}
                            for record in outcome.manifest.collectors
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            print(f"Performance: {performance_path}")
        if configuration.runtime.package_completed_runs:
            package_path, hash_path = package_run(outcome)
            print(f"Package: {package_path}\nSHA-256: {hash_path}")
        print(f"Run: {outcome.manifest_path}\nStatus: {outcome.manifest.status.value}")
        return 0 if outcome.manifest.status.value == "succeeded" else 1
    except (LogicyticsError, PermissionError, ValueError) as error:
        print(f"Error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
