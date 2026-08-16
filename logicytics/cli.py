"""Minimal command-line interface for v4 planning and supervised execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from logicytics.configuration import load_config
from logicytics.contracts import Capability, RunRequest
from logicytics.discovery import preflight
from logicytics.errors import LogicyticsError
from logicytics.packaging import package_run
from logicytics.planner import build_plan
from logicytics.runtime import RunSupervisor


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _request(arguments: argparse.Namespace, default_workers: int) -> RunRequest:
    return RunRequest(
        profile=arguments.profile,
        include=tuple(arguments.include),
        exclude=tuple(arguments.exclude),
        enable_plugins=arguments.plugins,
        max_workers=arguments.workers or default_workers,
        acknowledge_authorization=getattr(arguments, "acknowledge_authorization", False),
        approved_capabilities=tuple(Capability(value) for value in arguments.allow_capability),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Logicytics v4 run-oriented evidence framework")
    parser.add_argument("--config", type=Path, help="Path to a v4 JSON configuration file")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "plan", "run"):
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
            subparser.add_argument(
                "--acknowledge-authorization",
                action="store_true",
                help="Confirm you are authorized to collect the selected evidence.",
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
                "valid": [candidate.metadata.id for candidate in report.valid if candidate.metadata],
                "invalid": [
                    {"path": str(candidate.path), "errors": candidate.static_errors or [candidate.runtime_error]}
                    for candidate in report.invalid
                ],
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not report.invalid else 2
        plan = build_plan(report, _request(arguments, configuration.runtime.default_max_workers))
        if arguments.command == "plan":
            print("\n".join(candidate.metadata.id for candidate in plan.collectors if candidate.metadata))
            return 0
        outcome = RunSupervisor(root, configuration).run(plan)
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
