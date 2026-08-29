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
from logicytics.environment import inspect_environment
from logicytics.errors import LogicyticsError
from logicytics.manifest import MANIFEST_SCHEMA_VERSION
from logicytics.planner import BUILTIN_PROFILES, build_plan
from logicytics.runtime import RunSupervisor
from logicytics.sysinternals import ensure_sysinternals


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _request(arguments: argparse.Namespace, default_workers: int) -> RunRequest:
    profile = "minimal" if getattr(arguments, "minimal", False) else "deep" if getattr(arguments, "depth",
                                                                                       False) else "standard" if getattr(
        arguments, "default_mode", False) or getattr(arguments, "threaded", False) else arguments.profile
    sequential = getattr(arguments, "sequential", False)
    parallel = getattr(arguments, "parallel", False)
    performance_check = getattr(arguments, "performance_check", False)
    default_mode = getattr(arguments, "default_mode", False)
    if sequential and arguments.workers is not None and arguments.workers != 1:
        raise ValueError("sequential execution requires --workers=1")
    if parallel and (performance_check or default_mode):
        raise ValueError("parallel execution conflicts with sequential performance/default mode")
    if sequential and getattr(arguments, "threaded", False):
        raise ValueError("sequential execution conflicts with legacy --threaded mode")
    worker_count = 1 if sequential or performance_check or default_mode else arguments.workers or default_workers
    if parallel and worker_count < 2:
        raise ValueError("parallel execution requires at least two configured workers")
    parent_run_id: str | None = None
    if rerun_path := getattr(arguments, "rerun_from", None):
        if not arguments.include:
            raise ValueError("--rerun-from requires at least one explicit --include collector ID")
        manifest_path = rerun_path / "manifest.json" if rerun_path.is_dir() else rerun_path
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"original run manifest cannot be loaded: {error}") from error
        if not isinstance(previous, dict) or not isinstance(previous.get("run_id"), str):
            raise ValueError("original run manifest must contain a valid run_id")
        manifest_schema_version = previous.get("manifest_schema_version")
        if (
                not isinstance(manifest_schema_version, int)
                or isinstance(manifest_schema_version, bool)
                or manifest_schema_version != MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                f"original run manifest uses unsupported schema_version {manifest_schema_version!r}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        if previous.get("status") not in {"succeeded", "partial", "failed", "cancelled"}:
            raise ValueError("original run manifest must describe a finalized run")
        resolved = previous.get("resolved_plan")
        if not isinstance(resolved, list) or not all(isinstance(item, str) for item in resolved):
            raise ValueError("original run manifest must contain a valid resolved_plan")
        unknown = sorted(set(arguments.include) - set(resolved))
        if unknown:
            raise ValueError(f"rerun collectors were not present in the original run: {', '.join(unknown)}")
        parent_run_id = previous["run_id"]
    return RunRequest(
        profile=profile,
        include=tuple(arguments.include),
        exclude=tuple(arguments.exclude),
        enable_plugins=arguments.plugins,
        max_workers=worker_count,
        acknowledge_authorization=getattr(arguments, "acknowledge_authorization", False),
        approved_capabilities=tuple(Capability(value) for value in arguments.allow_capability),
        performance_check=performance_check,
        rerun_from=parent_run_id,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Logicytics v4 run-oriented evidence framework")
    parser.add_argument("--config", type=Path, help="Path to a v4 JSON configuration file")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "debug", "update", "plan", "run"):
        subparser = subcommands.add_parser(command)
        subparser.add_argument(
            "--profile",
            default="standard",
            choices=tuple(BUILTIN_PROFILES),
            help="Named built-in collector membership and access policy.",
        )
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
                "--rerun-from",
                type=Path,
                help="Rerun explicitly included collector IDs from a finalized run manifest or directory.",
            )
            execution = subparser.add_mutually_exclusive_group()
            execution.add_argument(
                "--sequential",
                action="store_true",
                help="Explicitly run isolated collectors one at a time for deterministic debugging.",
            )
            execution.add_argument(
                "--parallel",
                action="store_true",
                help="Explicitly run isolated collectors with the configured bounded worker limit.",
            )
            mode = subparser.add_mutually_exclusive_group()
            mode.add_argument("--default", dest="default_mode", action="store_true",
                              help="Run the standard built-in profile.")
            mode.add_argument("--threaded", action="store_true",
                              help="Run the standard built-in profile with configured parallel workers.")
            mode.add_argument("--minimal", action="store_true", help="Run the minimal built-in profile.")
            mode.add_argument("--depth", action="store_true", help="Run the deep built-in profile.")
            subparser.add_argument(
                "--performance-check",
                action="store_true",
                help="Run serially and save per-collector duration measurements.",
            )
            subparser.add_argument(
                "--acknowledge-authorization",
                action="store_true",
                help="Confirm you are authorized to collect the selected evidence.",
            )
        if command == "update":
            subparser.add_argument("--apply", action="store_true",
                                   help="Explicitly run git pull after repository checks.")
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
        report = preflight(root, configuration_hash=configuration.fingerprint())
        if arguments.command == "preflight":
            validation = report.to_dict(
                selected_plugins=tuple(arguments.include),
                enable_plugins=arguments.plugins,
            )
            payload = {
                "environment": inspect_environment().to_dict(),
                "sysinternals": ensure_sysinternals(root).to_dict(),
                **validation,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not validation["invalid"] else 2
        if arguments.command == "debug":
            payload = {
                "configuration": configuration.to_manifest_dict(),
                "environment": inspect_environment().to_dict(),
                "python": {"executable": sys.executable, "implementation": platform.python_implementation(),
                           "version": platform.python_version(), "prefix": sys.prefix,
                           "virtual_environment": sys.prefix != sys.base_prefix,
                           "psutil_available": importlib.util.find_spec("psutil") is not None,
                           "cpu_count": os.cpu_count()},
                "sysinternals": ensure_sysinternals(root).to_dict(),
                "preflight": {"valid_collectors": len(report.valid), "invalid_collectors": len(report.invalid)},
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not report.invalid else 2
        if arguments.command == "update":
            git = subprocess.run(["git", "--version"], capture_output=True, check=False, text=True)
            is_repository = (root / ".git").exists()
            payload = {"git_available": git.returncode == 0, "git_version": git.stdout.strip() or None,
                       "is_repository": is_repository, "applied": False}
            if arguments.apply:
                if git.returncode != 0 or not is_repository:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                    return 2
                pulled = subprocess.run(["git", "pull"], cwd=root, capture_output=True, check=False, text=True)
                payload.update({"applied": True, "returncode": pulled.returncode, "stdout": pulled.stdout,
                                "stderr": pulled.stderr})
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not arguments.apply or payload.get("returncode") == 0 else 1
        plan = build_plan(report, _request(arguments, configuration.runtime.default_max_workers))
        if arguments.command == "plan":
            print("\n".join(candidate.metadata.id for candidate in plan.collectors if candidate.metadata))
            return 0
        outcome = RunSupervisor(root, configuration).run(plan)
        print("Collectors:")
        for record in outcome.manifest.collectors:
            duration = "not-started" if record.duration_seconds is None else f"{record.duration_seconds:.3f}"
            print(
                f"- {record.id} status={record.status} duration_seconds={duration} "
                f"summary={record.summary or 'not-finished'}"
            )
            if record.failure is not None:
                print(
                    f"  failure operation={record.failure['operation']} "
                    f"retry_safe={str(record.failure['retry_safe']).lower()} "
                    f"platform_error={record.failure['platform_error']} "
                    f"remediation={record.failure['remediation']}"
                )
        if arguments.performance_check:
            performance_path = outcome.run_directory / "logs" / "performance.json"
            print(f"Performance: {performance_path}")
        if outcome.manifest.package and "path" in outcome.manifest.package:
            print(
                f"Package: {outcome.manifest.package['path']}\n"
                f"SHA-256: {outcome.manifest.package.get('sha256_path', 'unavailable')}"
            )
        print(f"Run: {outcome.manifest_path}\nStatus: {outcome.manifest.status.value}")
        return 0 if outcome.manifest.status.value == "succeeded" else 1
    except (LogicyticsError, PermissionError, ValueError) as error:
        print(f"Error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
