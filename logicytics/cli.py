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

from logicytics.configuration import AppConfig, load_config
from logicytics.contracts import Capability, OutputPolicy, PostRunAction, RunRequest
from logicytics.discovery import preflight
from logicytics.environment import inspect_environment
from logicytics.errors import LogicyticsError
from logicytics.manifest import MANIFEST_SCHEMA_VERSION
from logicytics.interaction import load_history, match_flag, record_match, usage_statistics, write_usage_graph
from logicytics.maintenance import (
    build_manifest,
    compare_files,
    developer_checks,
    load_local_manifest,
    local_version,
    maintenance_diagnostics,
    write_local_manifest,
)
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
    modded = getattr(arguments, "modded", False)
    nopy = getattr(arguments, "nopy", False)
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
        enable_mods=getattr(arguments, "mods", False) or modded or nopy,
        non_python_only=nopy,
        max_workers=worker_count,
        acknowledge_authorization=getattr(arguments, "acknowledge_authorization", False),
        approved_capabilities=tuple(Capability(value) for value in arguments.allow_capability),
        performance_check=performance_check,
        rerun_from=parent_run_id,
        output_policy=(
            OutputPolicy.MANIFEST_ONLY
            if getattr(arguments, "no_package", False)
            else OutputPolicy.PACKAGE
        ),
        post_run_action=(
            PostRunAction.REBOOT
            if getattr(arguments, "reboot", False)
            else PostRunAction.SHUTDOWN
            if getattr(arguments, "shutdown", False)
            else PostRunAction.NONE
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Logicytics v4 run-oriented evidence framework")
    parser.add_argument("--config", type=Path, help="Path to a v4 JSON configuration file")
    parser.add_argument("--usage", action="store_true", help="Show local interaction statistics and create a usage graph.")
    parser.add_argument("--match", metavar="TEXT", help="Suggest the closest documented action for natural-language input.")
    subcommands = parser.add_subparsers(dest="command")
    for command in ("preflight", "debug", "update", "dev", "plan", "run"):
        subparser = subcommands.add_parser(command, help=f"Run the {command} action.")
        subparser.add_argument(
            "--profile",
            default="standard",
            choices=tuple(BUILTIN_PROFILES),
            help="Named built-in collector membership and access policy.",
        )
        subparser.add_argument(
            "--include", action="append", default=[], metavar="COLLECTOR_ID",
            help="Include a collector by exact dotted ID; repeat for multiple collectors.",
        )
        subparser.add_argument(
            "--exclude", action="append", default=[], metavar="COLLECTOR_ID",
            help="Exclude a collector by exact dotted ID; repeat for multiple collectors.",
        )
        subparser.add_argument(
            "--plugins", action="store_true",
            help="Enable all valid opt-in plugin collectors for the selected profile.",
        )
        subparser.add_argument(
            "--mods", action="store_true",
            help="Enable valid sidecar-declared scripts from the MODS directory.",
        )
        subparser.add_argument(
            "--workers", type=int, metavar="COUNT",
            help="Bound concurrent isolated workers to this positive count.",
        )
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
            mode.add_argument("--modded", action="store_true", help="Run the standard profile plus all valid MODS scripts.")
            mode.add_argument("--nopy", action="store_true", help="Run only non-Python PowerShell, batch, and executable MODS scripts.")
            mode.add_argument(
                "--performance-check",
                action="store_true",
                help="Run serially and save per-collector duration measurements.",
            )
            subparser.add_argument(
                "--no-package",
                action="store_true",
                help="Finalize the run manifest without creating a ZIP package.",
            )
            power_action = subparser.add_mutually_exclusive_group()
            power_action.add_argument(
                "--reboot",
                action="store_true",
                help="Schedule a reboot only after the run is packaged successfully.",
            )
            power_action.add_argument(
                "--shutdown",
                action="store_true",
                help="Schedule a shutdown only after the run is packaged successfully.",
            )
            subparser.add_argument(
                "--acknowledge-authorization",
                action="store_true",
                help="Confirm you are authorized to collect the selected evidence.",
            )
            subparser.add_argument(
                "--interactive",
                action="store_true",
                help="Pause at the final status so an interactive command window remains visible.",
            )
        if command == "update":
            subparser.add_argument("--apply", action="store_true",
                                   help="Explicitly run git pull after repository checks.")
            subparser.add_argument(
                "--performance-check",
                action="store_true",
                help="Run collectors sequentially and write a per-collector duration report.",
            )
        if command == "dev":
            subparser.add_argument(
                "--write-manifest",
                action="store_true",
                help=(
                    "Write the reviewed local integrity manifest; requires --next-version "
                    "unless interactive."
                ),
            )
            subparser.add_argument(
                "--next-version",
                help=(
                    "Semantic version to validate and record in a newly written "
                    "integrity manifest."
                ),
            )
            subparser.add_argument(
                "--interactive",
                action="store_true",
                help=(
                    "Show contribution checks and prompt before changing the local "
                    "integrity manifest."
                ),
            )
    return parser


def _write_json(path: Path, payload: object) -> Path:
    """Atomically write a human-readable JSON diagnostic artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _status_marker(label: str, count: int) -> str:
    """Return a colored interactive repository-status marker when a terminal supports it."""
    color = "\033[32m" if count == 0 else "\033[33m"
    reset = "\033[0m"
    return f"{color}{label}: {count}{reset}" if sys.stdout.isatty() else f"{label}: {count}"


def _run_developer_action(
    root: Path, configuration: AppConfig, arguments: argparse.Namespace
) -> int:
    """Run read-only contribution checks and an explicitly confirmed manifest update."""
    settings = configuration.maintenance
    existing = load_local_manifest(root, settings)
    comparison = (
        compare_files(root, settings, existing)
        if existing
        else {"missing": [], "modified": [], "extra": [], "unchanged": []}
    )
    checks = developer_checks(root, settings)
    next_version = arguments.next_version
    write_requested = arguments.write_manifest
    if arguments.interactive:
        print("Contribution and repository organization checks")
        for name in ("missing", "modified", "extra", "unchanged"):
            print(_status_marker(name.title(), len(comparison[name])))
        organization_checks = (
            "naming_violations",
            "misplaced_python",
            "missing_module_docstrings",
            "crowded_modules",
        )
        for name in organization_checks:
            print(_status_marker(name.replace("_", " ").title(), len(checks[name])))
        if next_version is None:
            entered = input(f"Next semantic version [{local_version(root)}]: ").strip()
            next_version = entered or local_version(root)
        answer = input("Write the reviewed integrity manifest? [y/N]: ").strip().casefold()
        write_requested = answer in {"y", "yes"}
    if arguments.write_manifest and next_version is None:
        raise ValueError("--write-manifest requires --next-version unless --interactive is used")
    manifest_path: str | None = None
    if write_requested:
        if next_version is None:
            raise ValueError("a semantic next version is required to write the integrity manifest")
        manifest = build_manifest(root, settings, next_version)
        manifest_path = str(write_local_manifest(root, settings, manifest))
    payload = {
        "checks": checks,
        "comparison": comparison,
        "current_version": local_version(root),
        "existing_manifest_version": existing.version if existing else None,
        "manifest_written": manifest_path,
        "next_version": next_version,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the selected preflight, planning, or supervised execution command."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.usage and arguments.match:
        parser.error("--usage cannot be combined with --match")
    if arguments.command is not None and (arguments.usage or arguments.match):
        parser.error("--usage and --match are standalone actions")
    if arguments.usage:
        arguments.command = "usage"
    elif arguments.match:
        arguments.command = "match"
    if arguments.command is None:
        parser.print_help()
        return 0
    root = _project_root()
    try:
        configuration = load_config(root, arguments.config)
        history_path = configuration.runtime.output_root / "interaction_history.json.gz"
        if arguments.command == "match":
            history = load_history(history_path)
            match = match_flag(
                arguments.match,
                threshold=configuration.interaction.similarity_threshold,
                model_name=configuration.interaction.model_name,
                history=history,
            )
            if configuration.interaction.history_enabled:
                record_match(history_path, match)
            payload = {
                "input": match.input,
                "matched_flag": match.matched_flag,
                "accuracy": match.accuracy,
                "source": match.source,
                "history_persisted": configuration.interaction.history_enabled,
            }
            if configuration.interaction.model_debug:
                payload["model_debug"] = {"model_name": match.model_name, "threshold": configuration.interaction.similarity_threshold}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if match.matched_flag is not None else 1
        if arguments.command == "usage":
            statistics = usage_statistics(load_history(history_path))
            graph_path = write_usage_graph(configuration.runtime.output_root / "flag_usage.svg", statistics)
            print(json.dumps({**statistics, "graph_path": str(graph_path)}, indent=2, sort_keys=True))
            return 0
        report = preflight(root, configuration_hash=configuration.fingerprint())
        if arguments.command == "preflight":
            validation = report.to_dict(
                selected_plugins=tuple(arguments.include),
                enable_plugins=arguments.plugins,
                enable_mods=arguments.mods,
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
                "maintenance": maintenance_diagnostics(root, configuration.maintenance),
            }
            debug_path = configuration.runtime.output_root.parent / "logs" / "debug" / "debug.json"
            payload["debug_log"] = str(debug_path)
            _write_json(debug_path, payload)
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
        if arguments.command == "dev":
            return _run_developer_action(root, configuration, arguments)
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
        exit_code = 0 if outcome.manifest.status.value == "succeeded" else 1
        if arguments.interactive:
            try:
                input("Press Enter to exit...")
            except EOFError:
                pass
        return exit_code
    except (LogicyticsError, OSError, PermissionError, ValueError) as error:
        print(f"Error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
