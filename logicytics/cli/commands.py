"""Command-line orchestration for v4 planning and supervised execution."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Literal

from logicytics.contracts import Capability, OutputPolicy, PostRunAction, RunRequest
from logicytics.module.configuration import AppConfig, load_config
from logicytics.module.discovery import preflight
from logicytics.module.environment import inspect_environment
from logicytics.module.errors import LogicyticsError
from logicytics.module.interaction import (
    load_history,
    match_flag,
    record_command,
    record_match,
    usage_statistics,
    write_usage_graph,
)
from logicytics.module.logging import (
    ApplicationLogger,
    HumanArgumentParser,
    collector_log_source,
    get_application_logger,
)
from logicytics.module.maintenance import (
    build_manifest,
    compare_files,
    developer_checks,
    load_local_manifest,
    local_version,
    maintenance_diagnostics,
    write_local_manifest,
)
from logicytics.module.manifest import MANIFEST_SCHEMA_VERSION
from logicytics.module.modes import (
    EXECUTION_MODES,
    LEGACY_MODE_ALIASES,
    ExecutionStrategy,
    mode_matrix,
    resolve_execution_mode,
)
from logicytics.module.output_layout import ensure_output_layout
from logicytics.module.planner import build_plan
from logicytics.module.runtime import RunSupervisor
from logicytics.module.sysinternals import ensure_sysinternals
from logicytics.platform_adapters import process_adapter
from logicytics.terminal import terminal_lifecycle
from logicytics.virtual_environment import (
    is_running_in_virtual_environment,
    virtual_environment_details,
    virtual_environment_error,
)


class CLI:
    """Translate command-line arguments into validated application requests."""

    @staticmethod
    def render_preflight(
        logger: ApplicationLogger,
        validation: dict[str, list[dict[str, object]]],
        sysinternals: dict[str, str],
    ) -> None:
        """Present validated collectors and diagnostics without exposing internal JSON."""
        valid = validation["valid"]
        invalid = validation["invalid"]
        quarantined = validation["quarantined"]
        logger.box(
            "Preflight summary",
            (
                f"Valid collectors: {len(valid)}",
                f"Quarantined extensions: {len(quarantined)}",
                f"Blocking failures: {len(invalid)}",
                f"Sysinternals: {sysinternals['status']}",
            ),
        )
        for item in (*invalid, *quarantined):
            raw_diagnostics = item.get("diagnostics", [])
            diagnostics = raw_diagnostics if isinstance(raw_diagnostics, list) else []
            details = "; ".join(
                str(diagnostic.get("message", "invalid collector")) for diagnostic in diagnostics if isinstance(diagnostic, dict)
            )
            logger.event(
                "ERROR" if item in invalid else "WARNING",
                details or "collector validation failed",
                source="logicytics.cli.commands",
                collector=str(item["id"]),
            )

    @staticmethod
    def project_root() -> Path:
        """Return the repository root from the relocated CLI package."""
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def request(
        arguments: argparse.Namespace,
        default_workers: int,
        configured_blocked_capabilities: tuple[Capability, ...] = (),
    ) -> RunRequest:
        """Build an immutable run request while enforcing mode and rerun conflicts."""
        legacy_flags = {flag: getattr(arguments, flag, False) for flag in LEGACY_MODE_ALIASES}

        selected_mode = resolve_execution_mode(getattr(arguments, "mode", None), legacy_flags)
        profile_mode = resolve_execution_mode(getattr(arguments, "profile", None), {})

        if selected_mode is not None and profile_mode is not None:
            raise ValueError("--profile cannot be combined with a named collection mode")

        mode = selected_mode or profile_mode
        profile = mode.profile if mode is not None else "standard"

        explicit_sequential = getattr(arguments, "sequential", False)
        explicit_parallel = getattr(arguments, "parallel", False)

        if mode is not None:
            if explicit_sequential and mode.strategy is ExecutionStrategy.PARALLEL:
                if legacy_flags["threaded"]:
                    raise ValueError("sequential execution conflicts with legacy --threaded mode")
                raise ValueError(f"{mode.name} mode requires configured parallel execution")

            if explicit_parallel and mode.strategy is ExecutionStrategy.SEQUENTIAL:
                if legacy_flags["performance_check"] or legacy_flags["default_mode"]:
                    raise ValueError("parallel execution conflicts with sequential performance/default mode")
                raise ValueError(f"{mode.name} mode requires sequential execution")

        sequential = explicit_sequential or (mode is not None and mode.strategy is ExecutionStrategy.SEQUENTIAL)
        parallel = explicit_parallel or (mode is not None and mode.strategy is ExecutionStrategy.PARALLEL)

        performance_check = mode.performance_check if mode is not None else False

        enable_mods = getattr(arguments, "mods", False) or (mode.enable_mods if mode is not None else False)

        requested_workers = getattr(arguments, "workers", None)

        if sequential and requested_workers is not None and requested_workers != 1:
            raise ValueError("sequential execution requires --workers=1")

        worker_count = 1 if sequential else requested_workers or default_workers

        if parallel and worker_count < 2:
            raise ValueError("parallel execution requires at least two configured workers")

        parent_run_id: str | None = None

        rerun_value = getattr(arguments, "rerun_from", None)

        if rerun_value is not None:
            if not isinstance(rerun_value, (str, os.PathLike)):
                raise TypeError(f"rerun_from must be a path-like value, got {type(rerun_value).__name__}")

            rerun_path = Path(rerun_value)

            if not arguments.include:
                raise ValueError("--rerun-from requires at least one explicit --include collector ID")

            manifest_path = rerun_path / "manifest.json" if rerun_path.is_dir() else rerun_path

            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"original run manifest cannot be loaded: {error}") from error

            if not isinstance(previous, dict) or not isinstance(
                previous.get("run_id"),
                str,
            ):
                raise ValueError("original run manifest must contain a valid run_id")

            manifest_schema_version = previous.get("manifest_schema_version")

            if (
                not isinstance(manifest_schema_version, int)
                or isinstance(manifest_schema_version, bool)
                or manifest_schema_version != MANIFEST_SCHEMA_VERSION
            ):
                schema_version_display = "None" if manifest_schema_version is None else repr(manifest_schema_version)

                raise ValueError(
                    f"original run manifest uses unsupported schema_version {schema_version_display}; expected {MANIFEST_SCHEMA_VERSION}"
                )

            if previous.get("status") not in {
                "succeeded",
                "partial",
                "failed",
                "cancelled",
            }:
                raise ValueError("original run manifest must describe a finalized run")

            resolved = previous.get("resolved_plan")

            if not isinstance(resolved, list) or not all(isinstance(item, str) for item in resolved):
                raise ValueError("original run manifest must contain a valid resolved_plan")

            unknown = sorted(set(arguments.include) - set(resolved))

            if unknown:
                raise ValueError(f"rerun collectors were not present in the original run: {', '.join(unknown)}")

            parent_run_id = previous["run_id"]

        selection_only = getattr(arguments, "command", None) == "collector"
        include_arguments = tuple(getattr(arguments, "include", ()) or ())
        exclude_arguments = tuple(getattr(arguments, "exclude", ()) or ())
        plugins_enabled = bool(getattr(arguments, "plugins", False))
        mods_enabled = bool(getattr(arguments, "mods", False))

        if selection_only and (include_arguments or exclude_arguments or arguments.profile is not None or plugins_enabled or mods_enabled):
            raise ValueError("collector execution cannot combine its ID with profile or selection flags")

        includes = (arguments.collector_id,) if selection_only else include_arguments

        requested_blocked_capabilities = tuple(Capability(value) for value in getattr(arguments, "block_capability", ()))
        blocked_capabilities = tuple(dict.fromkeys((*configured_blocked_capabilities, *requested_blocked_capabilities)))

        return RunRequest(
            profile=profile,
            include=includes,
            exclude=exclude_arguments,
            selection_only=selection_only,
            enable_plugins=plugins_enabled,
            enable_mods=enable_mods,
            max_workers=worker_count,
            acknowledge_authorization=getattr(
                arguments,
                "acknowledge_authorization",
                False,
            ),
            blocked_capabilities=blocked_capabilities,
            performance_check=performance_check,
            rerun_from=parent_run_id,
            output_policy=(OutputPolicy.MANIFEST_ONLY if getattr(arguments, "no_package", False) else OutputPolicy.PACKAGE),
            post_run_action=(
                PostRunAction.REBOOT
                if getattr(arguments, "reboot", False)
                else PostRunAction.SHUTDOWN
                if getattr(arguments, "shutdown", False)
                else PostRunAction.NONE
            ),
        )

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """Create the complete parser for maintenance, planning, and collection commands."""
        parser = HumanArgumentParser(description="Logicytics v4 run-oriented evidence framework")
        parser.add_argument(
            "--config",
            type=Path,
            help="Path to the authoritative Logicytics YAML configuration file",
        )
        parser.add_argument(
            "--usage",
            action="store_true",
            help="Show local interaction statistics and create a usage graph.",
        )
        parser.add_argument("--modes", action="store_true", help="Show the typed execution-mode inclusion matrix.")
        parser.add_argument(
            "--match",
            metavar="TEXT",
            help="Suggest the closest documented action for natural-language input.",
        )
        subcommands = parser.add_subparsers(dest="command", parser_class=HumanArgumentParser)
        for command in ("preflight", "debug", "update", "dev", "plan", "run", "collector"):
            subparser = subcommands.add_parser(command, help=f"Run the {command} action.")
            subparser.add_argument(
                "--config",
                type=Path,
                default=argparse.SUPPRESS,
                help="Path to the authoritative Logicytics YAML configuration file",
            )
            subparser.add_argument(
                "--profile",
                default=None,
                choices=tuple(EXECUTION_MODES),
                help="Named user-facing collection mode.",
            )
            if command in {"preflight", "plan"}:
                subparser.add_argument(
                    "--invalidate-cache",
                    action="store_true",
                    help="Discard cached preflight metadata before validation.",
                )
            subparser.add_argument(
                "--include",
                action="append",
                default=[],
                metavar="COLLECTOR_ID",
                help="Include a collector by exact dotted ID; repeat for multiple collectors.",
            )
            subparser.add_argument(
                "--exclude",
                action="append",
                default=[],
                metavar="COLLECTOR_ID",
                help="Exclude a collector by exact dotted ID; repeat for multiple collectors.",
            )
            subparser.add_argument(
                "--plugins",
                action="store_true",
                help="Enable all valid opt-in plugin collectors for the selected profile.",
            )
            subparser.add_argument(
                "--mods",
                action="store_true",
                help="Enable valid sidecar-declared scripts from the MODS directory.",
            )
            subparser.add_argument(
                "--workers",
                type=int,
                metavar="COUNT",
                help="Bound concurrent isolated workers to this positive count.",
            )
            subparser.add_argument(
                "--block-capability",
                action="append",
                default=[],
                choices=[capability.value for capability in Capability],
                help="Block a declared capability for the selected collectors; repeat as needed.",
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
                mode.add_argument(
                    "--mode",
                    choices=tuple(EXECUTION_MODES),
                    help="Select one user-facing typed execution mode.",
                )
                mode.add_argument(
                    "--default",
                    dest="default_mode",
                    action="store_true",
                    help="Run the standard built-in profile.",
                )
                mode.add_argument(
                    "--threaded",
                    action="store_true",
                    help="Run the standard built-in profile with configured parallel workers.",
                )
                mode.add_argument("--minimal", action="store_true", help="Run the minimal built-in profile.")
                mode.add_argument("--depth", action="store_true", help="Run the deep built-in profile.")
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
            if command == "collector":
                subparser.add_argument("collector_id", help="Exact dotted ID of the collector to run independently.")
                subparser.add_argument(
                    "--acknowledge-authorization",
                    action="store_true",
                    help="Confirm you are authorized to collect this collector's evidence.",
                )
                subparser.add_argument(
                    "--no-package",
                    action="store_true",
                    help="Finalize the direct collector manifest without creating a ZIP package.",
                )
                subparser.add_argument(
                    "--interactive",
                    action="store_true",
                    help="Pause at the final status so an interactive command window remains visible.",
                )
            if command == "update":
                subparser.add_argument(
                    "--apply",
                    action="store_true",
                    help="Explicitly run git pull after repository checks.",
                )
                subparser.add_argument(
                    "--launch-action",
                    choices=("preflight", "debug", "dev"),
                    help="Select a safe maintenance action to launch after a successful update check.",
                )
                subparser.add_argument(
                    "--new-window",
                    action="store_true",
                    help="Launch --launch-action in a visible, separate Windows command window.",
                )
                subparser.add_argument(
                    "--performance-check",
                    action="store_true",
                    help="Run collectors sequentially and write a per-collector duration report.",
                )
            if command == "dev":
                subparser.add_argument(
                    "--write-manifest",
                    action="store_true",
                    help=("Write the reviewed local integrity manifest; requires --next-version unless interactive."),
                )
                subparser.add_argument(
                    "--next-version",
                    help=("Semantic version to validate and record in a newly written integrity manifest."),
                )
                subparser.add_argument(
                    "--interactive",
                    action="store_true",
                    help=("Show contribution checks and prompt before changing the local integrity manifest."),
                )
        return parser

    @staticmethod
    def write_json(path: Path, payload: object) -> Path:
        """Atomically write a human-readable JSON diagnostic artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    @staticmethod
    def launch_action_window(root: Path, action: str) -> int:
        """Launch one allowlisted maintenance action in a separate visible Windows console."""
        if sys.platform != "win32":
            raise OSError("new command windows are supported only on Windows")
        if action not in {"preflight", "debug", "dev"}:
            raise ValueError(f"unsupported new-window action: {action}")
        command = [sys.executable, "-m", "logicytics", action]
        process = process_adapter.popen(
            command,
            cwd=root,
            shell=False,
            creationflags=process_adapter.create_new_console,
            close_fds=True,
        )
        return process.pid

    @staticmethod
    def repository_status(root: Path) -> dict[str, bool | int | str | None]:
        """Check the local Git worktree, origin declaration, and remote reachability."""
        status: dict[str, bool | int | str | None] = {
            "git_available": False,
            "git_version": None,
            "is_repository": False,
            "origin_configured": False,
            "remote_reachable": False,
            "repository_returncode": None,
            "origin_returncode": None,
            "reachability_returncode": None,
        }
        try:
            git = process_adapter.run(
                ["git", "--version"],
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError:
            return status

        git_output = git.stdout if isinstance(git.stdout, str) else ""
        status["git_available"] = git.returncode == 0
        status["git_version"] = git_output.strip() or None
        if git.returncode != 0:
            return status

        try:
            repository = process_adapter.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=root,
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError:
            return status
        status["repository_returncode"] = repository.returncode
        repository_output = repository.stdout if isinstance(repository.stdout, str) else ""
        status["is_repository"] = repository.returncode == 0 and repository_output.strip().casefold() == "true"
        if not status["is_repository"]:
            return status

        try:
            origin = process_adapter.run(
                ["git", "remote", "get-url", "origin"],
                cwd=root,
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError:
            return status
        status["origin_returncode"] = origin.returncode
        origin_output = origin.stdout if isinstance(origin.stdout, str) else ""
        status["origin_configured"] = origin.returncode == 0 and bool(origin_output.strip())
        if not status["origin_configured"]:
            return status

        try:
            reachability = process_adapter.run(
                ["git", "ls-remote", "--exit-code", "origin", "HEAD"],
                cwd=root,
                capture_output=True,
                check=False,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            return status
        status["reachability_returncode"] = reachability.returncode
        status["remote_reachable"] = reachability.returncode == 0
        return status

    def run_developer_action(
        self,
        root: Path,
        configuration: AppConfig,
        arguments: argparse.Namespace,
        debug_logs: Path,
        logger: ApplicationLogger,
        repository: dict[str, bool | int | str | None],
    ) -> int:
        """Run read-only contribution checks and an explicitly confirmed manifest update."""
        settings = configuration.maintenance
        logger.event(
            "INFO",
            "development_checks_started",
            source="logicytics.cli.commands",
            manifest_write_requested=arguments.write_manifest,
            interactive=arguments.interactive,
        )
        existing = load_local_manifest(root, settings)

        comparison = (
            compare_files(root, settings, existing)
            if existing
            else {
                "missing": [],
                "modified": [],
                "extra": [],
                "unchanged": [],
            }
        )

        checks = developer_checks(root, settings)
        next_version = arguments.next_version
        write_requested = arguments.write_manifest
        logger.event(
            "INFO",
            "development_integrity_reviewed",
            source="logicytics.cli.commands",
            manifest_available=existing is not None,
            missing_files=len(comparison["missing"]),
            modified_files=len(comparison["modified"]),
            extra_files=len(comparison["extra"]),
        )

        if arguments.interactive:
            organization_checks: tuple[
                Literal[
                    "naming_violations",
                    "misplaced_python",
                    "missing_module_docstrings",
                    "crowded_modules",
                ],
                ...,
            ] = (
                "naming_violations",
                "misplaced_python",
                "missing_module_docstrings",
                "crowded_modules",
            )
            logger.box(
                "Contribution and repository organization checks",
                (
                    *(
                        f"{name.title()}: {len(comparison[name])}"
                        for name in (
                            "missing",
                            "modified",
                            "extra",
                            "unchanged",
                        )
                    ),
                    *(f"{name.replace('_', ' ').title()}: {len(checks[name])}" for name in organization_checks),
                ),
            )

            if next_version is None:
                current_version = local_version(root)
                entered = input(f"Next semantic version [{current_version}]: ").strip()
                next_version = entered or current_version

            answer = input("Write the reviewed integrity manifest? [y/N]: ").strip().casefold()

            write_requested = answer in {"y", "yes"}

        if arguments.write_manifest and next_version is None:
            raise ValueError("--write-manifest requires --next-version unless --interactive is used")

        manifest_path: str | None = None

        if write_requested:
            if next_version is None:
                raise ValueError("a semantic next version is required to write the integrity manifest")

            logger.event(
                "INFO",
                "development_manifest_write_started",
                source="logicytics.cli.commands",
                next_version=next_version,
            )
            manifest = build_manifest(root, settings, next_version)
            written_manifest = write_local_manifest(
                root,
                settings,
                manifest,
            )

            manifest_path = str(written_manifest)
            logger.event(
                "INFO",
                "development_manifest_written",
                source="logicytics.cli.commands",
                manifest_path=manifest_path,
            )

        payload = {
            "checks": checks,
            "comparison": comparison,
            "repository": repository,
            "current_version": local_version(root),
            "existing_manifest_version": (existing.version if existing else None),
            "manifest_written": manifest_path,
            "next_version": next_version,
        }

        development_path = debug_logs / "development.json"
        self.write_json(development_path, payload)
        organization_issues = sum(
            len(checks[name])
            for name in (
                "naming_violations",
                "misplaced_python",
                "missing_module_docstrings",
                "crowded_modules",
            )
        )
        logger.event(
            "INFO" if organization_issues == 0 else "WARNING",
            "development_structure_reviewed",
            source="logicytics.cli.commands",
            organization_issues=organization_issues,
            github_reachable=bool(repository["remote_reachable"]),
        )
        logger.box(
            "Development summary",
            (
                f"Integrity: {len(comparison['missing'])} missing, "
                f"{len(comparison['modified'])} modified, {len(comparison['extra'])} extra",
                f"Organization issues: {organization_issues}",
                f"GitHub: {'reachable' if repository['remote_reachable'] else 'unreachable'}",
                f"Manifest: {'written' if manifest_path else 'not changed'}",
                *((f"Manifest path: {manifest_path}",) if manifest_path else ()),
                f"Diagnostic report: {development_path}",
            ),
        )
        return 0 if organization_issues == 0 and repository["remote_reachable"] else 1


def main(argv: list[str] | None = None) -> int:
    """Run the selected preflight, planning, or supervised execution command."""
    if not is_running_in_virtual_environment():
        ApplicationLogger.render_alert(
            sys.stderr,
            "Logicytics startup error",
            virtual_environment_error(CLI.project_root()),
        )
        return 2
    cli_parser = cli_methods.parser()
    try:
        arguments = cli_parser.parse_args(argv)
    except KeyboardInterrupt:
        ApplicationLogger.render_section(sys.stderr, "Command cancelled", ("Interrupted by user.",))
        return 130

    standalone_actions = sum(
        bool(value)
        for value in (
            arguments.usage,
            arguments.match,
            arguments.modes,
        )
    )

    if standalone_actions > 1:
        cli_parser.error("--usage, --match, and --modes are mutually exclusive")

    if arguments.command is not None and standalone_actions:
        cli_parser.error("--usage, --match, and --modes are standalone actions")

    if arguments.usage:
        arguments.command = "usage"
    elif arguments.match:
        arguments.command = "match"
    elif arguments.modes:
        arguments.command = "modes"

    if arguments.command is None:
        if arguments.config is None:
            cli_parser.print_help()
            return 0
        arguments.command = "config"

    root = cli_methods.project_root()
    application_logger = None
    command_started_at: float | None = None

    try:
        configuration = load_config(root, arguments.config)

        layout = ensure_output_layout(configuration.runtime.output_root)

        application_logger = get_application_logger(
            layout.application_log,
            configuration.logging,
        )

        application_logger.event(
            "INFO",
            "command_started",
            source="logicytics.cli.commands",
            command=arguments.command,
        )
        started_at = perf_counter()
        command_started_at = started_at

        def finish_command(
            exit_code: int,
            *,
            status: str | None = None,
            **fields: float | str,
        ) -> int:
            """Record one command's terminal status and elapsed time."""
            application_logger.event(
                "INFO" if exit_code == 0 else "WARNING",
                "command_finished",
                source="logicytics.cli.commands",
                command=str(arguments.command),
                exit_code=exit_code,
                status=status or ("succeeded" if exit_code == 0 else "failed"),
                duration_seconds=round(perf_counter() - started_at, 3),
                **fields,
            )
            return exit_code

        history_path = configuration.runtime.output_root / "interaction_history.json.gz"

        if configuration.interaction.history_enabled and arguments.command not in {"match", "usage"}:
            record_command(
                history_path,
                str(arguments.command),
                mode=getattr(arguments, "mode", None) or getattr(arguments, "profile", None),
            )

        if arguments.command == "config":
            selected_path = arguments.config.resolve()
            application_logger.box(
                "Configuration",
                (
                    "Configuration validated successfully.",
                    f"Path: {selected_path}",
                    f"Schema version: {configuration.schema_version}",
                ),
            )
            return finish_command(0, status="configuration_validated", configuration_path=str(selected_path))

        last_preflight_progress_at = 0.0

        def report_preflight_progress(phase: str, checked: int, total: int, current: str) -> None:
            """Keep long sequential collector validation visibly alive without flooding output."""
            nonlocal last_preflight_progress_at
            now = perf_counter()
            should_report = (
                    last_preflight_progress_at == 0.0 or
                    now - last_preflight_progress_at >= 1.0 or
                    (phase == "checked" and checked == total)
            )
            if should_report:
                application_logger.event(
                    "INFO",
                    "preflight_progress",
                    console=False,
                    source="logicytics.cli.commands",
                    phase="validating collectors",
                    checked=checked,
                    total=total,
                    current=current,
                )
                application_logger.progress(
                    "Preflight", checked, total, current
                )
                last_preflight_progress_at = now

        if arguments.command == "match":
            history = load_history(history_path)

            match = match_flag(
                arguments.match,
                threshold=(configuration.interaction.similarity_threshold),
                model_name=configuration.interaction.model_name,
                history=history,
            )

            if configuration.interaction.history_enabled:
                record_match(history_path, match)

            application_logger.box(
                "Match result",
                (
                    f"Input: {match.input}",
                    f"Matched flag: {match.matched_flag or 'none'}",
                    f"Confidence: {match.accuracy:.1%}",
                    f"Match source: {match.source.replace('_', ' ')}",
                    f"History persisted: {'yes' if configuration.interaction.history_enabled else 'no'}",
                    *(
                        (f"Model: {match.model_name} (threshold {configuration.interaction.similarity_threshold:.1%})",)
                        if configuration.interaction.model_debug
                        else ()
                    ),
                ),
            )

            exit_code = 0 if match.matched_flag is not None else 1
            return finish_command(
                exit_code,
                status="matched" if exit_code == 0 else "not_matched",
                matched_flag=match.matched_flag or "none",
            )

        if arguments.command == "usage":
            statistics = usage_statistics(load_history(history_path))
            raw_average_accuracy = statistics.get("average_accuracy", 0.0)
            average_accuracy = float(raw_average_accuracy) if isinstance(raw_average_accuracy, (int, float)) else 0.0

            graph_path = write_usage_graph(
                configuration.runtime.output_root / "flag_usage.svg",
                statistics,
            )

            frequencies = statistics.get("per_flag_frequency", {})
            frequency_rows = (
                tuple(f"{flag}: {count}" for flag, count in sorted(frequencies.items()))
                if isinstance(frequencies, dict) and frequencies
                else ("none",)
            )
            application_logger.box(
                "Interaction usage",
                (
                    f"Total interactions: {statistics['total_interactions']}",
                    f"Average confidence: {average_accuracy:.1%}",
                    f"Common device: {statistics['common_device'] or 'none'}",
                    f"Common input: {statistics['common_input'] or 'none'}",
                    "Flag frequency:",
                    *(f"  {row}" for row in frequency_rows),
                    f"Usage graph: {graph_path}",
                ),
            )

            return finish_command(0, status="usage_written")

        if arguments.command == "modes":
            modes_configuration_hash = configuration.fingerprint()
            application_logger.event(
                "INFO",
                "preflight_started",
                source="logicytics.cli.commands",
                configuration_hash=modes_configuration_hash,
                purpose="mode_matrix",
            )
            report = preflight(
                root,
                configuration_hash=modes_configuration_hash,
                progress=report_preflight_progress,
            )
            application_logger.event(
                "INFO",
                "preflight_finished",
                source="logicytics.cli.commands",
                valid_collectors=len(report.valid),
                invalid_collectors=len(report.invalid),
                purpose="mode_matrix",
            )

            matrix_payload = mode_matrix((*report.valid, *report.invalid))
            matrix_path = layout.debug_logs / "modes.json"
            cli_methods.write_json(matrix_path, matrix_payload)
            mode_rows = []
            for item in matrix_payload["modes"]:
                aliases = ", ".join(item["legacy_aliases"]) or "none"
                mode_rows.append(f"{item['name']}: {item['description']} ({len(item['collector_ids'])} collectors; aliases: {aliases})")
            application_logger.box(
                "Execution modes",
                (
                    *mode_rows,
                    f"Valid collectors: {len(report.valid)}",
                    f"Invalid collectors: {len(report.invalid)}",
                    f"Machine-readable matrix: {matrix_path}",
                ),
            )

            return finish_command(
                0,
                status="mode_matrix_written",
                valid_collectors=len(report.valid),
                invalid_collectors=len(report.invalid),
            )

        application_logger.event(
            "INFO",
            "preflight_started",
            source="logicytics.cli.commands",
            configuration_hash=configuration.fingerprint(),
        )

        report = preflight(
            root,
            configuration_hash=configuration.fingerprint(),
            progress=report_preflight_progress,
            invalidate_cache=getattr(arguments, "invalidate_cache", False),
        )
        application_logger.event(
            "INFO",
            "preflight_finished",
            source="logicytics.cli.commands",
            valid_collectors=len(report.valid),
            invalid_collectors=len(report.invalid),
        )

        if arguments.command == "preflight":
            validation = report.to_dict(
                selected_plugins=tuple(arguments.include),
                enable_plugins=arguments.plugins,
                enable_mods=arguments.mods,
            )

            sysinternals = ensure_sysinternals(root, configuration.maintenance).to_dict()
            cli_methods.render_preflight(application_logger, validation, sysinternals)

            exit_code = 0 if not validation["invalid"] else 2
            return finish_command(
                exit_code,
                status="validated" if exit_code == 0 else "invalid_collectors",
                valid_collectors=len(validation["valid"]),
                invalid_collectors=len(validation["invalid"]),
            )

        if arguments.command == "debug":
            payload: dict[str, object] = {
                "configuration": (configuration.to_manifest_dict()),
                "environment": inspect_environment().to_dict(),
                "python": {
                    "executable": sys.executable,
                    "implementation": (platform.python_implementation()),
                    "version": platform.python_version(),
                    "prefix": sys.prefix,
                    "virtual_environment": virtual_environment_details(),
                    "psutil_available": (importlib.util.find_spec("psutil") is not None),
                    "cpu_count": os.cpu_count(),
                },
                "sysinternals": (ensure_sysinternals(root, configuration.maintenance).to_dict()),
                "preflight": {
                    "valid_collectors": len(report.valid),
                    "invalid_collectors": len(report.invalid),
                },
                "maintenance": maintenance_diagnostics(
                    root,
                    configuration.maintenance,
                ),
            }

            debug_path = layout.debug_logs / "debug.json"

            payload["debug_log"] = str(debug_path)

            cli_methods.write_json(
                debug_path,
                payload,
            )

            application_logger.box(
                "Diagnostics",
                (
                    f"Valid collectors: {len(report.valid)}",
                    f"Invalid collectors: {len(report.invalid)}",
                    f"Virtual environment: {'yes' if is_running_in_virtual_environment() else 'no'}",
                    f"Diagnostic report: {debug_path}",
                ),
            )

            exit_code = 0 if not report.invalid else 2
            return finish_command(
                exit_code,
                status="diagnostics_written" if exit_code == 0 else "invalid_collectors",
                valid_collectors=len(report.valid),
                invalid_collectors=len(report.invalid),
            )

        if arguments.command == "update":
            if arguments.new_window != (arguments.launch_action is not None):
                raise ValueError("--new-window and --launch-action must be provided together")

            application_logger.event(
                "INFO",
                "update_started",
                source="logicytics.cli.commands",
                action="apply update" if arguments.apply else "check for updates",
            )
            repository = cli_methods.repository_status(root)
            payload: dict[str, object] = {
                **repository,
                "applied": False,
            }
            application_logger.event(
                "INFO" if repository["remote_reachable"] else "WARNING",
                "update_repository_checked",
                source="logicytics.cli.commands",
                git_available=bool(repository["git_available"]),
                repository_ready=bool(repository["is_repository"]),
                origin_configured=bool(repository["origin_configured"]),
                github_reachable=bool(repository["remote_reachable"]),
            )

            pull_returncode: int | None = None

            if not repository["remote_reachable"]:
                update_path = layout.debug_logs / "update.json"
                cli_methods.write_json(update_path, payload)
                if not repository["git_available"]:
                    next_step = "Install Git, then run the update command again."
                elif not repository["is_repository"]:
                    next_step = "Run the update command from a Git repository."
                elif not repository["origin_configured"]:
                    next_step = "Configure the repository's origin remote, then retry."
                else:
                    next_step = "Restore GitHub connectivity, then retry the update command."
                application_logger.event(
                    "WARNING",
                    "update_not_applied",
                    source="logicytics.cli.commands",
                    reason="repository remote is unavailable",
                    diagnostic_report=str(update_path),
                )
                application_logger.box(
                    "Update summary",
                    (
                        "Result: update was not applied.",
                        f"Next step: {next_step}",
                        f"Diagnostic report: {update_path}",
                    ),
                )
                return finish_command(
                    2,
                    status="repository_unreachable",
                    git_available=str(repository["git_available"]),
                    remote_reachable=str(repository["remote_reachable"]),
                )

            if arguments.apply:
                application_logger.event(
                    "INFO",
                    "update_apply_started",
                    source="logicytics.cli.commands",
                    command="git pull",
                )
                pulled = process_adapter.run(
                    ["git", "pull"],
                    cwd=root,
                    capture_output=True,
                    check=False,
                    text=True,
                )

                pull_returncode = pulled.returncode

                payload.update(
                    {
                        "applied": True,
                        "returncode": pulled.returncode,
                        "stdout": pulled.stdout,
                        "stderr": pulled.stderr,
                    }
                )
                application_logger.event(
                    "INFO" if pulled.returncode == 0 else "ERROR",
                    "update_apply_finished",
                    source="logicytics.cli.commands",
                    exit_code=pulled.returncode,
                )

            update_succeeded = not arguments.apply or pull_returncode == 0

            if arguments.new_window and update_succeeded:
                launched_process_id = cli_methods.launch_action_window(
                    root,
                    arguments.launch_action,
                )

                payload["launched_action"] = arguments.launch_action
                payload["launched_process_id"] = launched_process_id

            update_path = layout.debug_logs / "update.json"
            cli_methods.write_json(update_path, payload)
            update_lines = [
                f"Repository: {'ready' if repository['is_repository'] else 'not detected'}",
                "GitHub: reachable",
                f"Action: {'git pull completed' if arguments.apply else 'connectivity check only'}",
                f"Result: {'succeeded' if update_succeeded else 'failed'}",
            ]
            if arguments.apply:
                pull_exit_code = pull_returncode if pull_returncode is not None else "none"
                update_lines.append(f"Git pull exit code: {pull_exit_code}")
                if not update_succeeded:
                    update_lines.append("Next step: inspect the diagnostic report, resolve Git's error, then retry.")
            if payload.get("launched_action"):
                update_lines.extend(
                    (
                        f"Launched action: {payload['launched_action']}",
                        f"Launched process id: {payload['launched_process_id']}",
                    )
                )
            update_lines.append(f"Diagnostic report: {update_path}")
            application_logger.event(
                "INFO" if update_succeeded else "ERROR",
                "update_finished",
                source="logicytics.cli.commands",
                action="applied" if arguments.apply else "checked",
                status="succeeded" if update_succeeded else "failed",
            )
            application_logger.box("Update summary", tuple(update_lines))

            return finish_command(
                0 if update_succeeded else 1,
                status="updated" if update_succeeded else "update_failed",
                applied=arguments.apply,
            )

        if arguments.command == "dev":
            repository = cli_methods.repository_status(root)
            exit_code = cli_methods.run_developer_action(
                root,
                configuration,
                arguments,
                layout.debug_logs,
                application_logger,
                repository,
            )
            return finish_command(
                exit_code,
                status="checks_completed" if exit_code == 0 else "checks_failed",
            )

        request = cli_methods.request(
            arguments,
            configuration.runtime.default_max_workers,
            configuration.runtime.blocked_capabilities,
        )
        application_logger.event(
            "INFO",
            "plan_requested",
            source="logicytics.cli.commands",
            profile=request.profile,
            include_count=len(request.include),
            exclude_count=len(request.exclude),
            blocked_capabilities=len(request.blocked_capabilities),
            enable_plugins=request.enable_plugins,
            enable_mods=request.enable_mods,
            max_workers=request.max_workers,
        )
        plan = build_plan(
            report,
            request,
        )
        application_logger.event(
            "INFO",
            "plan_created",
            source="logicytics.cli.commands",
            collectors=len(plan.collectors),
            fingerprint=plan.fingerprint,
        )

        if arguments.command == "plan":
            application_logger.box(
                "Collection plan",
                tuple(candidate.metadata.id for candidate in plan.collectors if candidate.metadata is not None)
                or ("No collectors were selected.",),
            )
            application_logger.event(
                "INFO",
                "plan_rendered",
                source="logicytics.cli.commands",
                collectors=len(plan.collectors),
            )

            return finish_command(0, status="plan_rendered", collectors=len(plan.collectors))

        outcome = RunSupervisor(
            root,
            configuration,
        ).run(plan)

        status_counts: dict[str, int] = {}
        for record in outcome.manifest.collectors:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1

        result_level = {
            "succeeded": "INFO",
            "partial": "WARNING",
            "cancelled": "WARNING",
            "failed": "ERROR",
        }.get(outcome.manifest.status.value, "CRITICAL")
        application_logger.event(
            result_level,
            "collection_result",
            source="logicytics.cli.commands",
            status=outcome.manifest.status.value,
            collectors=len(outcome.manifest.collectors),
            succeeded_count=status_counts.get("succeeded", 0),
            skipped_count=status_counts.get("skipped", 0),
            failed_count=status_counts.get("failed", 0),
            cancelled_count=status_counts.get("cancelled", 0),
        )

        for record in outcome.manifest.collectors:
            if record.status == "succeeded":
                continue
            severity = "ERROR" if record.status == "failed" else "WARNING"
            if record.failure is not None:
                application_logger.event(
                    severity,
                    "collector_result",
                    source=collector_log_source(record.id) or "logicytics.cli.commands",
                    collector_id=record.id,
                    status=record.status,
                    summary=record.summary or "not-finished",
                    duration_seconds=record.duration_seconds or 0.0,
                    operation=str(record.failure.get("operation", "unknown")),
                    platform_error=str(record.failure.get("platform_error", "unknown")),
                    remediation=str(record.failure.get("remediation", "none")),
                )
                continue
            application_logger.event(
                severity,
                "collector_result",
                source=collector_log_source(record.id) or "logicytics.cli.commands",
                collector_id=record.id,
                status=record.status,
                summary=record.summary or "not-finished",
                duration_seconds=record.duration_seconds or 0.0,
            )

        if outcome.manifest.status.value == "succeeded":
            result_lines = [f"Collectors: {len(outcome.manifest.collectors)}"]
            if outcome.manifest.package and "path" in outcome.manifest.package:
                package_sha_path = outcome.manifest.package.get("sha256_path", "unavailable")
                result_lines.extend(
                    f"Package: {outcome.manifest.package['path']}\nSHA-256: {package_sha_path}".splitlines()
                )
            result_lines.append(f"Run: {outcome.manifest_path}")
            application_logger.box("Collection result", result_lines)

        exit_code = 0 if outcome.manifest.status.value == "succeeded" else 1

        if arguments.interactive:
            try:
                input("Press Enter to exit...")
            except EOFError:
                pass

        return finish_command(
            exit_code,
            status=outcome.manifest.status.value,
            run_id=outcome.manifest.run_id,
            collectors=len(outcome.manifest.collectors),
        )

    except KeyboardInterrupt:
        if application_logger is not None:
            with contextlib.suppress(OSError):
                application_logger.incomplete_progress("Preflight")

            with contextlib.suppress(OSError):
                application_logger.event(
                    "WARNING",
                    "command_cancelled",
                    source="logicytics.cli.commands",
                    command=str(arguments.command),
                    exit_code=130,
                    console=False,
                )

            application_logger.box(
                "Command cancelled",
                ("Interrupted by user.",),
            )

        else:
            ApplicationLogger.render_section(
                sys.stderr,
                "Command cancelled",
                ("Interrupted by user.",),
            )

        return 130
    except (
        LogicyticsError,
        OSError,
        PermissionError,
        ValueError,
    ) as error:
        error_title = "Configuration validation failed" if arguments.command == "config" else "Command error"
        if application_logger is not None:
            with contextlib.suppress(OSError):
                application_logger.event(
                    "EXCEPTION",
                    str(error),
                    source="logicytics.cli.commands",
                    command=str(arguments.command),
                    error_type=type(error).__name__,
                )

                if command_started_at is not None:
                    application_logger.event(
                        "ERROR",
                        "command_finished",
                        source="logicytics.cli.commands",
                        command=str(arguments.command),
                        exit_code=2,
                        status="failed",
                        duration_seconds=round(perf_counter() - command_started_at, 3),
                        error_type=type(error).__name__,
                    )

        if application_logger is not None:
            application_logger.box(error_title, (str(error),))
        else:
            ApplicationLogger.render_section(
                sys.stderr,
                error_title,
                (str(error),),
            )
        return 2


cli_methods = CLI()
if __name__ == "__main__":
    try:
        with terminal_lifecycle():
            raise SystemExit(main())
    except KeyboardInterrupt:
        ApplicationLogger.render_section(sys.stderr, "Command cancelled", ("Interrupted by user.",))
        raise SystemExit(130)
