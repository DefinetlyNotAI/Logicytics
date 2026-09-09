"""Contract-level checks for collectors shipped in the repository tree."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import tempfile
import unittest
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, NoReturn, cast
from unittest.mock import patch

from logicytics import ResourceClass
from logicytics.contracts import (
    ArtifactWriter,
    CollectorContext,
    CollectorStatus,
    EventLogger,
    EvidenceKind,
)
from logicytics.module.artifacts import WorkspaceArtifactWriter
from logicytics.module.discovery import preflight
from logicytics.module.modes import EXECUTION_MODES, LEGACY_MODE_ALIASES, mode_matrix
from logicytics.module.output_contracts import core_output_contract
from logicytics.platform_adapters import (
    filesystem_adapter,
    network_adapter,
    process_adapter,
    registry_adapter,
    windows_api_adapter,
)


class _RejectingWriter(ArtifactWriter):
    """Prove cancellation paths never attempt to publish evidence."""

    def register_file(
        self,
        source: Path,
        *,
        media_type: str = "application/octet-stream",
        evidence_kind: EvidenceKind = EvidenceKind.DERIVED,
        transformations: tuple[str, ...] = (),
    ) -> NoReturn:
        raise AssertionError("a cancelled collector must not register an artifact")


class _NoopLogger(EventLogger):
    """Discard test events."""

    def event(
        self,
        level: str,
        message: str,
        **fields: float | str,
    ) -> None:
        return None


def _load_collector(path: Path, class_name: str) -> Any:
    """Load and instantiate one collector class from a source module."""
    module_name = f"contract_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load collector module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    collector_object: object = getattr(module, class_name, None)

    if not isinstance(collector_object, type):
        raise TypeError(f"{class_name} in {path} is missing or is not a class")

    collector_factory = cast(Callable[[], Any], collector_object)
    return collector_factory()


class ShippedCollectorTests(unittest.TestCase):
    """Ensure every checked-in core collector passes strict discovery rules."""

    def test_shipped_collectors_pass_preflight(self) -> None:
        """The repository must never contain an invalid shipped collector."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)

        self.assertEqual((), report.invalid)

        discovered_ids: set[str] = set()

        for candidate in report.valid:
            metadata = candidate.metadata
            if metadata is None:
                continue

            discovered_ids.add(metadata.id)

        self.assertEqual(
            {
                "core.bluetooth.paired_devices",
                "core.bluetooth.bluetooth_addresses",
                "core.bluetooth.bluetooth_history",
                "core.browser.browser_data_backup",
                "core.network.network_identity",
                "core.network.active_connections",
                "core.network.connection_processes",
                "core.network.dns_cache",
                "core.network.firewall_profiles",
                "core.network.adapter_statistics",
                "core.network.bandwidth_sample",
                "core.network.network_interfaces",
                "core.network.network_adapters",
                "core.network.arp_cache",
                "core.network.routing_table",
                "core.packet.packet_capture",
                "core.packet.connection_graph",
                "core.memory.memory_snapshot",
                "core.media.media_backup",
                "core.process.running_processes",
                "core.process.detailed_processes",
                "core.process.process_memory",
                "core.process.memory_map",
                "core.registry.installed_applications",
                "core.registry.hklm_backup",
                "core.registry.startup_applications",
                "core.storage.logical_drives",
                "core.storage.physical_disks",
                "core.storage.mounted_volumes",
                "core.storage.volume_details",
                "core.ssh.ssh_backup",
                "core.system.system_info",
                "core.system.system_details",
                "core.system.bios_info",
                "core.system.operating_system",
                "core.system.computer_system",
                "core.system.defender_status",
                "core.system.session_snapshot",
                "core.system.environment_posture",
                "core.system.system_diagnostics",
                "core.system.installed_drivers",
                "core.system.installed_updates",
                "core.system.local_accounts",
                "core.system.scheduled_tasks",
                "core.system.windows_services",
                "core.system.windows_system_data_backup",
                "core.system.wmic_inventory",
                "core.diagnostics.sysinternals_report",
                "core.system.group_policy",
                "core.usb.usb_storage_inventory",
                "core.wireless.wifi_profiles",
                "core.wireless.wifi_interfaces",
                "core.wireless.wifi_profile_keys",
                "core.hardware.windows_features",
                "core.hardware.display_adapters",
                "core.hardware.battery_status",
                "core.integration.legacy_code_outputs",
                "core.event_log.system_events",
                "core.event_log.application_events",
                "core.event_log.security_events",
                "core.filesystem.system_drive_tree",
                "core.filesystem.startup_folder_entries",
                "core.filesystem.sensitive_file_inventory",
                "core.filesystem.system_drive_listing",
                "core.encryption.bitlocker_status",
                "core.encryption.bitlocker_volumes",
            },
            discovered_ids,
        )

    def test_event_log_collectors_explicitly_allow_bounded_parallel_scheduling(
        self,
    ) -> None:
        """The independent event channels may overlap while retaining separate identities."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)

        event_collectors = {}

        for candidate in report.valid:
            metadata = candidate.metadata
            if metadata is None:
                continue

            if not metadata.id.startswith("core.event_log."):
                continue

            event_collectors[metadata.id] = metadata

        self.assertEqual(
            {
                "core.event_log.application_events",
                "core.event_log.security_events",
                "core.event_log.system_events",
            },
            set(event_collectors),
        )

        for metadata in event_collectors.values():
            self.assertTrue(metadata.parallel_safe)
            self.assertIs(
                ResourceClass.GENERAL,
                metadata.resource_class,
            )
            self.assertEqual(
                ("text/csv",),
                metadata.output_media_types,
            )

    def test_mode_matrix_represents_every_discovered_collector_exactly_once(
        self,
    ) -> None:
        """Mode documentation covers selected, manual-only, and quarantined collectors."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)
        matrix = mode_matrix(report.candidates)

        collector_rows = matrix["collectors"]

        expected_ids: set[str] = set()

        for candidate in report.candidates:
            metadata = candidate.metadata

            if metadata is not None:
                expected_ids.add(metadata.id)
            else:
                expected_ids.add(candidate.selection_id)

        represented_ids = [row["id"] for row in collector_rows]

        self.assertEqual(
            expected_ids,
            set(represented_ids),
        )
        self.assertEqual(
            len(represented_ids),
            len(set(represented_ids)),
        )

        modes = {row["name"]: row for row in matrix["modes"]}

        self.assertEqual(
            set(EXECUTION_MODES),
            set(modes),
        )

        for collector in collector_rows:
            with self.subTest(collector=collector["id"]):
                self.assertEqual(
                    collector["valid"] and not collector["modes"],
                    collector["manual_only"],
                )

                for mode_name in collector["modes"]:
                    self.assertIn(
                        collector["id"],
                        modes[mode_name]["collector_ids"],
                    )

        for mode_name, mode in modes.items():
            expected = {row["id"] for row in collector_rows if mode_name in row["modes"]}

            self.assertEqual(
                expected,
                set(mode["collector_ids"]),
            )

    def test_every_core_collector_uses_context_artifacts_without_mutable_globals(
        self,
    ) -> None:
        """Migration is complete only when all core modules use owned workspaces and catalogs."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)

        core_candidates = [candidate for candidate in report.valid if candidate.kind.value == "core"]

        self.assertTrue(core_candidates)

        for candidate in core_candidates:
            metadata = candidate.metadata
            if metadata is None:
                self.fail(f"valid core candidate {candidate.selection_id} has no metadata")

            with self.subTest(collector=metadata.id):
                source = candidate.path.read_text(encoding="utf-8")
                tree = ast.parse(
                    source,
                    filename=str(candidate.path),
                )

                self.assertFalse(
                    any(
                        isinstance(
                            node,
                            (ast.Global, ast.Nonlocal),
                        )
                        for node in ast.walk(tree)
                    ),
                    "collectors must not declare shared mutable state",
                )

                collect: ast.FunctionDef | ast.AsyncFunctionDef | None = None

                for node in ast.walk(tree):
                    if (
                        isinstance(
                            node,
                            (
                                ast.FunctionDef,
                                ast.AsyncFunctionDef,
                            ),
                        )
                        and node.name == "collect"
                    ):
                        collect = node
                        break

                if collect is None:
                    self.fail(f"{metadata.id} does not define collect()")

                attributes = {node.attr for node in ast.walk(collect) if isinstance(node, ast.Attribute)}

                self.assertIn("workspace", attributes)
                self.assertIn("artifacts", attributes)
                self.assertIn("register_file", attributes)

    def test_every_core_collector_has_a_stable_output_contract(
        self,
    ) -> None:
        """Every shipped output has explicit names, formats, package paths, and retention."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)

        for candidate in report.valid:
            if candidate.kind.value != "core":
                continue

            metadata = candidate.metadata
            if metadata is None:
                self.fail(f"valid core candidate {candidate.selection_id} has no metadata")

            with self.subTest(collector=metadata.id):
                contract = core_output_contract(metadata)

                self.assertEqual(
                    metadata.output_media_types,
                    contract.media_types,
                )
                self.assertTrue(contract.workspace_patterns)
                self.assertTrue(all("\\" not in pattern and not pattern.startswith("/") for pattern in contract.workspace_patterns))
                self.assertEqual(
                    len(contract.workspace_patterns),
                    len(contract.package_patterns),
                )
                self.assertTrue(all(path.startswith("evidence/{kind}/core_") for path in contract.package_patterns))
                self.assertEqual(
                    "retained_with_run",
                    contract.retention,
                )

    def test_every_core_collector_obeys_the_typed_lifecycle_contract(
        self,
    ) -> None:
        """All shipped collectors fail closed on cancellation without platform access or artifacts."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)

        for candidate in report.valid:
            if candidate.kind.value != "core":
                continue

            metadata = candidate.metadata
            if metadata is None:
                self.fail(f"valid core candidate {candidate.selection_id} has no metadata")

            with (
                self.subTest(collector=metadata.id),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                cancellation = root / ".cancelled"
                cancellation.touch()

                context = CollectorContext(
                    run_id="run-" + "0" * 32,
                    collector_id=metadata.id,
                    workspace=root,
                    temporary_directory=root / "tmp",
                    artifacts=_RejectingWriter(),
                    logger=_NoopLogger(),
                    settings={},
                    cancellation_file=cancellation,
                )

                collector = _load_collector(
                    candidate.path,
                    candidate.expected_class,
                )

                validation = collector.validate(context)
                self.assertFalse(validation.valid)
                self.assertTrue(validation.reasons)

                prepared = collector.prepare(context)
                self.assertTrue(hasattr(prepared, "valid"))

                result = collector.collect(context)

                self.assertIs(
                    CollectorStatus.CANCELLED,
                    result.status,
                )
                self.assertFalse(result.artifacts)

                self.assertIs(
                    result,
                    collector.finalize(
                        context,
                        result,
                    ),
                )

                collector.cleanup(context)

    def test_every_registered_artifact_media_type_is_declared(
        self,
    ) -> None:
        """Static artifact calls cannot introduce an undeclared output format."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)

        for candidate in report.valid:
            if candidate.kind.value != "core":
                continue

            metadata = candidate.metadata
            if metadata is None:
                self.fail(f"valid core candidate {candidate.selection_id} has no metadata")

            tree = ast.parse(
                candidate.path.read_text(encoding="utf-8"),
                filename=str(candidate.path),
            )

            register_calls: list[ast.Call] = []

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                if (
                    isinstance(
                        node.func,
                        ast.Attribute,
                    )
                    and node.func.attr == "register_file"
                ):
                    register_calls.append(node)

            declared_at_calls: set[str] = set()

            for call in register_calls:
                for keyword in call.keywords:
                    if keyword.arg != "media_type":
                        continue

                    value = keyword.value

                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        declared_at_calls.add(value.value)

            has_implicit_media_type = any(not any(keyword.arg == "media_type" for keyword in call.keywords) for call in register_calls)

            if has_implicit_media_type:
                declared_at_calls.add("application/octet-stream")

            with self.subTest(collector=metadata.id):
                self.assertTrue(register_calls)
                self.assertTrue(declared_at_calls)
                self.assertLessEqual(
                    declared_at_calls,
                    set(metadata.output_media_types),
                )

    def test_migration_documentation_covers_every_supported_bridge(
        self,
    ) -> None:
        """Public compatibility stays explicit and canonical-output-only."""
        project_root = Path(__file__).resolve().parent.parent
        migration = (project_root / "docs" / "MIGRATION.md").read_text(encoding="utf-8")

        command_flags = {
            "default_mode": "default",
            "performance_check": "performance-check",
        }

        for flag in LEGACY_MODE_ALIASES:
            with self.subTest(flag=flag):
                self.assertIn(
                    f"`--{command_flags.get(flag, flag.replace('_', '-'))}`",
                    migration,
                )

        for bridge in (
            "CODE/config.ini",
            "core.integration.legacy_code_outputs",
            "MODS/",
        ):
            self.assertIn(
                bridge,
                migration,
            )

        self.assertIn(
            "There are no global `ACCESS/`, `RUNS/`, `LOGS/`, or `PACKAGES/`",
            migration,
        )

        legacy_metadata = None

        for candidate in preflight(project_root).valid:
            metadata = candidate.metadata
            if metadata is None:
                continue

            if metadata.id == "core.integration.legacy_code_outputs":
                legacy_metadata = metadata
                break

        if legacy_metadata is None:
            self.fail("core.integration.legacy_code_outputs was not discovered during preflight")

        contract = core_output_contract(legacy_metadata)

        self.assertEqual(
            ("legacy_code/**",),
            contract.workspace_patterns,
        )

    def test_each_core_module_owns_exactly_one_policy_contract(
        self,
    ) -> None:
        """Different permission, sensitivity, timeout, or output policies require separate IDs."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)

        signatures: dict[
            str,
            tuple[object, ...],
        ] = {}

        for candidate in report.valid:
            if candidate.kind.value != "core":
                continue

            metadata = candidate.metadata
            if metadata is None:
                self.fail(f"valid core candidate {candidate.selection_id} has no metadata")

            tree = ast.parse(
                candidate.path.read_text(encoding="utf-8"),
                filename=str(candidate.path),
            )

            collector_classes: list[ast.ClassDef] = []

            for node in tree.body:
                if not isinstance(
                    node,
                    ast.ClassDef,
                ):
                    continue

                inherits_core_collector = any(ast.unparse(base).endswith("CoreCollector") for base in node.bases)

                if inherits_core_collector:
                    collector_classes.append(node)

            with self.subTest(collector=metadata.id):
                self.assertEqual(
                    [candidate.expected_class],
                    [node.name for node in collector_classes],
                )

                contract = core_output_contract(metadata)

                signatures[metadata.id] = (
                    metadata.capabilities,
                    metadata.privilege_level,
                    metadata.network_access,
                    metadata.sensitive_data_categories,
                    metadata.timeout_seconds,
                    metadata.estimated_cost,
                    contract.workspace_patterns,
                    contract.media_types,
                )

        self.assertEqual(
            len(signatures),
            len(set(signatures)),
        )

    def test_every_core_collector_handles_mocked_windows_platform_responses(
        self,
    ) -> None:
        """Each collector returns a typed result when Windows adapters report unavailable data."""
        project_root = Path(__file__).resolve().parent.parent
        report = preflight(project_root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty_host = root / "empty-host"
            empty_host.mkdir()

            with ExitStack() as mocks:
                mocks.enter_context(
                    patch.object(
                        process_adapter,
                        "run",
                        return_value=subprocess.CompletedProcess(
                            (),
                            1,
                            "",
                            "Access is denied",
                        ),
                    )
                )

                mocks.enter_context(
                    patch.object(
                        registry_adapter,
                        "OpenKey",
                        side_effect=OSError("missing key"),
                    )
                )

                mocks.enter_context(
                    patch.object(
                        windows_api_adapter,
                        "load_library",
                        side_effect=OSError("API unavailable"),
                    )
                )

                mocks.enter_context(
                    patch.object(
                        windows_api_adapter,
                        "is_administrator",
                        return_value=False,
                    )
                )

                mocks.enter_context(
                    patch.object(
                        network_adapter,
                        "gethostname",
                        return_value="golden-host",
                    )
                )

                mocks.enter_context(
                    patch.object(
                        network_adapter,
                        "gethostbyname",
                        return_value="192.0.2.1",
                    )
                )

                mocks.enter_context(
                    patch.object(
                        network_adapter,
                        "getaddrinfo",
                        return_value=[],
                    )
                )

                mocks.enter_context(
                    patch.object(
                        network_adapter,
                        "socket",
                        side_effect=PermissionError("denied"),
                    )
                )

                mocks.enter_context(
                    patch.object(
                        filesystem_adapter,
                        "home",
                        return_value=empty_host,
                    )
                )

                mocks.enter_context(
                    patch.object(
                        filesystem_adapter,
                        "system_drive_root",
                        return_value=empty_host,
                    )
                )

                mocks.enter_context(
                    patch.object(
                        filesystem_adapter,
                        "environment_path",
                        return_value=empty_host,
                    )
                )

                for candidate in report.valid:
                    if candidate.kind.value != "core":
                        continue

                    metadata = candidate.metadata
                    if metadata is None:
                        self.fail(f"valid core candidate {candidate.selection_id} has no metadata")

                    with self.subTest(collector=metadata.id):
                        metadata = candidate.metadata
                        self.assertIsNotNone(metadata)
                        assert metadata is not None

                        workspace = root / metadata.id.replace(".", "_")

                        artifact_root = workspace / "published"

                        workspace.mkdir()
                        artifact_root.mkdir()

                        context = CollectorContext(
                            run_id="run-" + "0" * 32,
                            collector_id=metadata.id,
                            workspace=workspace,
                            temporary_directory=(workspace / "tmp"),
                            artifacts=WorkspaceArtifactWriter(
                                metadata.id,
                                workspace,
                                artifact_root,
                                metadata.maximum_output_bytes,
                                metadata.maximum_artifact_files,
                            ),
                            logger=_NoopLogger(),
                            settings={},
                            cancellation_file=(workspace / ".cancelled"),
                        )

                        collector = _load_collector(
                            candidate.path,
                            candidate.expected_class,
                        )

                        result = collector.collect(context)

                        self.assertIn(
                            result.status,
                            set(CollectorStatus),
                        )

    def test_default_profiles_keep_quick_runs_small_and_deep_runs_complete(
        self,
    ) -> None:
        """Minimal and standard stay bounded while deep explicitly owns the exhaustive surface."""
        project_root = Path(__file__).resolve().parent.parent

        metadata = []

        for candidate in preflight(project_root).valid:
            if candidate.kind.value != "core":
                continue

            candidate_metadata = candidate.metadata
            if candidate_metadata is None:
                continue

            metadata.append(candidate_metadata)

        memberships = {
            profile: {item.id for item in metadata if profile in item.default_profiles}
            for profile in (
                "minimal",
                "standard",
                "deep",
                "offline",
            )
        }

        self.assertEqual(
            3,
            len(memberships["minimal"]),
        )
        self.assertEqual(
            {
                "core.encryption.bitlocker_status",
                "core.encryption.bitlocker_volumes",
                "core.hardware.battery_status",
                "core.hardware.display_adapters",
                "core.hardware.windows_features",
                "core.memory.memory_snapshot",
                "core.network.firewall_profiles",
                "core.network.network_adapters",
                "core.network.network_identity",
                "core.process.running_processes",
                "core.registry.installed_applications",
                "core.registry.startup_applications",
                "core.storage.logical_drives",
                "core.storage.mounted_volumes",
                "core.storage.physical_disks",
                "core.storage.volume_details",
                "core.system.bios_info",
                "core.system.computer_system",
                "core.system.defender_status",
                "core.system.environment_posture",
                "core.system.group_policy",
                "core.system.installed_drivers",
                "core.system.installed_updates",
                "core.system.operating_system",
                "core.system.system_details",
                "core.system.system_diagnostics",
                "core.system.system_info",
                "core.system.windows_services",
            },
            memberships["standard"],
        )
        self.assertNotIn("core.system.wmic_inventory", memberships["deep"])
        self.assertNotIn("core.integration.legacy_code_outputs", memberships["deep"])
        self.assertEqual(
            {
                item.id
                for item in metadata
                if item.id not in {"core.system.wmic_inventory", "core.integration.legacy_code_outputs"}
            },
            memberships["deep"],
        )
        self.assertLess(
            memberships["minimal"],
            memberships["standard"],
        )
        self.assertLess(
            memberships["standard"],
            memberships["deep"],
        )
        self.assertEqual(
            memberships["minimal"],
            memberships["offline"],
        )
        self.assertTrue(
            all(
                set(item.sensitive_data_categories).issubset(
                    {
                        "encryption_configuration",
                        "hardware_inventory",
                        "security_configuration",
                        "system_configuration",
                    }
                )
                for item in metadata
                if item.id in memberships["standard"]
            )
        )


if __name__ == "__main__":
    unittest.main()
