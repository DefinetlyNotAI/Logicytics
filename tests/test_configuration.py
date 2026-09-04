from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any, cast

from fixtures.collectors import COLLECTOR
from logicytics.configuration import (
    default_config,
    load_config,
)
from logicytics.contracts import (
    RunRequest,
)
from logicytics.discovery import preflight
from logicytics.errors import PlanError
from logicytics.planner import build_plan
from logicytics.runtime import RunSupervisor


class ConfigurationTests(unittest.TestCase):
    """Configuration parsing, migration, validation, and manifest behavior."""

    def test_configuration_schema_version_is_enforced(self) -> None:
        """Current v4 and explicitly migrated v3 schemas are accepted; other versions are not."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            for unsupported in (0, 1, 2, 5):
                with self.subTest(schema_version=unsupported):
                    config_path.write_text(json.dumps({"schema_version": unsupported}), encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, "unsupported configuration schema_version"):
                        load_config(root)
            config_path.write_text('{"schema_version": 3}', encoding="utf-8")
            migrated = load_config(root)
            self.assertEqual(4, migrated.schema_version)
            self.assertEqual(3, migrated.migrated_from_schema)
            config_path.write_text('{"schema_version": 4, "collectors": {}}', encoding="utf-8")
            current = load_config(root)
            self.assertEqual(4, current.schema_version)
            self.assertIsNone(current.migrated_from_schema)
            config_path.write_text('{"schema_version":4,"runtime":{"package_completed_runs":"yes"}}', encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "package_completed_runs"):
                load_config(root)

    def test_legacy_configuration_migrates_runtime_and_collector_aliases_without_writing(self) -> None:
        """A supported v3 configuration migrates once in memory and preserves its source bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            settings = {"core.packet.packet_capture": {"packet_count": 7, "timeout_seconds": 5}}
            original = json.dumps({
                "schema_version": 3,
                "workers": 3,
                "max_workers": 6,
                "output_root": "ACCESS/RUNS",
                "collector_settings": settings,
            }, indent=2)
            config_path.write_text(original, encoding="utf-8")
            configuration = load_config(root)

            self.assertEqual(4, configuration.schema_version)
            self.assertEqual(3, configuration.migrated_from_schema)
            self.assertEqual(3, configuration.runtime.default_max_workers)
            self.assertEqual(6, configuration.runtime.maximum_workers)
            self.assertEqual(root / "output" / "data", configuration.runtime.output_root)
            self.assertEqual(settings["core.packet.packet_capture"],
                             configuration.settings_for("core.packet.packet_capture"))
            self.assertEqual(original, config_path.read_text(encoding="utf-8"))

    def test_historical_code_config_ini_migrates_into_typed_v4_settings(self) -> None:
        """The original CODE/config.ini is a bounded read-only fallback when JSON is absent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "CODE"
            code.mkdir()
            config_path = code / "config.ini"
            original = """[Settings]
log_using_debug = true
delete_old_logs = true
max_workers = 6
save_preferences = false

[Flag Settings]
accuracy_min = 30.0
model_to_use = all-MiniLM-L6-v2
model_debug = true

[DumpMemory Settings]
file_size_limit = 8
file_size_safety = 1.5

[NetWorkPsutil Settings]
sample_count = 5
interval = 1.5

[PacketSniffer Settings]
interface = WiFi
packet_count = 5000
timeout = 10
max_retry_time = 30
"""
            config_path.write_text(original, encoding="utf-8")

            configuration = load_config(root)

            self.assertEqual(3, configuration.migrated_from_schema)
            self.assertEqual(6, configuration.runtime.default_max_workers)
            self.assertEqual(6, configuration.runtime.maximum_workers)
            self.assertEqual("DEBUG", configuration.logging.level)
            self.assertTrue(configuration.logging.delete_previous)
            self.assertFalse(configuration.interaction.history_enabled)
            self.assertEqual(0.3, configuration.interaction.similarity_threshold)
            self.assertEqual("all-MiniLM-L6-v2", configuration.interaction.model_name)
            self.assertTrue(configuration.interaction.model_debug)
            self.assertEqual(
                {
                    "output_limit_bytes": 8 * 1024 * 1024,
                    "disk_safety_margin_bytes": 4 * 1024 * 1024,
                    "dump_directory": "memory_maps",
                },
                configuration.settings_for("core.process.memory_map"),
            )
            self.assertEqual(
                {"sample_count": 5, "interval_seconds": 1.5},
                configuration.settings_for("core.network.bandwidth_sample"),
            )
            self.assertEqual(5000, configuration.settings_for("core.packet.packet_capture")["packet_count"])
            self.assertEqual(original, config_path.read_text(encoding="utf-8"))

            (root / "logicytics.json").write_text('{"schema_version":4}', encoding="utf-8")
            self.assertIsNone(load_config(root).migrated_from_schema)
            self.assertFalse((root / "output").exists())
            self.assertEqual(3, configuration.to_manifest_dict()["migrated_from_schema"])

    def test_legacy_configuration_rejects_ambiguous_unsafe_and_plugin_enabling_migrations(self) -> None:
        """Migration cannot override settings, weaken validation, or silently enable extensions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            invalid = (
                ({"workers": 2, "worker_count": 3}, "conflicting settings"),
                ({"workers": 2, "runtime": {"default_max_workers": 2}}, "conflicting settings"),
                ({"runtime": {"workers": 2, "worker_count": 3}}, "conflicting settings"),
                ({"collectors": {}, "collector_settings": {}}, "conflicting collectors"),
                ({"enable_plugins": True}, "unsupported root"),
                ({"workers": True}, "worker limits"),
                ({"collector_settings": {"core.process.memory_map": {"dump_directory": "../outside"}}},
                 "collector-workspace"),
                ({"collector_settings": {"core.packet.packet_capture": {"packet_count": 0}}}, "packet_count"),
            )
            for legacy, message in invalid:
                with self.subTest(legacy=legacy):
                    config_path.write_text(json.dumps({"schema_version": 3, **legacy}), encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)
                    self.assertFalse((root / "output").exists())

    def test_migrated_configuration_run_preserves_manifest_provenance_and_legacy_evidence(self) -> None:
        """Migrated collection stays isolated, packages normally, and never moves old evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")

            legacy = root / "ACCESS" / "RUNS" / "previous.txt"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("keep legacy evidence", encoding="utf-8")

            config_path = root / "logicytics.json"
            original = json.dumps(
                {
                    "schema_version": 3,
                    "worker_count": 1,
                    "output_root": "ACCESS/RUNS",
                }
            )
            config_path.write_text(original, encoding="utf-8")

            configuration = load_config(root)
            plan = build_plan(
                preflight(root),
                RunRequest(
                    max_workers=1,
                    acknowledge_authorization=True,
                ),
            )
            outcome = RunSupervisor(root, configuration).run(plan)

            self.assertEqual(root / "output" / "data", outcome.run_directory.parent)
            self.assertEqual(3, outcome.manifest.configuration["migrated_from_schema"])
            self.assertEqual("keep legacy evidence", legacy.read_text(encoding="utf-8"))
            self.assertEqual(original, config_path.read_text(encoding="utf-8"))

            package = outcome.manifest.package
            if package is None:
                self.fail("expected the completed run to contain package metadata")

            with zipfile.ZipFile(Path(package["path"])) as archive:
                packaged = json.loads(archive.read("metadata/manifest.json"))

            self.assertEqual(4, packaged["configuration"]["schema_version"])
            self.assertEqual(3, packaged["configuration"]["migrated_from_schema"])

    def test_configuration_defaults_to_one_canonical_output_data_root(self) -> None:
        """Defaults and loaded settings share output/data while explicit roots remain supported."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "output" / "data"
            self.assertEqual(expected, default_config(root).runtime.output_root)
            self.assertEqual(expected, load_config(root).runtime.output_root)
            config_path = root / "logicytics.json"
            config_path.write_text('{"schema_version":4,"runtime":{}}', encoding="utf-8")
            self.assertEqual(expected, load_config(root).runtime.output_root)
            config_path.write_text(
                '{"schema_version":4,"runtime":{"output_root":"custom/evidence"}}',
                encoding="utf-8",
            )
            self.assertEqual(root / "custom" / "evidence", load_config(root).runtime.output_root)
            self.assertFalse(expected.exists())

    def test_configuration_manifest_redacts_nested_secrets_without_mutating_worker_settings(self) -> None:
        """Manifest snapshots hide credentials while collectors retain configured access."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_id = "plugin.private_keys"
            settings = {
                "password": "password-value",
                "nested": {"refresh_token": "token-value", "cookie": "cookie-value"},
                "private_key": "private-value",
                "ordinary": "safe",
            }
            (root / "logicytics.json").write_text(
                json.dumps({"schema_version": 4, "collectors": {collector_id: settings}}),
                encoding="utf-8",
            )
            configuration = load_config(root)
            self.assertEqual(settings, configuration.settings_for(collector_id))
            snapshot = configuration.to_manifest_dict()
            manifest_settings = cast(dict[str, Any], snapshot["collector_settings"][collector_id])
            self.assertEqual("[REDACTED]", manifest_settings["password"])
            self.assertEqual("[REDACTED]", manifest_settings["nested"]["refresh_token"])
            self.assertEqual("[REDACTED]", manifest_settings["nested"]["cookie"])
            self.assertEqual("[REDACTED]", manifest_settings["private_key"])
            self.assertEqual("safe", manifest_settings["ordinary"])
            for secret in ("password-value", "token-value", "cookie-value", "private-value"):
                self.assertNotIn(secret, json.dumps(snapshot))
            self.assertEqual(settings, configuration.settings_for(collector_id))

    def test_configuration_rejects_boolean_workers_and_invalid_output_roots(self) -> None:
        """Runtime worker limits and output locations must retain strict JSON types."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            invalid_configurations = (
                ('{"schema_version":true}', "schema_version"),
                ('{"schema_version":4,"runtime":{"default_max_workers":true}}', "worker limits"),
                ('{"schema_version":4,"runtime":{"maximum_workers":true}}', "worker limits"),
                ('{"schema_version":4,"runtime":{"maximum_run_output_bytes":true}}', "maximum_run_output_bytes"),
                ('{"schema_version":4,"runtime":{"maximum_run_output_bytes":0}}', "maximum_run_output_bytes"),
                ('{"schema_version":4,"runtime":{"output_root":false}}', "output_root"),
                ('{"schema_version":4,"runtime":{"output_root":"   "}}', "output_root"),
            )
            for payload, message in invalid_configurations:
                with self.subTest(payload=payload):
                    config_path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)

    def test_configuration_rejects_unknown_duplicate_and_non_finite_values(self) -> None:
        """Ambiguous keys, typos, malformed IDs, and nonstandard JSON fail before planning."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            invalid = (
                ('{"schema_version":4,"unexpected":true}', "unsupported root"),
                ('{"schema_version":4,"runtime":{"worker_typo":2}}', "unsupported settings"),
                ('{"schema_version":4,"schema_version":4}', "duplicate configuration key"),
                ('{"schema_version":4,"runtime":{"maximum_workers":NaN}}', "non-finite"),
                ('{"schema_version":4,"collectors":{"plugin.custom":{"value":1e999}}}', "non-finite"),
                ('{"schema_version":4,"collectors":{"../escape":{}}}', "invalid collector ID"),
                ('{"schema_version":4,"collectors":{"core.system.example":{"invalid-name":1}}}',
                 "invalid setting name"),
                ('{"schema_version":4,"collectors":{"core.packet.packet_capture":{"packet_typo":1}}}',
                 "unsupported settings"),
            )
            for payload, message in invalid:
                with self.subTest(payload=payload):
                    config_path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)
                    self.assertFalse((root / "output").exists())
            config_path.write_text(
                '{"schema_version":4,"collectors":{"plugin.custom":{"extension_setting":"allowed"}}}',
                encoding="utf-8",
            )
            self.assertEqual("allowed", load_config(root).settings_for("plugin.custom")["extension_setting"])

    def test_configuration_validates_filesystem_and_sensitive_inventory_bounds(self) -> None:
        """Traversal and sensitive inventory limits cannot silently coerce or expand in workers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            invalid = (
                ("core.filesystem.system_drive_tree", {"max_entries": True}, "max_entries"),
                ("core.filesystem.system_drive_tree", {"max_depth": 33}, "max_depth"),
                ("core.filesystem.system_drive_listing", {"workers": 2}, "unsupported settings"),
                ("core.filesystem.system_drive_listing", {"max_entries": "100"}, "max_entries"),
                ("core.filesystem.sensitive_file_inventory", {"max_directories": 50_001}, "max_directories"),
                ("core.filesystem.sensitive_file_inventory", {"max_matches": 0}, "max_matches"),
                ("core.filesystem.sensitive_file_inventory", {"root": "relative/path"}, "absolute"),
            )
            for collector_id, settings, message in invalid:
                with self.subTest(collector_id=collector_id, settings=settings):
                    config_path.write_text(
                        json.dumps({"schema_version": 4, "collectors": {collector_id: settings}}),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)
                    self.assertFalse((root / "output").exists())
            valid = {
                "core.filesystem.system_drive_tree": {"max_entries": 100, "max_depth": 3},
                "core.filesystem.system_drive_listing": {"max_entries": 200, "max_depth": 4},
                "core.filesystem.sensitive_file_inventory": {
                    "root": str(root), "max_directories": 100, "max_matches": 10,
                },
            }
            config_path.write_text(json.dumps({"schema_version": 4, "collectors": valid}), encoding="utf-8")
            configuration = load_config(root)
            for collector_id, settings in valid.items():
                self.assertEqual(settings, configuration.settings_for(collector_id))

    def test_configuration_validates_metadata_only_memory_map_limits_and_workspace_paths(self) -> None:
        """The existing metadata-only memory mapper rejects unsafe values before worker launch."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            collector_id = "core.process.memory_map"
            invalid = (
                ({"max_regions": True}, "max_regions"),
                ({"max_regions": 100_001}, "max_regions"),
                ({"output_limit_bytes": 1_023}, "output_limit_bytes"),
                ({"output_limit_bytes": 64 * 1024 * 1024 + 1}, "output_limit_bytes"),
                ({"disk_safety_margin_bytes": -1}, "disk_safety_margin_bytes"),
                ({"disk_safety_margin_bytes": "100"}, "disk_safety_margin_bytes"),
                ({"dump_directory": "../outside"}, "collector-workspace"),
                ({"dump_directory": str(root)}, "collector-workspace"),
                ({"dump_directory": "   "}, "non-empty"),
            )
            for settings, message in invalid:
                with self.subTest(settings=settings):
                    config_path.write_text(
                        json.dumps({"schema_version": 4, "collectors": {collector_id: settings}}),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(PlanError, message):
                        load_config(root)
                    self.assertFalse((root / "output").exists())
            settings = {
                "max_regions": 500,
                "output_limit_bytes": 4096,
                "disk_safety_margin_bytes": 0,
                "dump_directory": "bounded/maps",
            }
            config_path.write_text(
                json.dumps({"schema_version": 4, "collectors": {collector_id: settings}}),
                encoding="utf-8",
            )
            self.assertEqual(settings, load_config(root).settings_for(collector_id))

    def test_configuration_profile_collector_and_request_boundaries_are_isolated(self) -> None:
        """Product, profile, collector, and invocation settings remain separate contracts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_id = "core.process.memory_map"
            collector_settings = {
                "output_limit_bytes": 8192,
                "disk_safety_margin_bytes": 4096,
                "dump_directory": "maps",
            }
            (root / "logicytics.json").write_text(
                json.dumps({
                    "schema_version": 4,
                    "runtime": {"default_max_workers": 3, "maximum_workers": 8},
                    "collectors": {collector_id: collector_settings},
                }),
                encoding="utf-8",
            )
            configuration = load_config(root)
            request = RunRequest(
                profile="deep",
                include=(collector_id,),
                max_workers=1,
                acknowledge_authorization=True,
            )

            self.assertEqual(3, configuration.runtime.default_max_workers)
            self.assertEqual(collector_settings, configuration.settings_for(collector_id))
            self.assertEqual("deep", request.profile)
            self.assertEqual((collector_id,), request.include)
            self.assertEqual(1, request.max_workers)
            self.assertNotIn("profile", configuration.to_manifest_dict()["runtime"])
            self.assertNotIn("max_workers", configuration.to_manifest_dict()["runtime"])

    def test_configuration_validates_bounded_network_and_packet_settings(self) -> None:
        """Collector-specific settings fail early rather than being silently coerced at runtime."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "logicytics.json"
            config_path.write_text(
                '{"schema_version":4,"collectors":{"core.network.bandwidth_sample":'
                '{"sample_count":11}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PlanError, "sample_count"):
                load_config(root)
            config_path.write_text(
                '{"schema_version":4,"collectors":{"core.packet.packet_capture":'
                '{"packet_count":10,"timeout_seconds":5,"retry_window_seconds":2,"interface":"127.0.0.1"}}}',
                encoding="utf-8",
            )
            settings = load_config(root).settings_for("core.packet.packet_capture")
            self.assertEqual(10, settings["packet_count"])
            self.assertEqual(2, settings["retry_window_seconds"])


if __name__ == "__main__":
    unittest.main()
