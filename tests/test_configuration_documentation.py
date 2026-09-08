"""Keep the public configuration guide synchronized with the executable schema."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from logicytics.module import configuration
from logicytics.module.errors import PlanError


class ConfigurationDocumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.guide = (self.project_root / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")

    def test_every_engine_and_collector_setting_is_documented(self) -> None:
        field_groups = (
            configuration._ROOT_FIELDS,
            configuration._RUNTIME_FIELDS,
            configuration._INTERACTION_FIELDS,
            configuration._MAINTENANCE_FIELDS,
            configuration._LOGGING_FIELDS,
        )
        for fields in field_groups:
            for field in fields:
                with self.subTest(field=field):
                    self.assertIn(f"`{field}`", self.guide)

        for collector_id, schema in configuration._COLLECTOR_SETTING_SCHEMAS.items():
            with self.subTest(collector=collector_id):
                self.assertIn(f"`{collector_id}`", self.guide)
            for field in schema:
                with self.subTest(collector=collector_id, field=field):
                    self.assertIn(f"`{field}`", self.guide)

    def test_complete_example_loads_through_the_real_parser(self) -> None:
        match = re.search(r"## Complete example.*?```json\n(.*?)\n```", self.guide, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIsNotNone(match)
        assert match is not None

        payload = json.loads(match.group(1))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "logicytics.yaml"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = configuration.load_config(root)
        self.assertEqual(configuration.SCHEMA_VERSION, loaded.schema_version)
        self.assertEqual(4, loaded.runtime.default_max_workers)
        self.assertEqual(16, loaded.runtime.maximum_workers)
        self.assertEqual(3, loaded.settings_for("core.network.bandwidth_sample")["sample_count"])

    def test_repository_guides_link_the_configuration_contract(self) -> None:
        contributor = (self.project_root / "CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn("CONFIGURATION.md", contributor)
        readme = (self.project_root / "README.md").read_text(encoding="utf-8")
        self.assertIn("Logicytics Wiki", readme)

    def test_yaml_is_bounded_and_relative_explicit_paths_use_the_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "selected.yaml"
            selected.write_text("schema_version: 4\n", encoding="utf-8")
            loaded = configuration.load_config(root, Path("selected.yaml"))
            self.assertEqual(configuration.SCHEMA_VERSION, loaded.schema_version)

            selected.write_bytes(b" " * (configuration.MAXIMUM_CONFIGURATION_BYTES + 1))
            with self.assertRaisesRegex(PlanError, "exceeds the 2 MiB limit"):
                configuration.load_config(root, Path("selected.yaml"))

    def test_core_settings_require_an_explicit_schema_while_extensions_remain_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "logicytics.yaml"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "collectors": {"core.system.system_info": {"typo": 1}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PlanError, "does not declare configurable settings"):
                configuration.load_config(root)

            path.write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "collectors": {"plugin.example": {"extension_option": 1}},
                    }
                ),
                encoding="utf-8",
            )
            loaded = configuration.load_config(root)
            self.assertEqual(
                {"extension_option": 1},
                loaded.settings_for("plugin.example"),
            )


if __name__ == "__main__":
    unittest.main()
