"""Regression coverage for optional mod discovery and execution."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from logicytics.cli import cli_methods
from logicytics.contracts import Capability, RunRequest
from logicytics.module.configuration import default_config
from logicytics.module.discovery import preflight
from logicytics.module.errors import PreflightError
from logicytics.module.planner import build_plan
from logicytics.module.runtime import RunSupervisor
from tests.fixtures.collectors import mod_metadata


class ModTests(unittest.TestCase):
    """Legacy MOD discovery, validation, security, and execution behavior."""

    def test_mods_require_sidecars_and_run_as_isolated_registered_artifacts(self) -> None:
        """Legacy scripts enter the pipeline only through typed metadata and worker isolation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mods = root / "MODS"
            mods.mkdir()

            script = mods / "example.py"
            script.write_text(
                "from pathlib import Path\n"
                "Path('report.txt').write_text('mod evidence\\n', encoding='utf-8')\n"
                "print('INFO: fixture completed')\n",
                encoding="utf-8",
            )

            report = preflight(root)
            self.assertEqual(1, len(report.invalid))
            self.assertIn(
                "requires metadata sidecar",
                report.invalid[0].static_errors[0],
            )

            with self.assertRaises(PreflightError):
                build_plan(report, RunRequest(enable_mods=True))

            script.with_suffix(".py.mod.json").write_text(
                json.dumps(mod_metadata("example")),
                encoding="utf-8",
            )

            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            self.assertEqual(
                (),
                build_plan(report, RunRequest()).collectors,
            )

            plan = build_plan(
                report,
                RunRequest(
                    enable_mods=True,
                    approved_capabilities=(Capability.SUBPROCESS,),
                    acknowledge_authorization=True,
                    max_workers=1,
                ),
            )

            collector_ids: list[str] = []
            for item in plan.collectors:
                metadata = item.metadata
                if metadata is None:
                    self.fail("planned collector unexpectedly has no metadata")
                collector_ids.append(metadata.id)

            self.assertEqual(["mod.example"], collector_ids)

            outcome = RunSupervisor(
                root,
                default_config(root),
            ).run(plan)

            record = outcome.manifest.collectors[0]
            self.assertEqual(
                "succeeded",
                record.status,
                record.errors,
            )
            self.assertEqual(2, len(record.artifacts))

            report_artifact = next(item for item in record.artifacts if item["name"] == "report.txt")

            artifact_path = outcome.run_directory / "artifacts" / str(report_artifact["relative_path"])

            self.assertEqual(
                "mod evidence\n",
                artifact_path.read_text(encoding="utf-8"),
            )

            package = outcome.manifest.package
            if package is None:
                self.fail("successful MOD run unexpectedly has no package metadata")

            mods_package = Path(package["mods_path"])
            mods_hash = Path(package["mods_sha256_path"])

            self.assertTrue(mods_package.name.startswith("mods-run-"))
            self.assertTrue(mods_hash.is_file())

            with zipfile.ZipFile(mods_package) as archive:
                names = archive.namelist()

                self.assertIn(
                    "metadata/mods.json",
                    names,
                )
                self.assertTrue(any(name.endswith("/report.txt") for name in names))
                self.assertIsNone(archive.testzip())

    def test_python_mod_cannot_mutate_project_configuration_without_write_approval(
        self,
    ) -> None:
        """The Python MOD bootstrap blocks host writes while an independent MOD still succeeds."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration_path = root / "logicytics.yaml"

            configuration_path.write_text(
                '{"schema_version":4}\n',
                encoding="utf-8",
            )

            mods = root / "MODS"
            mods.mkdir()

            scripts = {
                "good.py": ("from pathlib import Path\nPath('report.txt').write_text('ok', encoding='utf-8')\n"),
                "malicious.py": (f"from pathlib import Path\nPath({str(configuration_path)!r}).write_text('replaced', encoding='utf-8')\n"),
            }

            for filename, source in scripts.items():
                script = mods / filename
                script.write_text(
                    source,
                    encoding="utf-8",
                )
                script.with_suffix(".py.mod.json").write_text(
                    json.dumps(mod_metadata(script.stem)),
                    encoding="utf-8",
                )

            report = preflight(root)
            self.assertEqual(
                2,
                len(report.valid),
                report.invalid,
            )

            plan = build_plan(
                report,
                RunRequest(
                    enable_mods=True,
                    approved_capabilities=(Capability.SUBPROCESS,),
                    acknowledge_authorization=True,
                    max_workers=2,
                ),
            )

            outcome = RunSupervisor(
                root,
                default_config(root),
            ).run(plan)

            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertEqual(
                "succeeded",
                records["mod.good"].status,
                records["mod.good"].errors,
            )
            self.assertEqual(
                "failed",
                records["mod.malicious"].status,
            )
            self.assertTrue(
                any("private workspace" in error for error in records["mod.malicious"].errors),
                records["mod.malicious"].errors,
            )

            self.assertEqual(
                '{"schema_version":4}\n',
                configuration_path.read_text(encoding="utf-8"),
            )

    def test_non_python_mod_files_are_rejected_before_sidecar_processing(self) -> None:
        """MODS rejects former native script formats before metadata can enable them."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mods = root / "MODS"
            mods.mkdir()

            script = mods / "native.bat"
            script.write_text(
                "@echo off\n",
                encoding="utf-8",
            )
            script.with_suffix(".bat.mod.json").write_text(
                json.dumps(mod_metadata("native", filesystem_write=True)),
                encoding="utf-8",
            )

            report = preflight(root)

            self.assertEqual(1, len(report.invalid))
            self.assertIn(
                "MODS supports Python (.py) scripts only",
                report.invalid[0].static_errors,
            )

    def test_extensions_mode_selects_python_mods_and_rejects_nopy(self) -> None:
        """The extension mode accepts Python MODS only and has no non-Python alias."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mods = root / "MODS"
            mods.mkdir()

            script = mods / "python_mod.py"
            script.write_text("pass\n", encoding="utf-8")
            script.with_suffix(".py.mod.json").write_text(
                json.dumps(mod_metadata("python_mod")),
                encoding="utf-8",
            )

            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)

            extensions = build_plan(
                report,
                RunRequest(
                    enable_mods=True,
                ),
            )
            self.assertEqual(
                ("mod.python_mod",),
                tuple(item.metadata.id for item in extensions.collectors if item.metadata is not None),
            )

            parser = cli_methods.parser()
            modded_request = cli_methods.request(
                parser.parse_args(["run", "--modded"]),
                2,
            )
            self.assertTrue(modded_request.enable_mods)
            with self.assertRaises(SystemExit):
                parser.parse_args(["run", "--nopy"])


if __name__ == "__main__":
    unittest.main()
