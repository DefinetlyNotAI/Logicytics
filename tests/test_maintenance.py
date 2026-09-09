"""Regression coverage for maintenance and integrity-manifest behavior."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from logicytics.cli import CLI, main
from logicytics.module import (
    maintenance,
)
from logicytics.module.configuration import (
    MaintenanceSettings,
    load_config,
)
from logicytics.module.errors import PlanError
from logicytics.module.maintenance import (
    build_manifest,
    compare_files,
    compare_versions,
    fetch_remote_manifest,
    project_files,
    write_local_manifest,
)
from logicytics.module.sysinternals import ensure_sysinternals


class MaintenanceTests(unittest.TestCase):
    """Maintenance, integrity manifests, developer actions, and Sysinternals behavior."""

    def test_maintenance_configuration_requires_pinned_https_and_python_order(
        self,
    ) -> None:
        """Integrity endpoints and Python policy fail closed during configuration loading."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "logicytics.yaml"
            valid = {
                "schema_version": 4,
                "maintenance": {
                    "remote_manifest_url": "https://example.invalid/project.manifest.json",
                    "remote_manifest_sha256": "a" * 64,
                    "local_manifest_path": "integrity/project.json",
                    "minimum_python": "3.11",
                    "recommended_python": "3.12",
                },
            }
            config.write_text(json.dumps(valid), encoding="utf-8")
            loaded = load_config(root, config)
            self.assertEqual(Path("integrity/project.json"), loaded.maintenance.local_manifest_path)
            self.assertEqual("3.12", loaded.maintenance.recommended_python)

            for mutation in (
                {"remote_manifest_url": "http://example.invalid/project.json"},
                {"remote_manifest_sha256": None},
                {"local_manifest_path": "../escape.json"},
                {"minimum_python": "3.12", "recommended_python": "3.11"},
            ):
                invalid = json.loads(json.dumps(valid))
                invalid["maintenance"].update(mutation)
                config.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(PlanError):
                    load_config(root, config)

    def test_authenticated_remote_manifest_is_strict_data_only_configuration(
        self,
    ) -> None:
        """Pinned HTTPS bytes are accepted, while unpinned or execution-bearing data is rejected."""
        payload = json.dumps(
            {
                "schema_version": 1,
                "version": "4.1.0-snapshot.2",
                "files": {"logicytics/cli/commands.py": "a" * 64},
            }
        ).encode("utf-8")
        settings = MaintenanceSettings(
            remote_manifest_url="https://example.invalid/project.manifest.json",
            remote_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = payload
        with patch.object(
            maintenance.urllib.request,
            "urlopen",
            return_value=response,
        ):
            manifest = fetch_remote_manifest(settings)

        if manifest is None:
            self.fail("expected the configured remote manifest to be fetched")
        self.assertEqual("4.1.0-snapshot.2", manifest.version)

        tampered = MaintenanceSettings(
            remote_manifest_url=settings.remote_manifest_url,
            remote_manifest_sha256="0" * 64,
        )
        with (
            patch.object(
                maintenance.urllib.request,
                "urlopen",
                return_value=response,
            ),
            self.assertRaisesRegex(ValueError, "does not match"),
        ):
            fetch_remote_manifest(tampered)

        execution_payload = json.dumps(
            {
                "schema_version": 1,
                "version": "4.1.0",
                "files": {},
                "collectors": ["remote.code"],
            }
        ).encode("utf-8")

        execution_response = MagicMock()
        execution_response.__enter__.return_value.read.return_value = execution_payload

        execution_settings = MaintenanceSettings(
            remote_manifest_url=settings.remote_manifest_url,
            remote_manifest_sha256=hashlib.sha256(execution_payload).hexdigest(),
        )

        with (
            patch.object(
                maintenance.urllib.request,
                "urlopen",
                return_value=execution_response,
            ),
            self.assertRaisesRegex(ValueError, "only schema_version"),
        ):
            fetch_remote_manifest(execution_settings)

    def test_integrity_manifest_comparison_and_snapshot_version_ordering(self) -> None:
        """Developer integrity reports file states and compare snapshots semantically."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = MaintenanceSettings()
            (root / "same.txt").write_text("same", encoding="utf-8")
            (root / "changed.txt").write_text("before", encoding="utf-8")
            (root / "removed.txt").write_text("remove", encoding="utf-8")
            manifest = build_manifest(root, settings, "4.1.0-snapshot.2")
            write_local_manifest(root, settings, manifest)
            (root / "changed.txt").write_text("after", encoding="utf-8")
            (root / "removed.txt").unlink()
            (root / "added.txt").write_text("add", encoding="utf-8")
            comparison = compare_files(root, settings, manifest)
            self.assertEqual(["removed.txt"], comparison["missing"])
            self.assertEqual(["changed.txt"], comparison["modified"])
            self.assertEqual(["added.txt"], comparison["extra"])
            self.assertEqual(["same.txt"], comparison["unchanged"])
            self.assertEqual("behind", compare_versions("4.1.0-snapshot.2", "4.1.0"))
            self.assertEqual("behind", compare_versions("4.1.0-snapshot.2", "4.1.0-snapshot.10"))
            cache = root / ".pytest_cache" / "ignored.txt"
            cache.parent.mkdir()
            cache.write_text("ignored", encoding="utf-8")
            tool = root / "tools" / "Sysinternals" / "ignored.exe"
            tool.parent.mkdir(parents=True)
            tool.write_bytes(b"ignored")
            names = {path.relative_to(root).as_posix() for path in project_files(root, settings)}
            self.assertNotIn(".pytest_cache/ignored.txt", names)
            self.assertNotIn("tools/Sysinternals/ignored.exe", names)

    def test_dev_writes_explicit_manifest_and_debug_persists_diagnostics(self) -> None:
        """Side actions remain separate from collection and write only their dedicated artifacts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = {
                "git_available": True,
                "git_version": "git version 2.0",
                "is_repository": True,
                "origin_configured": True,
                "remote_reachable": True,
                "repository_returncode": 0,
                "origin_returncode": 0,
                "reachability_returncode": 0,
            }
            (root / "pyproject.toml").write_text(
                '[project]\nname = "fixture"\nversion = "4.0.0"\n',
                encoding="utf-8",
            )
            (root / "logicytics.yaml").write_text(
                "schema_version: 4\nmaintenance:\n  sysinternals_enabled: false\n",
                encoding="utf-8",
            )
            with (
                patch.object(CLI, "project_root", return_value=root),
                patch.object(CLI, "repository_status", return_value=repository),
                patch(
                    "sys.stdout",
                    new_callable=io.StringIO,
                ),
            ):
                self.assertEqual(
                    0,
                    main(["dev", "--write-manifest", "--next-version", "4.1.0"]),
                )

            manifest_path = root / "project.manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("4.1.0", manifest_payload["version"])

            with (
                patch.object(CLI, "project_root", return_value=root),
                patch(
                    "sys.stdout",
                    new_callable=io.StringIO,
                ),
            ):
                self.assertEqual(0, main(["debug"]))
            debug_path = root / "output" / "logs" / "debug" / "debug.json"
            debug = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertEqual("4.0.0", debug["maintenance"]["local_version"])
            self.assertIn(
                debug["maintenance"]["python_support"]["status"],
                {"recommended", "supported", "incompatible"},
            )
            self.assertEqual("disabled", debug["sysinternals"]["status"])

    def test_dev_writes_manifest_without_mutating_legacy_ini(self) -> None:
        """The v4 developer action writes its JSON manifest and leaves legacy files untouched."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = {
                "git_available": True,
                "git_version": "git version 2.0",
                "is_repository": True,
                "origin_configured": True,
                "remote_reachable": True,
                "repository_returncode": 0,
                "origin_returncode": 0,
                "reachability_returncode": 0,
            }
            (root / "pyproject.toml").write_text(
                '[project]\nname = "fixture"\nversion = "4.0.0"\n',
                encoding="utf-8",
            )
            collector = root / "core" / "system" / "example.py"
            collector.parent.mkdir(parents=True)
            collector.write_text('"""Example collector module."""\n', encoding="utf-8")
            code = root / "CODE"
            code.mkdir()
            legacy = code / "config.ini"
            legacy.write_text(
                '# preserve this comment\n[System Settings]\nversion = 3.6.0\nfiles = "old.py"\n\n[Unrelated]\nvalue = untouched\n',
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch.object(CLI, "project_root", return_value=root),
                patch.object(CLI, "repository_status", return_value=repository),
                patch(
                    "sys.stderr",
                    output,
                ),
            ):
                self.assertEqual(
                    0,
                    main(["dev", "--write-manifest", "--next-version", "4.1.0"]),
                )
            rendered = output.getvalue()
            self.assertIn("Development checks", rendered)
            self.assertIn("Manifest written: yes", rendered)
            self.assertNotIn("{", rendered)
            development_path = root / "output" / "logs" / "debug" / "development.json"
            payload = json.loads(development_path.read_text(encoding="utf-8"))
            self.assertEqual(
                (root / "project.manifest.json").resolve(),
                Path(payload["manifest_written"]).resolve(),
            )
            self.assertEqual([], payload["checks"]["misplaced_python"])
            updated = legacy.read_text(encoding="utf-8")
            self.assertIn("# preserve this comment", updated)
            self.assertIn("version = 3.6.0", updated)
            self.assertIn('files = "old.py"', updated)
            self.assertIn("value = untouched", updated)
            self.assertTrue((root / "project.manifest.json").exists())

    def test_sysinternals_archive_lifecycle_honors_yaml_opt_out_and_extracts_safely(self) -> None:
        """The local bundle must honor YAML opt-out and extract only within its target directory."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            disabled = MaintenanceSettings(sysinternals_enabled=False)
            self.assertEqual("disabled", ensure_sysinternals(root, disabled).status)
            archive_path = root / "tools" / "SysinternalsSuite.zip"
            archive_path.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("PsInfo.exe", "fixture")
            state = ensure_sysinternals(root)
            self.assertEqual("extracted", state.status)
            self.assertTrue((state.extraction_directory / "PsInfo.exe").is_file())


if __name__ == "__main__":
    unittest.main()
