from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from fixtures.collectors import COLLECTOR, plugin_collector_source
from logicytics import (
    load_configuration,
    open_artifact,
    plan_run,
    query_run,
    read_artifact,
    run_collection,
)
from logicytics.cli import CLI, main
from logicytics.contracts import (
    RunRequest,
    RunStatus,
)
from logicytics.errors import ArtifactError, PlanError, PreflightError


class PublicApiTests(unittest.TestCase):
    """Public planning, execution, query, and artifact API behavior."""

    def test_public_package_import_does_not_start_runtime_or_create_files(self) -> None:
        """Importing contracts exposes lazy application services without host or disk side effects."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_root = Path(__file__).resolve().parents[1]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(project_root)
            probe = (
                "import pathlib, sys; "
                "before = tuple(pathlib.Path.cwd().iterdir()); "
                "import logicytics; "
                "assert 'logicytics.runtime' not in sys.modules; "
                "assert {'load_configuration', 'plan_run', 'run_collection', 'query_run', 'read_artifact'}"
                ".issubset(logicytics.__all__); "
                "assert tuple(pathlib.Path.cwd().iterdir()) == before"
            )
            result = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual([], list(root.iterdir()))

    def test_public_planning_is_read_only_bounded_and_keeps_plugins_opt_in(self) -> None:
        """The application planner validates configuration and extensions without starting work."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            plugin_path = root / "plugins" / "example_plugin.py"
            plugin_path.parent.mkdir()
            plugin_path.write_text(
                plugin_collector_source()
                .replace("SystemInfoCollector", "ExamplePluginCollector")
                .replace("core.system.system_info", "plugin.example_plugin"),
                encoding="utf-8",
            )
            (root / "settings.json").write_text(
                '{"schema_version":4,"runtime":{"default_max_workers":1,"maximum_workers":2}}',
                encoding="utf-8",
            )
            configuration = load_configuration(root, "settings.json")
            standard = plan_run(root, RunRequest(max_workers=1), configuration=configuration)
            selected = plan_run(
                root,
                RunRequest(max_workers=1, include=("plugin.example_plugin",)),
                configuration=configuration,
            )

            self.assertEqual(["core.system.system_info"], [item.metadata.id for item in standard.collectors])
            self.assertEqual(
                ["core.system.system_info", "plugin.example_plugin"],
                [item.metadata.id for item in selected.collectors],
            )
            with self.assertRaisesRegex(PlanError, "maximum_workers"):
                plan_run(root, RunRequest(max_workers=3), configuration=configuration)
            with self.assertRaisesRegex(PlanError, "either configuration or config_path"):
                plan_run(root, RunRequest(max_workers=1), configuration=configuration, config_path="settings.json")
            plugin_path.write_text('"""Invalid optional plugin."""\n', encoding="utf-8")
            quarantined = plan_run(root, RunRequest(max_workers=1), configuration=configuration)
            self.assertEqual(["core.system.system_info"], [item.metadata.id for item in quarantined.collectors])
            with self.assertRaises(PreflightError):
                plan_run(
                    root,
                    RunRequest(max_workers=1, include=("plugin.example_plugin",)),
                    configuration=configuration,
                )
            collector_path.write_text('"""Invalid shipped core collector."""\n', encoding="utf-8")
            with self.assertRaises(PreflightError):
                plan_run(root, RunRequest(max_workers=1), configuration=configuration)
            self.assertFalse(configuration.runtime.output_root.exists())

    def test_public_api_runs_queries_and_reads_verified_registered_evidence(self) -> None:
        """The complete public lifecycle retains isolation and immutable manifest-backed status."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            configuration = load_configuration(root)
            request = RunRequest(max_workers=1, acknowledge_authorization=True)
            outcome = run_collection(root, request, configuration=configuration)
            snapshot = query_run(root, outcome.manifest.run_id, configuration=configuration)

            self.assertEqual(RunStatus.SUCCEEDED, snapshot.status, outcome.manifest.package)
            self.assertEqual(outcome.run_directory, snapshot.run_directory)
            self.assertEqual(outcome.manifest_path, snapshot.manifest_path)
            self.assertEqual(1,
                             json.loads(outcome.manifest_path.read_text(encoding="utf-8"))["manifest_schema_version"])
            self.assertEqual(["core.system.system_info"], [item.collector_id for item in snapshot.collectors])
            self.assertEqual(["succeeded"], [item.status for item in snapshot.collectors])
            collector = snapshot.collectors[0]
            self.assertIsNotNone(collector.started_at)
            self.assertIsNotNone(collector.finished_at)
            duration_seconds = collector.duration_seconds
            if duration_seconds is None:
                self.fail("completed collector must expose duration_seconds")
            self.assertGreaterEqual(duration_seconds, 0)
            self.assertEqual("test artifact created", collector.summary)
            self.assertEqual((), collector.errors)
            self.assertIsNone(collector.failure)
            self.assertEqual(1, len(snapshot.artifacts))
            self.assertEqual(
                b"ok" + os.linesep.encode("ascii"),
                read_artifact(root, snapshot.run_id, snapshot.artifacts[0].id, configuration=configuration),
            )
            with self.assertRaises(FrozenInstanceError):
                snapshot.status = RunStatus.FAILED
            with self.assertRaises(FrozenInstanceError):
                collector.status = "failed"

    def test_public_run_snapshot_and_cli_expose_verified_collector_failure_and_duration(
            self,
    ) -> None:
        """Callers can inspect redacted lifecycle timing and actionable failure details without parsing manifests."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()

            collector_path.write_text(
                COLLECTOR.replace(
                    '        output = context.workspace / "system.txt"',
                    '        raise RuntimeError("fixture lifecycle failure")\n'
                    '        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )

            outcome = run_collection(
                root,
                RunRequest(
                    max_workers=1,
                    acknowledge_authorization=True,
                ),
            )

            snapshot = query_run(root, outcome.manifest.run_id)
            collector = snapshot.collectors[0]

            self.assertEqual("failed", collector.status)
            duration_seconds = collector.duration_seconds
            if duration_seconds is None:
                self.fail("failed collector must expose duration_seconds")

            self.assertGreaterEqual(duration_seconds, 0)
            self.assertIn("worker crashed", collector.summary)
            self.assertTrue(
                any(
                    "fixture lifecycle failure" in error
                    for error in collector.errors
                )
            )
            failure = collector.failure
            if failure is None:
                self.fail("failed collector must expose structured failure details")
            self.assertEqual("core.system.system_info", failure.collector_id)
            self.assertEqual("collect", failure.operation)
            self.assertIn("fixture lifecycle failure", failure.platform_error)
            self.assertTrue(failure.remediation)
            self.assertTrue(failure.retry_safe)

            collector_path.write_text(
                COLLECTOR,
                encoding="utf-8",
            )

            output = io.StringIO()

            with patch.object(
                    CLI,
                    "project_root",
                    return_value=root,
            ), patch(
                "sys.stdout",
                output,
            ), patch(
                "builtins.input",
                return_value="",
            ) as final_prompt:
                exit_code = main(
                    [
                        "run",
                        "--default",
                        "--interactive",
                        "--acknowledge-authorization",
                    ]
                )

            self.assertEqual(0, exit_code, output.getvalue())
            final_prompt.assert_called_once_with("Press Enter to exit...")
            self.assertIn("Collectors:", output.getvalue())
            self.assertIn(
                "core.system.system_info status=succeeded duration_seconds=",
                output.getvalue(),
            )
            self.assertIn(
                "summary=test artifact created",
                output.getvalue(),
            )

    def test_public_run_queries_reject_traversal_forged_identity_and_manifest_links(self) -> None:
        """Persisted status lookup cannot leave its configured run or trust a forged manifest."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            outcome = run_collection(root, RunRequest(max_workers=1, acknowledge_authorization=True))
            original = outcome.manifest_path.read_text(encoding="utf-8")

            for invalid in ("../outside", "run-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "run-short"):
                with self.subTest(run_id=invalid):
                    with self.assertRaisesRegex(PlanError, "canonical run identifier"):
                        query_run(root, invalid)
            payload = json.loads(original)
            payload["run_id"] = "run-" + "0" * 32
            outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "identity"):
                query_run(root, outcome.manifest.run_id)
            invalid_manifests = (
                ("manifest_schema_version", 2, "unsupported run manifest schema_version"),
                ("manifest_schema_version", True, "unsupported run manifest schema_version"),
                ("status", [], "unsupported run status"),
                ("resolved_plan", ["core.system.forged"], "resolved plan"),
                ("total_artifact_bytes", 999, "total_artifact_bytes"),
                ("requested_at", "not-a-timestamp", "requested_at"),
                ("finished_at", None, "finished_at"),
            )
            for field, value, message in invalid_manifests:
                with self.subTest(field=field):
                    payload = json.loads(original)
                    payload[field] = value
                    outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        query_run(root, outcome.manifest.run_id)
            payload = json.loads(original)
            payload.pop("manifest_schema_version")
            outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "unsupported run manifest schema_version"):
                query_run(root, outcome.manifest.run_id)
            invalid_collector_fields = (
                ("started_at", "not-a-timestamp", "collector started_at"),
                ("finished_at", None, "require finished_at"),
                ("duration_seconds", -1, "duration_seconds"),
                ("summary", "", "summary"),
                ("errors", [""], "errors"),
                (
                    "failure",
                    {
                        "collector_id": "core.system.system_info",
                        "operation": "collect",
                        "platform_error": "forged",
                        "remediation": "forged",
                        "retry_safe": True,
                    },
                    "must match failed status",
                ),
            )
            for field, value, message in invalid_collector_fields:
                with self.subTest(collector_field=field):
                    payload = json.loads(original)
                    payload["collectors"][0][field] = value
                    outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(PlanError, message):
                        query_run(root, outcome.manifest.run_id)
            payload = json.loads(original)
            payload["collectors"][0]["id"] = "../outside"
            payload["resolved_plan"] = ["../outside"]
            outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "collector record"):
                query_run(root, outcome.manifest.run_id)
            outcome.manifest_path.write_text(
                '{"run_id":"' + outcome.manifest.run_id + '","run_id":"duplicate"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PlanError, "duplicate run manifest key"):
                query_run(root, outcome.manifest.run_id)
            outcome.manifest_path.write_text(original, encoding="utf-8")
            outside = root / "outside-manifest.json"
            outside.write_text(original, encoding="utf-8")
            outcome.manifest_path.unlink()
            outcome.manifest_path.symlink_to(outside)
            with self.assertRaisesRegex(PlanError, "escapes its run-owned directory"):
                query_run(root, outcome.manifest.run_id)

    def test_public_artifact_reads_reject_forgery_tampering_escapes_and_unbounded_access(self) -> None:
        """Only exact, bounded, manifest-cataloged bytes from the producing collector may be read."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            outcome = run_collection(root, RunRequest(max_workers=1, acknowledge_authorization=True))
            artifact = outcome.manifest.artifact_list()[0]
            stored = outcome.run_directory / "artifacts" / artifact.relative_path
            original = stored.read_bytes()
            manifest_source = outcome.manifest_path.read_text(encoding="utf-8")

            with patch("logicytics.api.os.startfile") as startfile:
                self.assertEqual(
                    stored,
                    open_artifact(root, outcome.manifest.run_id, artifact.id),
                )
                startfile.assert_called_once_with(stored)

            with self.assertRaisesRegex(ArtifactError, "canonical"):
                read_artifact(root, outcome.manifest.run_id, "../outside")
            with self.assertRaisesRegex(ArtifactError, "not registered"):
                read_artifact(root, outcome.manifest.run_id, "artifact." + "0" * 32)
            for invalid_limit in (True, 0, 67_108_865):
                with self.subTest(maximum_bytes=invalid_limit):
                    with self.assertRaisesRegex(ArtifactError, "maximum_bytes"):
                        read_artifact(root, outcome.manifest.run_id, artifact.id, maximum_bytes=invalid_limit)
            with self.assertRaisesRegex(ArtifactError, "bounded read limit"):
                read_artifact(root, outcome.manifest.run_id, artifact.id, maximum_bytes=1)
            stored.write_bytes(b"bad")
            with self.assertRaisesRegex(ArtifactError, "SHA-256 verification"):
                read_artifact(root, outcome.manifest.run_id, artifact.id)
            with self.assertRaisesRegex(ArtifactError, "size"):
                open_artifact(root, outcome.manifest.run_id, artifact.id)
            stored.write_bytes(original)
            payload = json.loads(manifest_source)
            payload["artifact_catalog"][0]["producer_status"] = "failed"
            outcome.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PlanError, "artifact catalog"):
                read_artifact(root, outcome.manifest.run_id, artifact.id)
            outcome.manifest_path.write_text(manifest_source, encoding="utf-8")
            outside = root / "outside-evidence.txt"
            outside.write_bytes(original)
            stored.unlink()
            stored.symlink_to(outside)
            with self.assertRaisesRegex(ArtifactError, "collector-owned store"):
                read_artifact(root, outcome.manifest.run_id, artifact.id)
            stored.unlink()
            owner = stored.parent
            owner.rmdir()
            redirected_owner = root / "redirected-owner"
            redirected_owner.mkdir()
            (redirected_owner / stored.name).write_bytes(original)
            owner.symlink_to(redirected_owner, target_is_directory=True)
            with self.assertRaisesRegex(ArtifactError, "collector-owned store"):
                read_artifact(root, outcome.manifest.run_id, artifact.id)


if __name__ == "__main__":
    unittest.main()
