from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast
from unittest.mock import patch

from fixtures.collectors import COLLECTOR, delayed_collector_source
from logicytics import packaging
from logicytics.cli import cli_methods
from logicytics.configuration import (
    default_config,
    load_config,
)
from logicytics.contracts import (
    RunRequest,
)
from logicytics.discovery import preflight
from logicytics.manifest import write_manifest
from logicytics.packaging import package_run
from logicytics.planner import build_plan
from logicytics.runtime import RunSupervisor


class PackagingTests(unittest.TestCase):
    """Package construction, verification, redaction, and publication behavior."""

    def test_preflight_plan_run_and_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            plan = build_plan(
                report,
                RunRequest(profile="standard", max_workers=1, acknowledge_authorization=True),
            )
            configuration = default_config(root)
            persisted_states: list[str] = []

            def capture_manifest_state(path, manifest):
                persisted_states.append(manifest.status.value)
                write_manifest(path, manifest)

            with patch("logicytics.runtime.write_manifest", side_effect=capture_manifest_state):
                outcome = RunSupervisor(root, configuration).run(plan)
            self.assertEqual("planned", persisted_states[0])
            self.assertEqual("running", persisted_states[1])
            self.assertEqual("succeeded", persisted_states[-1])
            self.assertEqual("succeeded", outcome.manifest.status.value)
            self.assertEqual(("core.system.system_info",), outcome.manifest.resolved_plan)
            self.assertEqual(plan.fingerprint, outcome.manifest.plan_fingerprint)
            self.assertEqual(64, len(plan.fingerprint))
            self.assertFalse(outcome.manifest.cancellation_requested)
            self.assertEqual([], outcome.manifest.errors)
            self.assertEqual([], outcome.manifest.skipped_collectors)
            self.assertEqual(configuration.runtime.output_root, outcome.run_directory.parent)
            self.assertEqual(root / "output" / "data", outcome.run_directory.parent)
            self.assertTrue(outcome.run_directory.is_absolute())
            self.assertTrue((outcome.run_directory / "artifacts" / "core_system_system_info").is_dir())
            self.assertFalse((root / "system.txt").exists())
            self.assertEqual(1, len(outcome.manifest.artifact_list()))
            self.assertEqual("run", outcome.manifest.action)
            self.assertEqual("4.0", outcome.manifest.engine_version)
            self.assertIn("user", outcome.manifest.host)
            self.assertIn("is_administrator", outcome.manifest.host)
            self.assertIsNotNone(outcome.manifest.collectors[0].heartbeat_at)
            self.assertIsNotNone(outcome.manifest.collectors[0].last_progress_at)
            self.assertFalse((outcome.run_directory / "collectors" / "core_system_system_info" / "tmp").exists())
            package = outcome.manifest.package
            assert package is not None
            self.assertTrue(Path(package["path"]).is_file())
            self.assertTrue(Path(package["sha256_path"]).is_file())
            self.assertEqual(outcome.run_directory / "packages", Path(package["path"]).parent)
            self.assertEqual(outcome.run_directory / "hashes", Path(package["sha256_path"]).parent)
            self.assertTrue((outcome.run_directory / "logs" / "engine.jsonl").is_file())
            self.assertFalse((root / "ACCESS").exists())
            self.assertFalse((root / "output" / "RUNS").exists())
            self.assertFalse((root / "output" / "PACKAGES").exists())
            package_path, hash_path = package_run(outcome)
            self.assertTrue(package_path.is_file())
            self.assertTrue(hash_path.is_file())
            requested_at = datetime.fromisoformat(outcome.manifest.requested_at)
            timestamp = requested_at.strftime("%Y%m%dT%H%M%S.%fZ")
            self.assertEqual(f"run-{timestamp}-{outcome.manifest.run_id}.zip", package_path.name)
            self.assertEqual(f"{package_path.name}.sha256", hash_path.name)
            package_digest, sidecar_name = hash_path.read_text(encoding="ascii").split()
            self.assertEqual(package_path.name, sidecar_name)
            self.assertEqual(hashlib.sha256(package_path.read_bytes()).hexdigest(), package_digest)
            self.assertEqual(package_digest, package["sha256"])
            with zipfile.ZipFile(package_path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("metadata/manifest.json", archive.namelist())
                self.assertIn("reports/summary.txt", archive.namelist())
                self.assertIn("hashes/artifacts.sha256", archive.namelist())
                self.assertNotIn("logs/performance.json", archive.namelist())
                self.assertEqual(1, len([name for name in archive.namelist() if name.startswith("evidence/")]))
                artifact = outcome.manifest.artifact_list()[0]
                archived_bytes = archive.read(
                    f"evidence/{artifact.evidence_kind.value}/{artifact.relative_path}"
                )
                self.assertEqual(artifact.size_bytes, len(archived_bytes))
                self.assertEqual(artifact.sha256, hashlib.sha256(archived_bytes).hexdigest())
                self.assertEqual("system", artifact.source_category)
                datetime.fromisoformat(artifact.collected_at)
                self.assertEqual(("copied into run artifact store",), artifact.transformations)
                packaged_manifest = json.loads(archive.read("metadata/manifest.json"))
                self.assertEqual(["core.system.system_info"], packaged_manifest["resolved_plan"])
                self.assertEqual(plan.fingerprint, packaged_manifest["plan_fingerprint"])
                self.assertFalse(packaged_manifest["cancellation_requested"])
                self.assertEqual([], packaged_manifest["errors"])
                self.assertEqual([], packaged_manifest["skipped_collectors"])
                packaged_artifact = packaged_manifest["collectors"][0]["artifacts"][0]
                catalog_artifact = packaged_manifest["artifact_catalog"][0]
                self.assertEqual(artifact.source_category, packaged_artifact["source_category"])
                self.assertEqual(artifact.collected_at, packaged_artifact["collected_at"])
                self.assertEqual(list(artifact.transformations), packaged_artifact["transformations"])
                self.assertEqual(artifact.id, catalog_artifact["id"])
                self.assertEqual("system.txt", catalog_artifact["name"])
                self.assertEqual("text/plain", catalog_artifact["media_type"])
                self.assertEqual("registered", catalog_artifact["status"])
                self.assertEqual("succeeded", catalog_artifact["producer_status"])
                self.assertEqual("core.system.system_info", catalog_artifact["collector_id"])
                record = outcome.manifest.collectors[0]
                summary = archive.read("reports/summary.txt").decode("utf-8")
                self.assertIn(f"Status: {record.status}", summary)
                self.assertIn("Cancellation requested: false", summary)
                self.assertIn("Resolved collectors: 1", summary)
                self.assertIn(f"Plan fingerprint: {plan.fingerprint}", summary)
                self.assertIsNotNone(record.started_at)
                self.assertIsNotNone(record.finished_at)

                started_at = record.started_at
                finished_at = record.finished_at

                assert started_at is not None
                assert finished_at is not None

                self.assertIn(f"Started: {started_at}", summary)
                self.assertIn(f"Finished: {finished_at}", summary)

            repeated = RunSupervisor(root, configuration).run(plan)
            self.assertNotEqual(outcome.manifest.run_id, repeated.manifest.run_id)
            self.assertNotEqual(outcome.run_directory, repeated.run_directory)

    def test_package_separates_typed_evidence_reports_logs_hashes_and_metadata(self) -> None:
        """The versioned package layout is manifest-led and has no ambiguous root entries."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            source = COLLECTOR.replace(
                "CollectorResult, CoreCollector, Specialty",
                "CollectorResult, CoreCollector, EvidenceKind, Specialty",
            ).replace(
                'output_media_types=("text/plain",)',
                'output_media_types=("text/plain", "application/octet-stream")',
            ).replace(
                '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                '        raw = context.workspace / "source.bin"\n'
                '        raw.write_bytes(b"raw evidence")\n'
                '        raw_artifact = context.artifacts.register_file(\n'
                '            raw, evidence_kind=EvidenceKind.RAW,\n'
                '        )\n'
                '        return CollectorResult.succeeded(\n'
                '            "typed artifacts created", (artifact, raw_artifact),\n'
                '        )',
            )
            collector_path.write_text(source, encoding="utf-8")
            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            outcome = RunSupervisor(root, default_config(root)).run(
                build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            )
            package = outcome.manifest.package
            assert package is not None
            self.assertEqual("succeeded", outcome.manifest.status.value, package)

            package_path = Path(package["path"])
            with zipfile.ZipFile(package_path) as archive:
                names = archive.namelist()
                self.assertEqual(
                    {"evidence", "hashes", "logs", "metadata", "reports"},
                    {name.split("/", 1)[0] for name in names},
                )
                self.assertNotIn("manifest.json", names)
                self.assertNotIn("summary.txt", names)
                self.assertFalse(any(name.startswith("artifacts/") for name in names))
                artifacts = outcome.manifest.artifact_list()
                expected_catalog = "".join(
                    f"{artifact.sha256}  evidence/{artifact.evidence_kind.value}/{artifact.relative_path}\n"
                    for artifact in sorted(
                        artifacts,
                        key=lambda item: f"evidence/{item.evidence_kind.value}/{item.relative_path}",
                    )
                )
                self.assertEqual(
                    expected_catalog,
                    archive.read("hashes/artifacts.sha256").decode("ascii"),
                )
                for artifact in artifacts:
                    archive_name = f"evidence/{artifact.evidence_kind.value}/{artifact.relative_path}"
                    self.assertEqual(artifact.sha256, hashlib.sha256(archive.read(archive_name)).hexdigest())
                packaged = json.loads(archive.read("metadata/manifest.json"))
            self.assertEqual("1.0", packaged["package_layout_version"])
            self.assertEqual(1, packaged["manifest_schema_version"])
            self.assertEqual("evidence/raw/", packaged["package_sections"]["raw_evidence"])
            self.assertEqual("evidence/derived/", packaged["package_sections"]["derived_reports"])
            self.assertEqual(
                {"raw", "derived"},
                {item["evidence_kind"] for item in packaged["artifact_catalog"]},
            )
            outcome.manifest.manifest_schema_version = 2
            with self.assertRaisesRegex(ValueError, "unsupported run manifest schema_version"):
                outcome.manifest.to_dict()

    def test_configured_output_root_colocates_packages_without_touching_legacy_evidence(self) -> None:
        """Custom roots own runs and ZIPs; old ACCESS evidence is neither moved nor deleted."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            legacy_evidence = root / "ACCESS" / "RUNS" / "legacy-evidence.txt"
            legacy_evidence.parent.mkdir(parents=True)
            legacy_evidence.write_text("preserve existing evidence", encoding="utf-8")
            (root / "logicytics.json").write_text(
                '{"schema_version":4,"runtime":{"output_root":"custom/evidence"}}',
                encoding="utf-8",
            )
            configuration = load_config(root)
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, configuration).run(plan)
            package = outcome.manifest.package
            assert package is not None
            expected_root = root / "custom" / "evidence"
            package_path = Path(package["path"])
            hash_path = Path(package["sha256_path"])

            self.assertEqual(expected_root, outcome.run_directory.parent)
            self.assertEqual(outcome.run_directory / "packages", package_path.parent)
            self.assertEqual(outcome.run_directory / "hashes", hash_path.parent)
            self.assertEqual("preserve existing evidence", legacy_evidence.read_text(encoding="utf-8"))
            self.assertFalse((root / "custom" / "PACKAGES").exists())
            self.assertEqual(
                hashlib.sha256(package_path.read_bytes()).hexdigest(),
                hash_path.read_text(encoding="ascii").split()[0],
            )

    def test_partial_collector_evidence_is_packaged_and_clearly_labeled(self) -> None:
        """Partial terminal outcomes preserve evidence without masquerading as success."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    '        return CollectorResult.partial(\n'
                    '            "only one evidence source was available",\n'
                    '            (artifact,),\n'
                    '            errors=("secondary evidence source was unavailable",),\n'
                    '        )',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            outcome = RunSupervisor(root, default_config(root)).run(
                build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            )
            record = outcome.manifest.collectors[0]

            self.assertEqual("partial", outcome.manifest.status.value)
            self.assertEqual("partial", record.status)
            self.assertEqual(1, len(record.artifacts))
            self.assertIn("secondary evidence source", outcome.manifest.errors[0]["message"])
            package = outcome.manifest.package
            assert package is not None
            package_path = Path(package["path"])
            self.assertTrue(package_path.is_file())
            with zipfile.ZipFile(package_path) as archive:
                packaged = json.loads(archive.read("metadata/manifest.json"))
                summary = archive.read("reports/summary.txt").decode("utf-8")
                self.assertEqual(
                    ["ok"],
                    archive.read("evidence/derived/core_system_system_info/system.txt").decode("utf-8").splitlines(),
                )
            self.assertEqual("partial", packaged["status"])
            self.assertEqual("partial", packaged["collectors"][0]["status"])
            self.assertIn("Status: partial", summary)
            self.assertIn("secondary evidence source was unavailable", summary)

    def test_selected_collector_rerun_preserves_original_run_and_separates_evidence(self) -> None:
        """Explicit reruns execute only selected original IDs and own a distinct evidence package."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            for filename in ("a_original", "z_selected"):
                (core_directory / f"{filename}.py").write_text(
                    delayed_collector_source(filename, 0.0),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            configuration = default_config(root)
            original = RunSupervisor(root, configuration).run(
                build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            )
            original_package_metadata = original.manifest.package
            assert original_package_metadata is not None
            original_manifest = original.manifest_path.read_bytes()
            original_package_path = Path(original_package_metadata["path"])
            original_package = original_package_path.read_bytes()
            selected_id = "core.system.z_selected"
            arguments = cli_methods.parser().parse_args(
                [
                    "run",
                    "--rerun-from",
                    str(original.run_directory),
                    "--include",
                    selected_id,
                    "--acknowledge-authorization",
                    "--sequential",
                ]
            )
            request = cli_methods.request(arguments, default_workers=4)
            self.assertEqual(original.manifest.run_id, request.rerun_from)
            rerun = RunSupervisor(root, configuration).run(build_plan(report, request))
            rerun_package_metadata = rerun.manifest.package
            assert rerun_package_metadata is not None
            rerun_package_path = Path(rerun_package_metadata["path"])

            self.assertEqual("succeeded", rerun.manifest.status.value)
            self.assertEqual("rerun", rerun.manifest.action)
            self.assertEqual(original.manifest.run_id, rerun.manifest.parent_run_id)
            self.assertEqual((selected_id,), rerun.manifest.resolved_plan)
            self.assertEqual([selected_id], [record.id for record in rerun.manifest.collectors])
            self.assertNotEqual(original.run_directory, rerun.run_directory)
            self.assertNotEqual(original_package_path, rerun_package_path)
            self.assertTrue(original_package_path.name.startswith("run-"))
            self.assertTrue(rerun_package_path.name.startswith("rerun-"))
            self.assertIn(rerun.manifest.run_id, rerun_package_path.name)
            self.assertEqual(original_manifest, original.manifest_path.read_bytes())
            self.assertEqual(original_package, original_package_path.read_bytes())
            with zipfile.ZipFile(rerun_package_path) as archive:
                packaged_manifest = json.loads(archive.read("metadata/manifest.json"))
                summary = archive.read("reports/summary.txt").decode("utf-8")
                artifact_names = [name for name in archive.namelist() if name.startswith("evidence/")]
            self.assertEqual(original.manifest.run_id, packaged_manifest["parent_run_id"])
            self.assertEqual("rerun", packaged_manifest["action"])
            self.assertEqual(["evidence/derived/core_system_z_selected/system.txt"], artifact_names)
            self.assertIn("Action: rerun", summary)
            self.assertIn(f"Parent run: {original.manifest.run_id}", summary)

    def test_sensitive_artifact_bytes_are_preserved_while_packaged_diagnostics_are_redacted(self) -> None:
        """Evidence retains intentional secrets, but no packaged diagnostics disclose them."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '        output = context.workspace / "system.txt"',
                    '        password = context.settings["password"]\n'
                    '        token = context.settings["nested"]["access_token"]\n'
                    '        output = context.workspace / "system.txt"',
                ).replace(
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")',
                    '        output.write_text(f"password={password} token={token}", encoding="utf-8")\n'
                    '        context.logger.event("info", f"password={password}", access_token=token)\n'
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")',
                ).replace(
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    '        return CollectorResult.succeeded(f"password={password}", (artifact,))',
                ),
                encoding="utf-8",
            )
            (root / "logicytics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "collectors": {
                            "core.system.system_info": {
                                "password": "evidence-password",
                                "nested": {"access_token": "evidence-token"},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            configuration = replace(
                default_config(root),
                collector_settings=cast(Any, {
                    "core.system.system_info": {
                        "password": "evidence-password",
                        "nested": {"access_token": "evidence-token"},
                    }
                }),
            )
            outcome = RunSupervisor(root, configuration).run(plan)
            self.assertEqual("succeeded", outcome.manifest.status.value)
            self.assertEqual("password=[REDACTED]", outcome.manifest.collectors[0].summary)
            package = outcome.manifest.package
            assert package is not None
            with zipfile.ZipFile(Path(package["path"])) as archive:
                artifact = outcome.manifest.artifact_list()[0]
                self.assertEqual(
                    "password=evidence-password token=evidence-token",
                    archive.read(f"evidence/{artifact.evidence_kind.value}/{artifact.relative_path}").decode("utf-8"),
                )
                diagnostics = "\n".join(
                    archive.read(name).decode("utf-8")
                    for name in archive.namelist()
                    if not name.startswith("evidence/")
                )
                self.assertNotIn("evidence-password", diagnostics)
                self.assertNotIn("evidence-token", diagnostics)
                self.assertIn("[REDACTED]", diagnostics)

    def test_worker_exception_secrets_are_redacted_from_manifest_and_package(self) -> None:
        """Collector crashes remain actionable without leaking inline credentials."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '        output = context.workspace / "system.txt"',
                    '        raise RuntimeError("password=crash-password token=crash-token")',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            self.assertEqual("failed", outcome.manifest.collectors[0].status)
            self.assertIn("RuntimeError", "\n".join(outcome.manifest.collectors[0].errors))
            self.assertEqual("core.system.system_info", outcome.manifest.errors[0]["collector_id"])
            self.assertNotIn("crash-password", outcome.manifest.errors[0]["message"])
            self.assertNotIn("crash-token", outcome.manifest.errors[0]["message"])
            failure = outcome.manifest.collectors[0].failure
            self.assertIsNotNone(failure)
            assert failure is not None
            platform_error = failure["platform_error"]
            assert isinstance(platform_error, str)

            self.assertNotIn("crash-password", platform_error)
            self.assertNotIn("crash-token", platform_error)
            self.assertIn("[REDACTED]", platform_error)
            package = outcome.manifest.package
            assert package is not None

            with zipfile.ZipFile(Path(package["path"])) as archive:
                packaged_record = json.loads(archive.read("metadata/manifest.json"))["collectors"][0]
                self.assertEqual(failure, packaged_record["failure"])
                diagnostics = "\n".join(
                    archive.read(name).decode("utf-8")
                    for name in archive.namelist()
                    if not name.startswith("evidence/")
                )
                self.assertNotIn("crash-password", diagnostics)
                self.assertNotIn("crash-token", diagnostics)
                self.assertIn("[REDACTED]", diagnostics)

    def test_performance_report_is_finalized_before_automatic_packaging(self) -> None:
        """Performance-mode timing evidence must be present in the automatic ZIP."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            plan = build_plan(
                preflight(root),
                RunRequest(
                    max_workers=1,
                    acknowledge_authorization=True,
                    performance_check=True,
                ),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            performance_path = outcome.run_directory / "logs" / "performance.json"
            self.assertTrue(performance_path.is_file())
            report = json.loads(performance_path.read_text(encoding="utf-8"))
            self.assertEqual(outcome.manifest.run_id, report["run_id"])
            self.assertEqual("core.system.system_info", report["collectors"][0]["id"])
            self.assertIsNotNone(report["collectors"][0]["duration_seconds"])
            package = outcome.manifest.package
            assert package is not None
            with zipfile.ZipFile(Path(package["path"])) as archive:
                packaged_report = json.loads(archive.read("logs/performance.json"))
            self.assertEqual(report, packaged_report)
            performance_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "performance report"):
                package_run(outcome)

    def test_collector_resource_progress_persists_in_manifest_summary_and_performance_report(self) -> None:
        """Incremental progress fields remain visible live and in every packaged diagnostic."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace("from pathlib import Path\n",
                                  "from pathlib import Path\nfrom time import sleep\n").replace(
                    '        output = context.workspace / "system.txt"',
                    '        context.report_progress("scan_started", scanned_files=3)\n'
                    '        sleep(0.2)\n'
                    '        context.report_progress(\n'
                    '            "scan_finished", scanned_files=9, copied_files=4, bytes_written=25,\n'
                    '            observation_count=6, event_count=11,\n'
                    '        )\n'
                    '        sleep(0.2)\n'
                    '        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )
            plan = build_plan(
                preflight(root),
                RunRequest(max_workers=1, acknowledge_authorization=True, performance_check=True),
            )
            snapshots: list[dict[str, Any]] = []
            from logicytics.manifest import write_manifest as original_write_manifest

            def capture_manifest(path: Path, manifest: Any) -> None:
                snapshots.append(json.loads(json.dumps(manifest.to_dict())))
                original_write_manifest(path, manifest)

            with patch("logicytics.runtime.write_manifest", side_effect=capture_manifest):
                outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]
            expected = {
                "files_scanned": 9,
                "files_copied": 4,
                "bytes_written": 25,
                "packets_observed": 6,
                "events_processed": 11,
            }
            for field, value in expected.items():
                self.assertEqual(value, record.progress[field])
            self.assertGreater(float(record.progress["elapsed_seconds"]), 0)
            self.assertTrue(
                any(
                    snapshot["collectors"][0]["status"] == "running"
                    and snapshot["collectors"][0]["progress"]["files_scanned"] >= 3
                    for snapshot in snapshots
                )
            )
            package = outcome.manifest.package
            assert package is not None
            with zipfile.ZipFile(Path(package["path"])) as archive:
                packaged_record = json.loads(archive.read("metadata/manifest.json"))["collectors"][0]
                report = json.loads(archive.read("logs/performance.json"))["collectors"][0]
                summary = archive.read("reports/summary.txt").decode("utf-8")
            self.assertEqual(record.progress, packaged_record["progress"])
            self.assertEqual(record.progress, report["progress"])
            self.assertEqual(record.peak_memory_bytes, report["peak_memory_bytes"])
            self.assertIn("Files scanned: 9", summary)
            self.assertIn("Files copied: 4", summary)
            self.assertIn("Bytes written: 25", summary)
            self.assertIn("Packets observed: 6", summary)
            self.assertIn("Events processed: 11", summary)

    def test_package_excludes_unregistered_files_and_rejects_tampered_artifacts(self) -> None:
        """Only manifest artifacts may enter a package, and their final bytes must match."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            report = preflight(root)
            plan = build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            unregistered = outcome.run_directory / "artifacts" / "unregistered.txt"
            unregistered.write_text("must not be packaged\n", encoding="utf-8")

            package_path, _ = package_run(outcome)
            with zipfile.ZipFile(package_path) as archive:
                self.assertNotIn("evidence/derived/unregistered.txt", archive.namelist())

            original_write = packaging._stream_archive_member

            def tampering_write(
                    archive: zipfile.ZipFile,
                    filename: Path,
                    arcname: str,
            ) -> None:
                if arcname.startswith("evidence/"):
                    archive.writestr(arcname, b"tampered artifact bytes")
                    return
                original_write(archive, filename, arcname)

            with patch.object(packaging, "_stream_archive_member", new=tampering_write):
                with self.assertRaisesRegex(ValueError, "packaged artifact verification failed"):
                    package_run(outcome)
            self.assertFalse(package_path.with_suffix(".zip.tmp").exists())

    def test_package_excludes_unregistered_collector_and_engine_jsonl_files(self) -> None:
        """Collector-created JSONL files cannot bypass the registered artifact contract."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            collector_workspace = outcome.run_directory / "collectors" / "core_system_system_info"
            (collector_workspace / "unregistered.jsonl").write_text("secret workspace data", encoding="utf-8")
            (outcome.run_directory / "logs" / "unregistered.jsonl").write_text(
                "secret engine data",
                encoding="utf-8",
            )

            package_path, _ = package_run(outcome)
            with zipfile.ZipFile(package_path) as archive:
                self.assertIn("logs/engine.jsonl", archive.namelist())
                self.assertIn("logs/collectors/core_system_system_info/events.jsonl", archive.namelist())
                self.assertNotIn("logs/unregistered.jsonl", archive.namelist())
                self.assertNotIn("collectors/core_system_system_info/unregistered.jsonl", archive.namelist())

    def test_package_sidecar_failure_restores_existing_verified_package(self) -> None:
        """A failed hash publication cannot replace or corrupt a previously verified ZIP."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            package = outcome.manifest.package
            assert package is not None
            package_path = Path(package["path"])
            hash_path = Path(package["sha256_path"])
            original_package = package_path.read_bytes()
            original_hash = hash_path.read_bytes()
            original_replace = os.replace

            def fail_sidecar(source: str | Path, destination: str | Path) -> None:
                if Path(destination) == hash_path and Path(source).suffix == ".tmp":
                    raise OSError("simulated sidecar publication failure")
                original_replace(source, destination)

            with patch("logicytics.packaging.os.replace", side_effect=fail_sidecar):
                with self.assertRaisesRegex(OSError, "sidecar publication failure"):
                    package_run(outcome)
            self.assertEqual(original_package, package_path.read_bytes())
            self.assertEqual(original_hash, hash_path.read_bytes())
            self.assertFalse(package_path.with_suffix(".zip.tmp").exists())
            self.assertFalse(package_path.with_suffix(".zip.backup").exists())
            self.assertFalse(hash_path.with_suffix(".sha256.tmp").exists())
            self.assertFalse(hash_path.with_suffix(".sha256.backup").exists())

    def test_package_rejects_manifest_artifact_path_escape_and_forged_collector_ownership(self) -> None:
        """A modified manifest cannot package outside files or another collector's evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            artifact = outcome.manifest.collectors[0].artifacts[0]
            original_path = artifact["relative_path"]
            artifact["relative_path"] = "../outside.jsonl"
            with self.assertRaisesRegex(ValueError, "escapes its collector-owned store"):
                package_run(outcome)
            artifact["relative_path"] = original_path
            artifact["collector_id"] = "core.system.another"
            with self.assertRaisesRegex(ValueError, "collector ownership"):
                package_run(outcome)

    def test_package_name_rejects_unsafe_action_run_id_and_naive_timestamps(self) -> None:
        """Manipulated manifest identity cannot escape the output root or obscure run timing."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            invalid = (
                ("action", "../outside", "action"),
                ("run_id", "../outside", "run_id"),
                ("requested_at", "2026-01-01T00:00:00", "timezone"),
                ("requested_at", "not a timestamp", "timestamp"),
            )
            for field, value, message in invalid:
                with self.subTest(field=field, value=value):
                    original = getattr(outcome.manifest, field)
                    setattr(outcome.manifest, field, value)
                    try:
                        with self.assertRaisesRegex(ValueError, message):
                            package_run(outcome)
                    finally:
                        setattr(outcome.manifest, field, original)
            self.assertFalse((root / "outside").exists())

    def test_package_preserves_nested_registered_artifact_directories(self) -> None:
        """Deep collector-owned evidence paths survive catalog publication and ZIP creation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '        output = context.workspace / "system.txt"',
                    '        output = context.workspace / "nested" / "reports" / "system.txt"\n'
                    '        output.parent.mkdir(parents=True)',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            expected = "core_system_system_info/nested/reports/system.txt"
            self.assertEqual(expected, outcome.manifest.artifact_list()[0].relative_path)
            package = outcome.manifest.package
            assert package is not None
            with zipfile.ZipFile(Path(package["path"])) as archive:
                self.assertIn(f"evidence/derived/{expected}", archive.namelist())
                self.assertEqual(
                    expected,
                    json.loads(archive.read("metadata/manifest.json"))["artifact_catalog"][0]["relative_path"],
                )

    def test_package_rejects_forged_names_status_and_duplicate_catalog_identifiers(self) -> None:
        """A package is refused whenever its finalized evidence catalog is inconsistent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    '        second = context.workspace / "second.txt"\n'
                    '        second.write_text("second", encoding="utf-8")\n'
                    '        extra = context.artifacts.register_file(second, media_type="text/plain")\n'
                    '        return CollectorResult.succeeded("test artifacts created", (artifact, extra))',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            first, second = outcome.manifest.collectors[0].artifacts
            original_name = first["name"]
            first["name"] = "forged.txt"
            with self.assertRaisesRegex(ValueError, "name does not match"):
                package_run(outcome)
            first["name"] = original_name
            first["status"] = "deleted"
            with self.assertRaisesRegex(ValueError, "status must be registered"):
                package_run(outcome)
            first["status"] = "registered"
            second["id"] = first["id"]
            with self.assertRaisesRegex(ValueError, "duplicate artifact id"):
                package_run(outcome)

    def test_package_rejects_collector_event_log_symlinks_outside_the_run(self) -> None:
        """A collector event-channel symlink must not smuggle external files into a ZIP."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            outside = root / "outside.jsonl"
            outside.write_text("outside credentials", encoding="utf-8")
            events = outcome.run_directory / "collectors" / "core_system_system_info" / "events.jsonl"
            events.unlink()
            events.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "diagnostic log escapes"):
                package_run(outcome)

    def test_package_rejects_manifest_artifact_symlinks_outside_collector_storage(self) -> None:
        """Matching bytes cannot make an externally redirected artifact safe to package."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(COLLECTOR, encoding="utf-8")
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            artifact = outcome.manifest.artifact_list()[0]
            stored = outcome.run_directory / "artifacts" / artifact.relative_path
            outside = root / "outside.txt"
            outside.write_bytes(stored.read_bytes())
            stored.unlink()
            stored.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "escapes its collector-owned store"):
                package_run(outcome)

    def test_package_streams_large_evidence_and_hashes_with_bounded_reads(self) -> None:
        """Large artifacts must enter ZIP/hash verification without whole-file reads."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")',
                    '        output.write_bytes(b"x" * (2 * 1024 * 1024 + 17))\n'
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            artifact = outcome.manifest.artifact_list()[0]
            source = outcome.run_directory / "artifacts" / artifact.relative_path
            package = outcome.manifest.package
            assert package is not None
            expected_package_path = Path(package["path"]).with_suffix(".zip.tmp")
            original_open = cast(Callable[..., Any], Path.open)
            read_sizes: list[tuple[Path, int]] = []

            class BoundedReader:
                def __init__(self, stream: Any, path: Path) -> None:
                    self.stream = stream
                    self.path = path

                def __enter__(self) -> "BoundedReader":
                    self.stream.__enter__()
                    return self

                def __exit__(self, *arguments: Any) -> Any:
                    return self.stream.__exit__(*arguments)

                def read(self, size: int = -1) -> bytes:
                    if not 0 < size <= 1024 * 1024:
                        raise AssertionError(f"unbounded evidence read requested: {size}")
                    read_sizes.append((self.path, size))
                    return self.stream.read(size)

            def guarded_open(path: Path, *arguments: Any, **options: Any) -> Any:
                stream = original_open(path, *arguments, **options)
                if path in {source, expected_package_path} and arguments and arguments[0] == "rb":
                    return BoundedReader(stream, path)
                return stream

            with patch.object(Path, "open", new=guarded_open):
                package_path, hash_path = package_run(outcome)
            self.assertGreaterEqual(len(read_sizes), 6)
            self.assertEqual({source, expected_package_path}, {path for path, _ in read_sizes})
            self.assertEqual(artifact.size_bytes, 2 * 1024 * 1024 + 17)
            self.assertTrue(package_path.is_file())
            self.assertTrue(hash_path.is_file())


if __name__ == "__main__":
    unittest.main()
