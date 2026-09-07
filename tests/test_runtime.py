from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from logicytics.module.configuration import (
    default_config,
)
from logicytics.contracts import (
    Capability,
    RunRequest,
)
from logicytics import (
    STILL_ACTIVE,
    close_handle,
    get_exit_code_process,
    open_process,
)
from logicytics.module.discovery import preflight
from logicytics.module.planner import build_plan
from logicytics.module.runtime import RunSupervisor
from tests.fixtures.collectors import COLLECTOR, delayed_collector_source


class RuntimeTests(unittest.TestCase):
    """Worker lifecycle, retries, failures, timeouts, cleanup, and cancellation."""

    def test_failed_dependency_skips_dependent_without_launching_it(self) -> None:
        """A failed prerequisite must contain failure and prevent dependent execution."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()

            dependency_id = "core.system.a_dependency"

            (core_directory / "a_dependency.py").write_text(
                delayed_collector_source(
                    "a_dependency",
                    0.0,
                    fail=True,
                ),
                encoding="utf-8",
            )

            (core_directory / "b_dependent.py").write_text(
                delayed_collector_source(
                    "b_dependent",
                    0.0,
                    dependencies=(dependency_id,),
                ),
                encoding="utf-8",
            )

            report = preflight(root)
            self.assertEqual((), report.invalid)

            plan = build_plan(
                report,
                RunRequest(
                    max_workers=2,
                    acknowledge_authorization=True,
                ),
            )

            outcome = RunSupervisor(
                root,
                default_config(root),
            ).run(plan)

            records = {
                record.id: record
                for record in outcome.manifest.collectors
            }

            self.assertEqual(
                "failed",
                records[dependency_id].status,
            )

            dependent = records["core.system.b_dependent"]

            self.assertEqual(
                "skipped",
                dependent.status,
            )
            self.assertIsNone(dependent.started_at)
            self.assertTrue(
                any(
                    dependency_id in error
                    for error in dependent.errors
                )
            )

            self.assertEqual(
                ["core.system.b_dependent"],
                outcome.manifest.skipped_collectors,
            )

            self.assertEqual(
                {
                    dependency_id,
                    "core.system.b_dependent",
                },
                {
                    error["collector_id"]
                    for error in outcome.manifest.errors
                },
            )

            package = outcome.manifest.package
            assert package is not None

            with zipfile.ZipFile(Path(package["path"])) as archive:
                packaged = json.loads(
                    archive.read("metadata/manifest.json")
                )
                summary = archive.read(
                    "reports/summary.txt"
                ).decode("utf-8")

            self.assertEqual(
                ["core.system.b_dependent"],
                packaged["skipped_collectors"],
            )
            self.assertIn(
                "Skipped collectors: 1",
                summary,
            )

    def test_failed_collector_is_manifested_and_never_reported_as_success(self) -> None:
        """An isolated collector crash must produce a durable failed run outcome."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    'output = context.workspace / "system.txt"\n'
                    '        output.write_text("ok\\n", encoding="utf-8")\n'
                    '        artifact = context.artifacts.register_file(output, media_type="text/plain")\n'
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    'raise RuntimeError("intentional test collector failure")',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            plan = build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            self.assertEqual("failed", outcome.manifest.status.value)
            record = outcome.manifest.collectors[0]
            self.assertEqual("failed", record.status)
            self.assertTrue(any("RuntimeError" in error for error in record.errors))
            self.assertIsNotNone(outcome.manifest.package)
            package = outcome.manifest.package
            self.assertIsNotNone(package)
            assert package is not None

            package_path = Path(package["path"])
            self.assertTrue(package_path.is_file())
            with zipfile.ZipFile(package_path) as archive:
                summary = archive.read("reports/summary.txt").decode("utf-8")
            self.assertIn("Status: failed", summary)
            self.assertIn("Reasons:", summary)
            self.assertIn("RuntimeError", summary)
            self.assertIsNotNone(record.started_at)
            self.assertIsNotNone(record.finished_at)

            assert record.started_at is not None
            assert record.finished_at is not None

            self.assertIn(f"Started: {record.started_at}", summary)
            self.assertIn(f"Finished: {record.finished_at}", summary)

    def test_transient_collector_failure_retries_in_a_new_isolated_worker(self) -> None:
        """An explicitly retryable collector may recover before any evidence is registered."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n'
                    '            maximum_retries=1,\n'
                    '            retry_delay_seconds=0.01,',
                ).replace(
                    '        output = context.workspace / "system.txt"',
                    '        marker = context.workspace / "attempt.marker"\n'
                    '        if not marker.exists():\n'
                    '            marker.write_text("first attempt failed", encoding="utf-8")\n'
                    '            raise RuntimeError("temporary collector failure")\n'
                    '        output = context.workspace / "system.txt"',
                ),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)

            record = outcome.manifest.collectors[0]
            self.assertEqual("succeeded", record.status, record.errors)
            self.assertIsNone(record.failure)
            self.assertEqual(2, record.attempt_count)
            self.assertEqual(1, len(record.retry_history))
            self.assertEqual("collect", record.retry_history[0]["failure"]["operation"])
            self.assertTrue(record.retry_history[0]["failure"]["retry_safe"])
            self.assertTrue(
                any("temporary collector failure" in error for error in record.retry_history[0]["errors"])
            )
            self.assertIsInstance(record.worker_pid, int)
            self.assertIsInstance(record.retry_history[0]["worker_pid"], int)
            self.assertNotEqual(record.worker_pid, record.retry_history[0]["worker_pid"])
            self.assertEqual("failed", record.retry_history[0]["termination_reason"])
            self.assertEqual("completed", record.termination_reason)
            package = outcome.manifest.package
            assert package is not None

            with zipfile.ZipFile(Path(package["path"])) as archive:
                packaged_record = json.loads(
                    archive.read("metadata/manifest.json")
                )["collectors"][0]
                summary = archive.read("reports/summary.txt").decode("utf-8")
            self.assertEqual(2, packaged_record["attempt_count"])
            self.assertEqual(1, len(packaged_record["retry_history"]))
            self.assertIsNone(packaged_record["failure"])
            self.assertTrue(packaged_record["retry_history"][0]["failure"]["retry_safe"])
            self.assertIn("Attempts: 2", summary)

    def test_retry_policy_stops_after_declared_attempt_limit(self) -> None:
        """A repeatedly failing collector may not exceed its explicitly declared retry cap."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            maximum_retries=1,',
                ).replace(
                    '        output = context.workspace / "system.txt"',
                    '        raise RuntimeError("persistent collector failure")',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]

            self.assertEqual("failed", record.status)
            self.assertEqual(2, record.attempt_count)
            self.assertEqual(1, len(record.retry_history))
            self.assertTrue(any("persistent collector failure" in error for error in record.errors))

    def test_failed_collector_with_registered_evidence_is_never_retried(self) -> None:
        """Evidence-producing failures must not be repeated or overwrite forensic state."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            collector_path.write_text(
                COLLECTOR.replace(
                    '            supported_platforms=("win32",),',
                    '            supported_platforms=("win32",),\n            maximum_retries=2,',
                ).replace(
                    '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                    '        raise RuntimeError("failure after evidence registration")',
                ),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=1, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]

            self.assertEqual("failed", record.status)
            self.assertEqual(1, record.attempt_count)
            self.assertEqual([], record.retry_history)
            self.assertTrue((outcome.run_directory / "artifacts" / "core_system_system_info" / "system.txt").is_file())
            self.assertEqual(1, len(record.artifacts))
            self.assertEqual(1, len(outcome.manifest.artifact_list()))

            failure = record.failure
            assert failure is not None

            self.assertEqual("core.system.system_info", failure["collector_id"])
            self.assertEqual("collect", failure["operation"])

            platform_error = failure["platform_error"]
            assert isinstance(platform_error, str)

            self.assertIn(
                "failure after evidence registration",
                platform_error,
            )

            self.assertFalse(failure["retry_safe"])
            package = outcome.manifest.package
            assert package is not None

            with zipfile.ZipFile(Path(package["path"])) as archive:
                self.assertIn(
                    "evidence/derived/core_system_system_info/system.txt",
                    archive.namelist(),
                )
                packaged = json.loads(
                    archive.read("metadata/manifest.json")
                )["collectors"][0]
                summary = archive.read("reports/summary.txt").decode("utf-8")
            self.assertEqual(record.failure, packaged["failure"])
            self.assertIn("Failed operation: collect", summary)
            self.assertIn("Retry safe: false", summary)
            self.assertIn("Remediation:", summary)

    def test_crashed_collector_runs_cleanup_and_packages_partial_evidence_and_isolation_results(self) -> None:
        """A failed process finalizes once, preserves evidence, and cannot stop its neighbor."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            failing_source = delayed_collector_source("a_failed", 0.0).replace(
                '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                '        raise RuntimeError("collection failed after evidence registration")',
            ).replace(
                '        """Release test resources."""',
                '        """Release test resources."""\n'
                '        (context.workspace / "cleanup.marker").write_text("cleaned", encoding="utf-8")\n'
                '        (context.temporary_directory / "scratch.txt").write_text("scratch", encoding="utf-8")',
            )
            (core_directory / "a_failed.py").write_text(failing_source, encoding="utf-8")
            (core_directory / "z_independent.py").write_text(
                delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}
            failed = records["core.system.a_failed"]
            succeeded = records["core.system.z_independent"]
            workspace = outcome.run_directory / "collectors" / "core_system_a_failed"

            self.assertEqual("partial", outcome.manifest.status.value)
            self.assertEqual("failed", failed.status)
            self.assertEqual("succeeded", succeeded.status)
            self.assertEqual("cleaned", (workspace / "cleanup.marker").read_text(encoding="utf-8"))
            self.assertFalse((workspace / "tmp").exists())
            self.assertEqual(1, len(failed.artifacts))
            self.assertIn("RuntimeError", "\n".join(failed.errors))
            self.assertEqual("process", failed.isolation_mode)
            self.assertIsInstance(failed.worker_pid, int)
            self.assertEqual("failed", failed.termination_reason)
            self.assertEqual("completed", succeeded.termination_reason)
            self.assertNotEqual(failed.worker_pid, succeeded.worker_pid)
            package = outcome.manifest.package
            assert package is not None

            with zipfile.ZipFile(Path(package["path"])) as archive:
                packaged = {
                    item["id"]: item
                    for item in json.loads(
                        archive.read("metadata/manifest.json")
                    )["collectors"]
                }
                summary = archive.read("reports/summary.txt").decode("utf-8")
                self.assertIn(
                    "evidence/derived/core_system_a_failed/system.txt",
                    archive.namelist(),
                )
            self.assertEqual("failed", packaged["core.system.a_failed"]["status"])
            self.assertEqual("process", packaged["core.system.a_failed"]["isolation_mode"])
            self.assertEqual(failed.worker_pid, packaged["core.system.a_failed"]["worker_pid"])
            self.assertEqual("succeeded", packaged["core.system.z_independent"]["status"])
            self.assertIn("Isolation: process", summary)
            self.assertIn("Termination: failed", summary)
            self.assertIn("collection failed after evidence registration", summary)

    def test_cleanup_failure_preserves_original_collector_crash_and_independent_results(self) -> None:
        """Cleanup errors must be reported without replacing the original failure."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            failing_source = delayed_collector_source("a_failed", 0.0).replace(
                '        return CollectorResult.succeeded("test artifact created", (artifact,))',
                '        raise RuntimeError("original collection failure")',
            ).replace(
                '        """Release test resources."""',
                '        """Release test resources."""\n'
                '        raise ValueError("secondary cleanup failure")',
            )
            (core_directory / "a_failed.py").write_text(failing_source, encoding="utf-8")
            (core_directory / "z_independent.py").write_text(
                delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )
            plan = build_plan(preflight(root), RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}
            errors = "\n".join(records["core.system.a_failed"].errors)

            self.assertEqual("failed", records["core.system.a_failed"].status)
            self.assertIn("RuntimeError: original collection failure", errors)
            self.assertIn("collector cleanup failed: ValueError: secondary cleanup failure", errors)
            self.assertEqual("succeeded", records["core.system.z_independent"].status)
            self.assertEqual("partial", outcome.manifest.status.value)

    def test_cleanup_failure_after_success_preserves_registered_evidence(self) -> None:
        """A failed finalizer turns success into failure without discarding registered bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()

            collector_path.write_text(
                COLLECTOR.replace(
                    '        """Release test resources."""',
                    '        """Release test resources."""\n'
                    '        raise RuntimeError("finalizer failed after collecting evidence")',
                ),
                encoding="utf-8",
            )

            plan = build_plan(
                preflight(root),
                RunRequest(
                    max_workers=1,
                    acknowledge_authorization=True,
                ),
            )

            outcome = RunSupervisor(
                root,
                default_config(root),
            ).run(plan)

            record = outcome.manifest.collectors[0]

            self.assertEqual("failed", record.status)
            self.assertEqual("collector cleanup failed", record.summary)
            self.assertEqual(1, len(record.artifacts))
            self.assertIn(
                "finalizer failed after collecting evidence",
                "\n".join(record.errors),
            )

            failure = record.failure
            assert failure is not None

            self.assertEqual("cleanup", failure["operation"])

            remediation = failure["remediation"]
            if not isinstance(remediation, str):
                self.fail(
                    f"remediation must be str, got {type(remediation).__name__}"
                )

            self.assertIn("cleanup", remediation)

            retry_safe = failure["retry_safe"]
            if not isinstance(retry_safe, bool):
                self.fail(
                    f"retry_safe must be bool, got {type(retry_safe).__name__}"
                )

            self.assertFalse(retry_safe)

            package = outcome.manifest.package
            assert package is not None
            assert "path" in package

            with zipfile.ZipFile(Path(package["path"])) as archive:
                self.assertIn(
                    "evidence/derived/core_system_system_info/system.txt",
                    archive.namelist(),
                )

    def test_collector_timeout_preserves_independent_worker_results(self) -> None:
        """A timed-out worker is terminated without cancelling unrelated collection."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()

            timeout_source = delayed_collector_source("a_timeout", 2.0).replace(
                '            supported_platforms=("win32",),',
                '            supported_platforms=("win32",),\n'
                '            timeout_seconds=1,',
            )

            (core_directory / "a_timeout.py").write_text(
                timeout_source,
                encoding="utf-8",
            )

            (core_directory / "z_independent.py").write_text(
                delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )

            report = preflight(root)
            self.assertEqual((), report.invalid)

            plan = build_plan(
                report,
                RunRequest(
                    max_workers=2,
                    acknowledge_authorization=True,
                ),
            )

            outcome = RunSupervisor(
                root,
                default_config(root),
            ).run(plan)

            records = {
                record.id: record
                for record in outcome.manifest.collectors
            }

            timeout_record = records["core.system.a_timeout"]

            self.assertEqual("failed", timeout_record.status)
            self.assertTrue(
                any("timeout" in error for error in timeout_record.errors)
            )
            self.assertEqual(
                "timeout_exceeded",
                timeout_record.termination_reason,
            )

            timeout_failure = timeout_record.failure
            assert timeout_failure is not None

            retry_safe = timeout_failure["retry_safe"]
            if not isinstance(retry_safe, bool):
                self.fail(
                    f"retry_safe must be bool, got {type(retry_safe).__name__}"
                )

            self.assertTrue(retry_safe)

            remediation = timeout_failure["remediation"]
            if not isinstance(remediation, str):
                self.fail(
                    f"remediation must be str, got {type(remediation).__name__}"
                )

            self.assertIn("timeout", remediation)

            independent_record = records["core.system.z_independent"]

            self.assertEqual("succeeded", independent_record.status)
            self.assertIsNone(independent_record.failure)
            self.assertEqual("partial", outcome.manifest.status.value)

    @unittest.skipUnless(os.name == "nt", "Windows collector subprocess-tree containment")
    def test_collector_timeout_terminates_spawned_subprocesses_without_stopping_peers(self) -> None:
        """A timed-out worker cannot leave its child alive or terminate independent collectors."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            source = delayed_collector_source("a_tree", 0.0).replace(
                "from pathlib import Path\n",
                "from pathlib import Path\nimport subprocess\nimport sys\nfrom logicytics import Capability\n",
            ).replace(
                '            supported_platforms=("win32",),',
                '            supported_platforms=("win32",),\n'
                '            capabilities=(Capability.SUBPROCESS,),\n            timeout_seconds=2,',
            ).replace(
                '        output = context.workspace / "system.txt"',
                '        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])\n'
                '        (context.workspace / "child.pid").write_text(str(child.pid), encoding="ascii")\n'
                '        sleep(5)\n'
                '        output = context.workspace / "system.txt"',
            )
            (core_directory / "a_tree.py").write_text(source, encoding="utf-8")
            (core_directory / "z_independent.py").write_text(
                delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(
                report,
                RunRequest(
                    max_workers=2,
                    acknowledge_authorization=True,
                    approved_capabilities=(Capability.SUBPROCESS,),
                ),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}
            child_pid = int(
                (outcome.run_directory / "collectors" / "core_system_a_tree" / "child.pid").read_text(
                    encoding="ascii"
                )
            )

            handle = open_process(child_pid)
            if handle is not None:
                try:
                    exit_code = get_exit_code_process(handle)
                    self.assertIsNotNone(exit_code)
                    self.assertNotEqual(
                        STILL_ACTIVE,
                        exit_code,
                        "collector subprocess survived worker termination",
                    )
                finally:
                    close_handle(handle)

            self.assertEqual("failed", records["core.system.a_tree"].status)
            self.assertEqual("timeout_exceeded", records["core.system.a_tree"].termination_reason)
            self.assertEqual("succeeded", records["core.system.z_independent"].status)

    @unittest.skipUnless(os.name == "nt", "Windows working-set enforcement test")
    def test_collector_memory_limit_preserves_independent_worker_results(self) -> None:
        """A memory-limit violation terminates only the offending isolated worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()

            limited_source = delayed_collector_source("a_limited", 2.0).replace(
                '            supported_platforms=("win32",),',
                '            supported_platforms=("win32",),\n'
                '            maximum_memory_bytes=1,',
            )

            (core_directory / "a_limited.py").write_text(
                limited_source,
                encoding="utf-8",
            )

            (core_directory / "z_independent.py").write_text(
                delayed_collector_source("z_independent", 0.0),
                encoding="utf-8",
            )

            report = preflight(root)
            self.assertEqual((), report.invalid)

            plan = build_plan(
                report,
                RunRequest(
                    max_workers=2,
                    acknowledge_authorization=True,
                ),
            )

            outcome = RunSupervisor(
                root,
                default_config(root),
            ).run(plan)

            records = {
                record.id: record
                for record in outcome.manifest.collectors
            }

            limited = records["core.system.a_limited"]

            self.assertEqual("failed", limited.status)

            peak_memory_bytes = limited.peak_memory_bytes
            assert peak_memory_bytes is not None

            self.assertGreater(peak_memory_bytes, 1)
            self.assertTrue(
                any(
                    "maximum_memory_bytes=1" in error
                    for error in limited.errors
                )
            )
            self.assertEqual(
                "memory_limit_exceeded",
                limited.termination_reason,
            )

            limited_failure = limited.failure
            assert limited_failure is not None

            self.assertEqual(
                "collect",
                limited_failure["operation"],
            )

            remediation = limited_failure["remediation"]
            if not isinstance(remediation, str):
                self.fail(
                    f"remediation must be str, got {type(remediation).__name__}"
                )

            self.assertIn(
                "memory limit",
                remediation,
            )

            retry_safe = limited_failure["retry_safe"]
            if not isinstance(retry_safe, bool):
                self.fail(
                    f"retry_safe must be bool, got {type(retry_safe).__name__}"
                )

            self.assertTrue(retry_safe)

            independent = records["core.system.z_independent"]

            self.assertEqual("succeeded", independent.status)
            self.assertEqual("partial", outcome.manifest.status.value)

    def test_cancelled_run_writes_a_recoverable_package_and_manifest(self) -> None:
        """Keyboard cancellation must affect only this run and preserve its partial report."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()

            collector_path.write_text(
                COLLECTOR,
                encoding="utf-8",
            )

            report = preflight(root)

            plan = build_plan(
                report,
                RunRequest(
                    max_workers=1,
                    acknowledge_authorization=True,
                ),
            )

            supervisor = RunSupervisor(
                root,
                default_config(root),
            )

            with patch.object(
                    supervisor,
                    supervisor._supervise.__name__,
                    side_effect=KeyboardInterrupt,
            ):
                outcome = supervisor.run(plan)

            self.assertEqual("cancelled", outcome.manifest.status.value)
            self.assertTrue(outcome.manifest.cancellation_requested)

            record = outcome.manifest.collectors[0]

            self.assertEqual("cancelled", record.status)
            self.assertEqual("run cancelled by user", record.summary)
            self.assertTrue((outcome.run_directory / ".cancelled").is_file())

            package = outcome.manifest.package
            assert package is not None
            assert "path" in package

            package_path = Path(package["path"])

            self.assertTrue(package_path.is_file())

            with zipfile.ZipFile(package_path) as archive:
                summary = archive.read(
                    "reports/summary.txt"
                ).decode("utf-8")

                packaged_manifest = json.loads(
                    archive.read("metadata/manifest.json")
                )

            self.assertTrue(packaged_manifest["cancellation_requested"])
            self.assertIn("Status: cancelled", summary)
            self.assertIn("Cancellation requested: true", summary)
            self.assertIn("Summary: run cancelled by user", summary)
            self.assertIn("Started: not started", summary)

            finished_at = record.finished_at
            assert finished_at is not None

            self.assertIn(
                f"Finished: {finished_at}",
                summary,
            )

    def test_worker_runs_typed_prepare_collect_finalize_and_cleanup_lifecycle(self) -> None:
        """Every accepted collector executes the complete typed lifecycle inside its worker."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector_path = root / "core" / "system" / "system_info.py"
            collector_path.parent.mkdir(parents=True)
            (root / "plugins").mkdir()
            source = COLLECTOR.replace(
                "    def collect(self, context: CollectorContext) -> CollectorResult:",
                "    def prepare(self, context: CollectorContext) -> ValidationResult:\n"
                "        \"\"\"Prepare worker-local lifecycle state.\"\"\"\n"
                "        context.settings['prepared'] = 'yes'\n"
                "        return ValidationResult(True)\n\n"
                "    def collect(self, context: CollectorContext) -> CollectorResult:",
            ).replace(
                "    def cleanup(self, context: CollectorContext) -> None:",
                "    def finalize(self, context: CollectorContext, result: CollectorResult) -> CollectorResult:\n"
                "        \"\"\"Finalize the typed result before it crosses the worker boundary.\"\"\"\n"
                "        if context.settings.get('prepared') != 'yes':\n"
                "            raise RuntimeError('prepare phase did not run')\n"
                "        return CollectorResult.succeeded('finalized lifecycle', result.artifacts)\n\n"
                "    def cleanup(self, context: CollectorContext) -> None:",
            )
            collector_path.write_text(source, encoding="utf-8")

            report = preflight(root)
            self.assertEqual(1, len(report.valid), report.invalid)
            plan = build_plan(
                report,
                RunRequest(max_workers=1, acknowledge_authorization=True),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            record = outcome.manifest.collectors[0]
            self.assertEqual("finalized lifecycle", record.summary)


if __name__ == "__main__":
    unittest.main()
