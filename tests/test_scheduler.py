"""Regression coverage for collector scheduling and resource coordination."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from logicytics.cli import cli_methods
from logicytics.contracts import (
    ResourceClass,
    RunRequest,
)
from logicytics.module.configuration import (
    default_config,
)
from logicytics.module.discovery import preflight
from logicytics.module.planner import build_plan
from logicytics.module.runtime import RunSupervisor
from tests.fixtures.collectors import delayed_collector_source


class SchedulerTests(unittest.TestCase):
    """Parallel scheduling, resource classes, and dependency ordering."""

    def test_parallel_completion_preserves_deterministic_manifest_order(self) -> None:
        """Collector completion order must not reorder the preflighted execution plan."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()

            (core_directory / "a_slow.py").write_text(
                delayed_collector_source("a_slow", 0.6),
                encoding="utf-8",
            )
            (core_directory / "z_fast.py").write_text(
                delayed_collector_source("z_fast", 0.0),
                encoding="utf-8",
            )

            report = preflight(root)
            self.assertEqual((), report.invalid)

            plan = build_plan(
                report,
                RunRequest(max_workers=2, acknowledge_authorization=True),
            )

            planned_ids: list[str] = []
            for candidate in plan.collectors:
                assert candidate.metadata is not None
                planned_ids.append(candidate.metadata.id)

            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertEqual(
                planned_ids,
                [record.id for record in outcome.manifest.collectors],
            )

            fast_finished = records["core.system.z_fast"].finished_at
            slow_finished = records["core.system.a_slow"].finished_at

            assert fast_finished is not None
            assert slow_finished is not None

            self.assertLess(fast_finished, slow_finished)

    def test_explicit_execution_modes_control_isolated_worker_overlap(self) -> None:
        """First-class CLI policies determine real sequential versus bounded worker overlap."""
        for execution_mode, expect_overlap in (("--sequential", False), ("--parallel", True)):
            with (
                self.subTest(execution_mode=execution_mode),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                core_directory = root / "core" / "system"
                core_directory.mkdir(parents=True)
                (root / "plugins").mkdir()
                for filename in ("a_first", "z_second"):
                    (core_directory / f"{filename}.py").write_text(
                        delayed_collector_source(filename, 0.3),
                        encoding="utf-8",
                    )
                arguments = cli_methods.parser().parse_args(["run", execution_mode, "--acknowledge-authorization"])
                run_request = cli_methods.request(arguments, default_workers=2)
                report = preflight(root)
                self.assertEqual((), report.invalid)
                outcome = RunSupervisor(root, default_config(root)).run(build_plan(report, run_request))
                records = {record.id: record for record in outcome.manifest.collectors}
                first = records["core.system.a_first"]
                second = records["core.system.z_second"]

                self.assertEqual("succeeded", first.status, first.errors)
                self.assertEqual("succeeded", second.status, second.errors)
                if expect_overlap:
                    assert second.started_at is not None and first.finished_at is not None
                    self.assertLess(second.started_at, first.finished_at)
                else:
                    assert first.finished_at is not None and second.started_at is not None
                    self.assertLessEqual(first.finished_at, second.started_at)

    def test_parallel_unsafe_collector_runs_without_worker_overlap(self) -> None:
        """A collector declaring parallel_safe=False must run alone between bounded workers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            fixtures = (
                ("a_parallel", 0.3, True),
                ("m_serial", 0.2, False),
                ("z_parallel", 0.0, True),
            )
            for filename, delay, parallel_safe in fixtures:
                (core_directory / f"{filename}.py").write_text(
                    delayed_collector_source(filename, delay, parallel_safe=parallel_safe),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(
                report,
                RunRequest(max_workers=3, acknowledge_authorization=True),
            )
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            first = records["core.system.a_parallel"]
            serial = records["core.system.m_serial"]
            last = records["core.system.z_parallel"]
            assert first.finished_at is not None and serial.started_at is not None
            self.assertLessEqual(first.finished_at, serial.started_at)
            assert serial.finished_at is not None and last.started_at is not None
            self.assertLessEqual(serial.finished_at, last.started_at)

    def test_conflicting_resource_classes_run_without_worker_overlap(self) -> None:
        """Disk, network, and registry resource conflicts each serialize their owners."""
        for resource_class in (
                ResourceClass.DISK_HEAVY,
                ResourceClass.NETWORK_HEAVY,
                ResourceClass.REGISTRY_SENSITIVE,
        ):
            with (
                self.subTest(resource_class=resource_class),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                core_directory = root / "core" / "system"
                core_directory.mkdir(parents=True)
                (root / "plugins").mkdir()
                for filename in ("a_first", "z_second"):
                    (core_directory / f"{filename}.py").write_text(
                        delayed_collector_source(filename, 0.15, resource_class=resource_class),
                        encoding="utf-8",
                    )
                report = preflight(root)
                self.assertEqual((), report.invalid)
                plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
                outcome = RunSupervisor(root, default_config(root)).run(plan)
                records = {record.id: record for record in outcome.manifest.collectors}
                first = records["core.system.a_first"]
                second = records["core.system.z_second"]

                self.assertEqual("succeeded", first.status, first.errors)
                self.assertEqual("succeeded", second.status, second.errors)
                assert first.finished_at is not None and second.started_at is not None
                self.assertLessEqual(first.finished_at, second.started_at)

    def test_independent_resource_classes_preserve_bounded_parallelism(self) -> None:
        """Different noninteractive resource owners can overlap in isolated workers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            fixtures = (
                ("a_disk", ResourceClass.DISK_HEAVY),
                ("z_network", ResourceClass.NETWORK_HEAVY),
            )
            for filename, resource_class in fixtures:
                (core_directory / f"{filename}.py").write_text(
                    delayed_collector_source(filename, 0.35, resource_class=resource_class),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            self.assertEqual("succeeded", records["core.system.a_disk"].status)
            self.assertEqual("succeeded", records["core.system.z_network"].status)
            network_started = records["core.system.z_network"].started_at
            disk_finished = records["core.system.a_disk"].finished_at
            assert network_started is not None and disk_finished is not None
            self.assertLess(network_started, disk_finished)

    def test_interactive_resource_class_runs_without_any_worker_overlap(self) -> None:
        """Interactive collectors require an exclusive deterministic scheduler window."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            fixtures = (
                ("a_disk", ResourceClass.DISK_HEAVY),
                ("m_interactive", ResourceClass.INTERACTIVE),
                ("z_general", ResourceClass.GENERAL),
            )
            for filename, resource_class in fixtures:
                (core_directory / f"{filename}.py").write_text(
                    delayed_collector_source(filename, 0.15, resource_class=resource_class),
                    encoding="utf-8",
                )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=3, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            disk_finished = records["core.system.a_disk"].finished_at
            interactive_started = records["core.system.m_interactive"].started_at
            interactive_finished = records["core.system.m_interactive"].finished_at
            general_started = records["core.system.z_general"].started_at
            assert disk_finished is not None and interactive_started is not None
            assert interactive_finished is not None and general_started is not None
            self.assertLessEqual(disk_finished, interactive_started)
            self.assertLessEqual(interactive_finished, general_started)

    def test_dependencies_finish_before_dependents_start(self) -> None:
        """Topological order must become an execution barrier under bounded parallelism."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_directory = root / "core" / "system"
            core_directory.mkdir(parents=True)
            (root / "plugins").mkdir()
            dependency_id = "core.system.a_dependency"
            (core_directory / "a_dependency.py").write_text(
                delayed_collector_source("a_dependency", 0.3),
                encoding="utf-8",
            )
            (core_directory / "b_dependent.py").write_text(
                delayed_collector_source("b_dependent", 0.0, dependencies=(dependency_id,)),
                encoding="utf-8",
            )
            report = preflight(root)
            self.assertEqual((), report.invalid)
            plan = build_plan(report, RunRequest(max_workers=2, acknowledge_authorization=True))
            outcome = RunSupervisor(root, default_config(root)).run(plan)
            records = {record.id: record for record in outcome.manifest.collectors}

            dependency = records[dependency_id]
            dependent = records["core.system.b_dependent"]
            self.assertEqual("succeeded", dependent.status, dependent.errors)
            self.assertIsNotNone(dependency.finished_at)
            self.assertIsNotNone(dependent.started_at)
            assert dependency.finished_at is not None and dependent.started_at is not None
            self.assertLessEqual(dependency.finished_at, dependent.started_at)


if __name__ == "__main__":
    unittest.main()
