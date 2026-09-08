from __future__ import annotations

import unittest

from logicytics.contracts import (
    Artifact,
    CollectorMetadata,
    CollectorResult,
    CollectorStatus,
    EvidenceKind,
    ResourceClass,
    Specialty,
)


class ContractTests(unittest.TestCase):
    """Typed metadata, result, specialty, and artifact contracts."""

    def test_custom_specialty_is_plugin_only(self) -> None:
        """Plugins may extend specialties, but core metadata remains on the closed set."""
        metadata = CollectorMetadata(
            id="plugin.example",
            name="Example",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            description="Test metadata.",
            author="Test",
        ).to_dict()
        metadata["specialty"] = "evidence_graph"
        self.assertEqual(
            "evidence_graph",
            CollectorMetadata.from_dict(metadata, allow_custom_specialty=True).specialty,
        )
        with self.assertRaises(ValueError):
            CollectorMetadata.from_dict(metadata)

    def test_collector_metadata_rejects_invalid_identity_and_limits(self) -> None:
        """Collector metadata must be a complete typed declaration rather than free text."""
        common = dict(
            id="core.system.example",
            name="Example",
            version="4.0.0",
            specialty=Specialty.SYSTEM,
            description="Example collector.",
            author="tests",
        )
        with self.assertRaisesRegex(ValueError, "id has an invalid schema"):
            CollectorMetadata(**{**common, "id": "example"})
        with self.assertRaisesRegex(ValueError, "semantic versioning"):
            CollectorMetadata(**{**common, "version": "four"})
        with self.assertRaisesRegex(ValueError, "maximum_artifact_files"):
            CollectorMetadata(**{**common, "maximum_artifact_files": 0})
        with self.assertRaisesRegex(ValueError, "maximum_memory_bytes"):
            CollectorMetadata(**{**common, "maximum_memory_bytes": True})
        with self.assertRaisesRegex(ValueError, "maximum_artifact_bytes"):
            CollectorMetadata(**{**common, "maximum_artifact_bytes": True})
        with self.assertRaisesRegex(ValueError, "maximum_artifact_bytes"):
            CollectorMetadata(**{**common, "maximum_output_bytes": 8, "maximum_artifact_bytes": 9})
        with self.assertRaisesRegex(ValueError, "resource_class"):
            CollectorMetadata(**{**common, "resource_class": "disk_heavy"})
        with self.assertRaisesRegex(ValueError, "ResourceClass"):
            CollectorMetadata.from_dict({**CollectorMetadata(**common).to_dict(), "resource_class": "unknown"})
        with self.assertRaisesRegex(ValueError, "sensitive collectors"):
            CollectorMetadata(**{**common, "sensitive_data_categories": ("credentials",)})
        for retries in (-1, 4, True):
            with self.subTest(maximum_retries=retries):
                with self.assertRaisesRegex(ValueError, "maximum_retries"):
                    CollectorMetadata(**{**common, "maximum_retries": retries})
        for retry_delay in (-1, 31, True, float("inf")):
            with self.subTest(retry_delay_seconds=retry_delay):
                with self.assertRaisesRegex(ValueError, "retry_delay_seconds"):
                    CollectorMetadata(**{**common, "retry_delay_seconds": retry_delay})
        metadata = CollectorMetadata(**{**common, "maximum_output_bytes": 8})
        self.assertEqual(8, metadata.maximum_artifact_bytes)
        self.assertIs(ResourceClass.GENERAL, metadata.resource_class)
        disk_metadata = CollectorMetadata(**{**common, "resource_class": ResourceClass.DISK_HEAVY})
        self.assertEqual("disk_heavy", disk_metadata.to_dict()["resource_class"])
        self.assertIs(
            ResourceClass.DISK_HEAVY,
            CollectorMetadata.from_dict(disk_metadata.to_dict()).resource_class,
        )

    def test_collector_results_expose_all_strict_typed_terminal_states(self) -> None:
        """Every terminal collector outcome has an explicit, validated result constructor."""
        outcomes = (
            (CollectorResult.succeeded("complete"), CollectorStatus.SUCCEEDED),
            (
                CollectorResult.partial("incomplete", errors=("one source unavailable",)),
                CollectorStatus.PARTIAL,
            ),
            (CollectorResult.skipped("prerequisite unavailable"), CollectorStatus.SKIPPED),
            (CollectorResult.cancelled("operator cancelled"), CollectorStatus.CANCELLED),
            (
                CollectorResult.failed("collection failed", errors=("access denied",)),
                CollectorStatus.FAILED,
            ),
        )
        for result, status in outcomes:
            with self.subTest(status=status):
                self.assertIs(status, result.status)
        invalid = (
            ({"status": "succeeded", "summary": "complete"}, "status"),
            ({"status": CollectorStatus.SUCCEEDED, "summary": ""}, "summary"),
            (
                {"status": CollectorStatus.SUCCEEDED, "summary": "complete", "artifacts": []},
                "artifacts",
            ),
            (
                {"status": CollectorStatus.FAILED, "summary": "failed", "errors": ["failure"]},
                "errors",
            ),
            ({"status": CollectorStatus.FAILED, "summary": "failed", "errors": ("",)}, "errors"),
            (
                {
                    "status": CollectorStatus.PARTIAL,
                    "summary": "partial",
                    "metrics": {"count": True},
                },
                "metrics",
            ),
            (
                {
                    "status": CollectorStatus.PARTIAL,
                    "summary": "partial",
                    "metrics": {"count": float("inf")},
                },
                "metrics",
            ),
        )
        for options, message in invalid:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    CollectorResult(**options)

    def test_artifact_contract_rejects_malformed_catalog_metadata(self) -> None:
        """Evidence records validate identity, MIME type, provenance, timestamps, and status."""
        valid = {
            "id": "artifact." + "a" * 32,
            "relative_path": "core_system_example/report.json",
            "sha256": "b" * 64,
            "size_bytes": 2,
            "media_type": "application/json",
            "collector_id": "core.system.example",
            "source_category": "system",
            "collected_at": "2026-01-01T00:00:00+00:00",
            "transformations": ("normalized",),
            "evidence_kind": EvidenceKind.DERIVED,
            "name": "report.json",
            "status": "registered",
        }
        invalid = (
            ({"id": "artifact.invalid"}, "id"),
            ({"sha256": "bad"}, "sha256"),
            ({"size_bytes": True}, "size_bytes"),
            ({"size_bytes": -1}, "size_bytes"),
            ({"media_type": "not a mime type"}, "media_type"),
            ({"collector_id": "another"}, "collector_id"),
            ({"source_category": "System"}, "source_category"),
            ({"collected_at": "2026-01-01T00:00:00"}, "timezone"),
            ({"transformations": ["normalized"]}, "transformations"),
            ({"evidence_kind": "derived"}, "evidence_kind"),
            ({"name": "../report.json"}, "name"),
            ({"status": "failed"}, "status"),
        )
        self.assertEqual("report.json", Artifact(**valid).name)
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    Artifact(**{**valid, **changes})


if __name__ == "__main__":
    unittest.main()
