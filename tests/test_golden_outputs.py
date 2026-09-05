"""Byte-stable golden tests for every structured v4 output family."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from core.packet.connection_graph import render_connection_graph
from core.process import running_processes
from core.system import system_details
from core.system.bios_info import render_bios_table
from logicytics.contracts import (
    Artifact, ArtifactWriter, CollectorContext, EvidenceKind, EventLogger, RunStatus,
)
from logicytics.manifest import CollectorRecord, RunManifest

_GOLDEN = Path(__file__).parent / "golden"


class _Logger(EventLogger):
    def event(self, level: str, message: str, **fields: int | float | str) -> None:
        return None


class _CaptureWriter(ArtifactWriter):
    def __init__(self) -> None:
        self.contents = b""

    def register_file(self, source: Path, *, media_type: str = "application/octet-stream",
                      transformations: tuple[str, ...] = (), **kwargs: object) -> Artifact:
        self.contents = source.read_bytes()
        return Artifact(
            id="artifact." + "0" * 32,
            relative_path=f"core_test/{source.name}",
            sha256="0" * 64,
            size_bytes=len(self.contents),
            media_type=media_type,
            collector_id="core.test",
            source_category="test",
            collected_at="2026-01-02T03:04:05+00:00",
            transformations=transformations,
            evidence_kind=EvidenceKind.DERIVED,
            name=source.name,
        )


def _context(root: Path, collector_id: str, writer: _CaptureWriter) -> CollectorContext:
    return CollectorContext(
        run_id="run-" + "0" * 32,
        collector_id=collector_id,
        workspace=root,
        temporary_directory=root / "tmp",
        artifacts=writer,
        logger=_Logger(),
        settings={},
        cancellation_file=root / ".cancelled",
    )


class GoldenOutputTests(unittest.TestCase):
    def test_text_output_matches_golden_file(self) -> None:
        expected = (_GOLDEN / "system_details.txt").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            writer = _CaptureWriter()
            with patch.object(
                    system_details.subprocess,
                    system_details.subprocess.run.__name__,
                    return_value=CompletedProcess([], 0, expected, ""),
            ):
                system_details.SystemDetailsCollector().collect(
                    _context(
                        Path(temporary),
                        "core.system.system_details",
                        writer,
                    )
                )
        self.assertEqual(expected.encode(), writer.contents)

    def test_csv_output_matches_golden_file(self) -> None:
        expected = (_GOLDEN / "running_processes.csv").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            writer = _CaptureWriter()
            with patch.object(
                    running_processes.subprocess,
                    running_processes.subprocess.run.__name__,
                    return_value=CompletedProcess([], 0, expected, ""),
            ):
                running_processes.RunningProcessesCollector().collect(
                    _context(
                        Path(temporary),
                        "core.process.running_processes",
                        writer,
                    )
                )
        self.assertEqual(expected.encode(), writer.contents)

    def test_html_output_matches_golden_file(self) -> None:
        rendered = render_bios_table({
            "Manufacturer": "Example & Sons",
            "Name": "Golden <BIOS>",
            "SMBIOSBIOSVersion": "1.2.3",
            "Version": "EXAMPLE - 1",
            "ReleaseDate": "2026-01-02",
        })
        self.assertEqual((_GOLDEN / "bios_info.html").read_text(encoding="utf-8"), rendered)

    def test_graph_output_matches_golden_file(self) -> None:
        rendered = render_connection_graph(
            "UDP [::1]:5353 [::1]:5354 999\n"
            "TCP 10.0.0.5:50000 1.1.1.1:443 ESTABLISHED 4242\n"
            "TCP 10.0.0.5:50000 1.1.1.1:443 ESTABLISHED 4242\n"
        )
        self.assertEqual((_GOLDEN / "connection_graph.dot").read_text(encoding="utf-8"), rendered)

    def test_normalized_manifest_matches_golden_shape(self) -> None:
        manifest = RunManifest(
            run_id="run-" + "0" * 32,
            requested_at="<TIMESTAMP>",
            status=RunStatus.SUCCEEDED,
            request={"profile": "minimal"},
            configuration={"schema_version": 4},
            collectors=[CollectorRecord(
                id="core.system.system_info", source="<SOURCE>", status="succeeded",
                started_at="<TIMESTAMP>", finished_at="<TIMESTAMP>", summary="collected",
            )],
            host={"platform": "win32", "hostname": "<HOST>", "python": "<PYTHON>",
                  "user": "<USER>", "is_administrator": "false"},
            resolved_plan=("core.system.system_info",),
            plan_fingerprint="0" * 64,
            finished_at="<TIMESTAMP>",
        )
        normalized = json.loads(json.dumps(manifest.to_dict(), sort_keys=True))
        expected = json.loads((_GOLDEN / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(expected, normalized)


if __name__ == "__main__":
    unittest.main()
