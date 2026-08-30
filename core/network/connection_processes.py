"""Export local active connections correlated with process names as bounded CSV evidence."""

from __future__ import annotations

import csv
import io
from logicytics.platform_adapters import process_adapter as subprocess
from shutil import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from Windows command output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


def _parse_connections(netstat_output: str) -> list[dict[str, str]]:
    """Parse the stable token layout of Windows netstat -ano connection rows."""
    connections: list[dict[str, str]] = []
    for line in netstat_output.splitlines():
        fields = line.split()
        if not fields or fields[0] not in {"TCP", "UDP"}:
            continue
        if fields[0] == "TCP" and len(fields) >= 5:
            protocol, local, remote, state, pid = fields[:5]
        elif fields[0] == "UDP" and len(fields) >= 4:
            protocol, local, remote, pid = fields[:4]
            state = ""
        else:
            continue
        connections.append(
            {"protocol": protocol, "local_endpoint": local, "remote_endpoint": remote, "state": state, "pid": pid})
    return connections


def _parse_processes(tasklist_output: str) -> dict[str, str]:
    """Map tasklist CSV process identifiers to their image names."""
    processes: dict[str, str] = {}
    for row in csv.reader(io.StringIO(tasklist_output)):
        if len(row) >= 2 and row[1].isdigit():
            processes[row[1]] = row[0]
    return processes


class ConnectionProcessesCollector(CoreCollector):
    """Capture active connection endpoints with the local process name for each PID."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated connection-process CSV artifact contract."""
        return CollectorMetadata(
            id="core.network.connection_processes",
            name="Connection process associations",
            version="4.0.0",
            specialty=Specialty.NETWORK,
            output_media_types=("text/csv",),
            description="Correlates active Netstat TCP/UDP endpoints with local process names and PIDs.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("network_identifiers", "process_metadata"),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=4 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and required Windows command availability."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        missing = tuple(command for command in ("netstat", "tasklist") if which(command) is None)
        if missing:
            return ValidationResult(False, reasons=(f"required command is unavailable: {', '.join(missing)}",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Read Netstat and Tasklist data, then register their correlation as CSV."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before connection-process collection")
        context.report_progress("connection_processes_started")
        netstat = subprocess.run(["netstat", "-ano"], capture_output=True, check=False, text=True, timeout=25)
        tasklist = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, check=False, text=True,
                                  timeout=25)
        failures = [result for result in (netstat, tasklist) if result.returncode != 0]
        if failures:
            detail = "\n".join(result.stderr.strip() or f"command exit code {result.returncode}" for result in failures)
            if _is_access_denied(detail):
                return CollectorResult(CollectorStatus.SKIPPED,
                                       "connection-process access was denied for the current account", errors=(detail,))
            return CollectorResult(CollectorStatus.FAILED, "connection-process query failed", errors=(detail,))
        processes = _parse_processes(tasklist.stdout)
        rows = _parse_connections(netstat.stdout)
        output = context.workspace / "connection_processes.csv"
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("protocol", "local_endpoint", "remote_endpoint", "state", "pid",
                                                        "process_name"))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "process_name": processes.get(row["pid"], "unavailable")})
        artifact = context.artifacts.register_file(output, media_type="text/csv")
        context.report_progress("connection_processes_finished", connection_count=len(rows),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("connection-process associations collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because both commands exit before the result is returned."""
