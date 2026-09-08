"""Export a labeled local connection graph in DOT format."""

from __future__ import annotations

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import process_adapter as subprocess
from logicytics.platform_adapters import which


def _is_access_denied(detail: str) -> bool:
    """Recognize common Windows permission-denied wording."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


def render_connection_graph(command_output: str) -> str:
    """Normalize netstat rows into deterministic, sorted Graphviz DOT source."""
    edges: set[tuple[str, str, str]] = set()
    for line in command_output.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0].upper() not in {"TCP", "UDP"}:
            continue
        protocol, source, destination = fields[:3]
        if destination in {"*:*", "*"}:
            continue
        edges.add((source, destination, protocol.upper()))

    def quote(value: str) -> str:
        """Escape one label for safe inclusion in a quoted DOT string."""
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = ["digraph connection_graph {", "  rankdir=LR;"]
    lines.extend(f"  {quote(source)} -> {quote(destination)} [label={quote(protocol)}];" for source, destination, protocol in sorted(edges))
    lines.append("}")
    return "\n".join(lines) + "\n"


class ConnectionGraphCollector(CoreCollector):
    """Build a DOT connection graph without retaining packet payloads."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated network graph artifact contract."""
        return CollectorMetadata(
            id="core.packet.connection_graph",
            name="Connection graph",
            version="4.0.0",
            specialty=Specialty.PACKET,
            output_media_types=("text/vnd.graphviz",),
            description="Exports a DOT source/destination graph with TCP or UDP protocol edge labels.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("network_metadata",),
            default_profiles=("deep",),
            timeout_seconds=45,
            maximum_output_bytes=2 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and netstat availability before graph collection."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("netstat") is None:
            return ValidationResult(False, reasons=("netstat is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Read active connections and write a bounded DOT graph artifact."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before connection graph collection")
        context.report_progress("connection_graph_started")
        completed = subprocess.run(["netstat", "-ano"], capture_output=True, check=False, text=True, timeout=40)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"netstat exit code {completed.returncode}"
            if _is_access_denied(detail):
                return CollectorResult(
                    CollectorStatus.SKIPPED,
                    "connection graph access was denied for the current account",
                    errors=(detail,),
                )
            return CollectorResult(CollectorStatus.FAILED, "connection graph query failed", errors=(detail,))
        rendered = render_connection_graph(completed.stdout)
        output = context.workspace / "connection_graph.dot"
        output.write_text(rendered, encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/vnd.graphviz")
        edge_count = sum(1 for line in rendered.splitlines() if " -> " in line)
        context.report_progress("connection_graph_finished", edge_count=edge_count, bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("connection graph collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Retain no graph or plot state; DOT generation uses only collection-local values."""
