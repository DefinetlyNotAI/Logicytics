"""Capture bounded local IPv4 packet observations after explicit approval."""

from __future__ import annotations

import csv
import struct
import time

import select

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    NetworkAccess,
    PrivilegeLevel,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import network_adapter as socket


def _packet_row(payload: bytes) -> dict[str, str] | None:
    """Decode a minimal IPv4/TCP/UDP/ICMP observation without retaining payload data."""
    if len(payload) < 20 or payload[0] >> 4 != 4:
        return None
    header_length = (payload[0] & 15) * 4
    if header_length < 20 or len(payload) < header_length:
        return None
    protocol = payload[9]
    names = {1: "ICMP", 6: "TCP", 17: "UDP"}
    source = socket.inet_ntoa(payload[12:16])
    destination = socket.inet_ntoa(payload[16:20])
    source_port = destination_port = ""
    if protocol in {6, 17} and len(payload) >= header_length + 4:
        source_port, destination_port = (str(value) for value in
                                         struct.unpack("!HH", payload[header_length: header_length + 4]))
    return {
        "source_ip": source,
        "destination_ip": destination,
        "protocol": names.get(protocol, str(protocol)),
        "source_port": source_port,
        "destination_port": destination_port,
        "packet_bytes": str(len(payload)),
    }


class PacketCaptureCollector(CoreCollector):
    """Capture metadata-only packet observations inside the isolated worker process."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the explicitly approved packet-capture CSV artifact contract."""
        return CollectorMetadata(
            id="core.packet.packet_capture",
            name="IPv4 packet capture",
            version="4.0.0",
            specialty=Specialty.PACKET,
            output_media_types=("text/csv",),
            description="Captures bounded IPv4 packet metadata without saving packet payloads.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(
                Capability.NETWORK,
                Capability.PACKET_CAPTURE,
                Capability.ELEVATED_PRIVILEGES,
            ),
            privilege_level=PrivilegeLevel.ELEVATED,
            network_access=NetworkAccess.LOCAL,
            sensitive_data_categories=("network_metadata",),
            default_profiles=("deep",),
            timeout_seconds=90,
            maximum_output_bytes=2 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Validate bounded capture settings before raw-socket creation."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        try:
            count = context.setting_int("packet_count", 100)
            timeout = context.setting_float("timeout_seconds", 10.0)
            retry_window = context.setting_float("retry_window_seconds", 0.0)
        except (TypeError, ValueError):
            return ValidationResult(
                False,
                reasons=("packet_count, timeout_seconds, and retry_window_seconds must be numeric",),
            )
        if not 1 <= count <= 10_000 or not 1 <= timeout <= 60 or not 0 <= retry_window <= 60:
            return ValidationResult(
                False,
                reasons=("packet_count must be 1-10000, timeout_seconds 1-60, and retry_window_seconds 0-60",),
            )
        return ValidationResult(True)

    @staticmethod
    def _close_capture(capture_socket: socket.socket) -> None:
        """Disable Windows promiscuous capture mode and close the socket."""
        try:
            capture_socket.ioctl(
                socket.SIO_RCVALL,
                socket.RCVALL_OFF,
            )
        except OSError:
            pass
        capture_socket.close()

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Capture bounded metadata and save it as CSV without retaining payload bytes."""
        if context.is_cancelled:
            return CollectorResult(
                CollectorStatus.CANCELLED,
                "cancelled before packet capture",
            )

        count = context.setting_int("packet_count", 100)
        timeout = context.setting_float("timeout_seconds", 10.0)
        retry_window = context.setting_float("retry_window_seconds", 0.0)
        interface = context.setting_str(
            "interface",
            socket.gethostbyname(socket.gethostname()),
        )

        context.report_progress(
            "packet_capture_started",
            interface=interface,
            packet_count=count,
            retry_window_seconds=retry_window,
        )

        output = context.workspace / "packet_capture.csv"
        capture: socket.socket | None = None
        observation_count = 0
        early_result: CollectorResult | None = None

        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "source_ip",
                    "destination_ip",
                    "protocol",
                    "source_port",
                    "destination_port",
                    "packet_bytes",
                ),
            )
            writer.writeheader()

            try:
                active_capture = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_RAW,
                    socket.IPPROTO_IP,
                )
                capture = active_capture

                active_capture.bind((interface, 0))
                active_capture.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_HDRINCL,
                    1,
                )
                active_capture.ioctl(
                    socket.SIO_RCVALL,
                    socket.RCVALL_ON,
                )

                deadline = time.monotonic() + timeout
                retry_deadline = time.monotonic() + retry_window

                while observation_count < count and time.monotonic() < deadline:
                    if context.is_cancelled:
                        early_result = CollectorResult(
                            CollectorStatus.CANCELLED,
                            "cancelled during packet capture",
                        )
                        break

                    remaining = max(0.0, deadline - time.monotonic())
                    ready, _, _ = select.select(
                        [active_capture],
                        [],
                        [],
                        min(1.0, remaining),
                    )

                    if not ready:
                        continue

                    try:
                        payload = active_capture.recv(65_535)
                    except OSError:
                        if time.monotonic() < retry_deadline:
                            time.sleep(0.1)
                            continue
                        raise

                    row = _packet_row(payload)
                    if row is not None:
                        writer.writerow(row)
                        observation_count += 1

            except PermissionError as error:
                early_result = CollectorResult(
                    CollectorStatus.SKIPPED,
                    "raw packet capture requires an elevated account",
                    errors=(str(error),),
                )

            except OSError as error:
                if error.winerror in {5, 10013}:
                    early_result = CollectorResult(
                        CollectorStatus.SKIPPED,
                        "raw packet capture was denied for the current account",
                        errors=(str(error),),
                    )
                else:
                    early_result = CollectorResult(
                        CollectorStatus.FAILED,
                        "raw packet capture failed",
                        errors=(str(error),),
                    )

            finally:
                if capture is not None:
                    self._close_capture(capture)

        if early_result is not None:
            output.unlink(missing_ok=True)
            return early_result

        if context.is_cancelled:
            output.unlink(missing_ok=True)
            return CollectorResult(
                CollectorStatus.CANCELLED,
                "cancelled after packet capture",
            )

        artifact = context.artifacts.register_file(
            output,
            media_type="text/csv",
        )

        context.report_progress(
            "packet_capture_finished",
            observation_count=observation_count,
            bytes_written=artifact.size_bytes,
        )

        return CollectorResult.succeeded(
            "packet metadata captured",
            (artifact,),
        )

    def cleanup(self, context: CollectorContext) -> None:
        """Raw socket cleanup happens in collect's finally block."""
