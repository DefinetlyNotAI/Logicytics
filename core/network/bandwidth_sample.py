"""Measure local adapter bandwidth from bounded read-only counter samples."""

from __future__ import annotations

import json
from logicytics.platform_adapters import process_adapter as subprocess
import time
from logicytics.platform_adapters import which

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus

_DEFAULT_SAMPLES = 3
_DEFAULT_INTERVAL_SECONDS = 1


def _setting(settings: object, key: str, default: int, maximum: int) -> int:
    """Return one bounded positive integer collector setting."""
    value = settings.get(key, default) if isinstance(settings, dict) else default
    return value if isinstance(value, int) and 1 <= value <= maximum else default


def _interval_setting(settings: object) -> float:
    """Return the configured finite positive sampling interval."""
    value = settings.get("interval_seconds", _DEFAULT_INTERVAL_SECONDS) if isinstance(settings, dict) else _DEFAULT_INTERVAL_SECONDS
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.1 <= value <= 60:
        return float(value)
    return float(_DEFAULT_INTERVAL_SECONDS)


def _is_access_denied(detail: str) -> bool:
    """Recognize common permission-denied wording from PowerShell output."""
    normalized = detail.casefold()
    return "permission denied" in normalized or ("access" in normalized and "denied" in normalized)


class BandwidthSampleCollector(CoreCollector):
    """Measure receive/send byte rates locally without generating network traffic."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the subprocess-gated bounded bandwidth-sampling artifact contract."""
        return CollectorMetadata(
            id="core.network.bandwidth_sample", name="Network bandwidth sample", version="4.0.0",
            specialty=Specialty.NETWORK,
            output_media_types=("application/json",),
            description="Calculates local per-interface average and peak bandwidth from adapter counter samples.",
            author="Logicytics",
            supported_platforms=("win32",), capabilities=(Capability.SUBPROCESS,),
            sensitive_data_categories=("network_identifiers",),
            default_profiles=("deep",), timeout_seconds=90, maximum_output_bytes=512 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state and PowerShell availability before sampling."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        if which("powershell") is None:
            return ValidationResult(False, reasons=("PowerShell is unavailable on this system",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Take bounded counter samples and register calculated rate evidence as JSON."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before bandwidth sampling")
        samples = _setting(context.settings, "sample_count", _DEFAULT_SAMPLES, 10)
        interval = _interval_setting(context.settings)
        command = "Get-NetAdapterStatistics | Select-Object Name, ReceivedBytes, SentBytes | ConvertTo-Json -Depth 3"
        observations: list[dict[str, dict[str, int]]] = []
        context.report_progress("bandwidth_sample_started", sample_count=samples, interval_seconds=interval)
        for index in range(samples):
            if context.is_cancelled:
                return CollectorResult(CollectorStatus.CANCELLED, "cancelled during bandwidth sampling")
            completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                       capture_output=True, check=False, text=True, timeout=20)
            if completed.returncode != 0:
                detail = completed.stderr.strip() or f"PowerShell exit code {completed.returncode}"
                if _is_access_denied(detail):
                    return CollectorResult(CollectorStatus.SKIPPED,
                                           "bandwidth-sample access was denied for the current account",
                                           errors=(detail,))
                return CollectorResult(CollectorStatus.FAILED, "bandwidth-sample query failed", errors=(detail,))
            try:
                raw = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                return CollectorResult(CollectorStatus.FAILED, "bandwidth-sample query returned invalid JSON",
                                       errors=(str(error),))
            records = raw if isinstance(raw, list) else [raw]
            if not all(isinstance(record, dict) for record in records):
                return CollectorResult(CollectorStatus.FAILED, "bandwidth-sample query returned an unexpected result")
            observations.append({str(record.get("Name", "unavailable")): {
                "received": int(record.get("ReceivedBytes", 0)), "sent": int(record.get("SentBytes", 0))} for record in
                                 records})
            if index + 1 < samples:
                time.sleep(interval)
        rates: dict[str, dict[str, float]] = {}
        for previous, current in zip(observations, observations[1:]):
            for name, counters in current.items():
                if name not in previous:
                    continue
                receive = max(0, counters["received"] - previous[name]["received"]) / interval
                sent = max(0, counters["sent"] - previous[name]["sent"]) / interval
                values = rates.setdefault(name, {"receive_average_bytes_per_second": 0.0,
                                                 "receive_peak_bytes_per_second": 0.0,
                                                 "send_average_bytes_per_second": 0.0,
                                                 "send_peak_bytes_per_second": 0.0, "sample_intervals": 0.0})
                values["sample_intervals"] += 1
                values["receive_average_bytes_per_second"] += receive
                values["send_average_bytes_per_second"] += sent
                values["receive_peak_bytes_per_second"] = max(values["receive_peak_bytes_per_second"], receive)
                values["send_peak_bytes_per_second"] = max(values["send_peak_bytes_per_second"], sent)
        for values in rates.values():
            values["receive_average_bytes_per_second"] /= values["sample_intervals"]
            values["send_average_bytes_per_second"] /= values["sample_intervals"]
        output = context.workspace / "bandwidth_sample.json"
        output.write_text(
            json.dumps({"sample_count": samples, "interval_seconds": interval, "interfaces": rates}, indent=2,
                       sort_keys=True) + "\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="application/json")
        context.report_progress("bandwidth_sample_finished", interface_count=len(rates),
                                bytes_written=artifact.size_bytes)
        return CollectorResult.succeeded("network bandwidth sample collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Release no resources because sampling subprocesses have already exited."""
