"""Stable artifact names and retention rules for shipped v4 core collectors."""

from __future__ import annotations

from dataclasses import dataclass

from logicytics.contracts import CollectorMetadata

_MEDIA_SUFFIX = {
    "application/json": ".json",
    "application/xml": ".xml",
    "application/zip": ".zip",
    "text/csv": ".csv",
    "text/html": ".html",
    "text/plain": ".txt",
    "text/vnd.graphviz": ".dot",
}

_SPECIAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "core.bluetooth.paired_devices": ("bluetooth_devices.json",),
    "core.browser.browser_data_backup": ("browser_data/*/*/*",),
    "core.filesystem.sensitive_file_inventory": ("sensitive_file_inventory/**",),
    "core.integration.legacy_code_outputs": ("legacy_code/**",),
    "core.media.media_backup": ("media_backup/**",),
    "core.process.memory_map": ("**/memory_map.json",),
    "core.registry.hklm_backup": ("hklm_backup.reg",),
    "core.system.windows_system_data_backup": ("windows_system_data/*/*",),
    "core.wireless.wifi_profile_keys": ("wifi_profiles_with_keys/*.xml",),
}


@dataclass(frozen=True, slots=True)
class OutputContract:
    """Canonical workspace paths, formats, package paths, and retention for one collector."""

    collector_id: str
    workspace_patterns: tuple[str, ...]
    media_types: tuple[str, ...]
    retention: str = "retained_with_run"

    @property
    def package_patterns(self) -> tuple[str, ...]:
        """Return canonical evidence paths used in verified packages."""
        owner = self.collector_id.replace(".", "_")
        return tuple(f"evidence/{{kind}}/{owner}/{pattern}" for pattern in self.workspace_patterns)


def core_output_contract(metadata: CollectorMetadata) -> OutputContract:
    """Derive the explicit v4 output contract for one shipped core collector."""
    if not metadata.id.startswith("core."):
        raise ValueError("stable built-in output contracts apply only to core collectors")
    patterns = _SPECIAL_PATTERNS.get(metadata.id)
    if patterns is None:
        if len(metadata.output_media_types) != 1:
            raise ValueError(f"{metadata.id} needs an explicit multi-format output contract")
        suffix = _MEDIA_SUFFIX.get(metadata.output_media_types[0])
        if suffix is None:
            raise ValueError(f"{metadata.id} needs an explicit output filename")
        patterns = (f"{metadata.id.rsplit('.', 1)[-1]}{suffix}",)
    return OutputContract(metadata.id, patterns, metadata.output_media_types)
