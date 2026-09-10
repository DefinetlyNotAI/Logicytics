# Configuration

`logicytics.yaml` is the authoritative user configuration. It is a strict, mapping-only YAML subset; JSON is accepted when it is also valid YAML. The parser rejects unknown keys, duplicate keys, unsafe paths, non-finite numbers, unsupported values, and files larger than 2 MiB before planning.

## Root keys

The root keys are `schema_version`, `runtime`, `interaction`, `maintenance`, `logging`, and `collectors`. The parser currently requires `schema_version: 4`; this is a file-schema identifier, not a release guide.

| Section       | Keys                                                                                                                                                                |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `runtime`     | `output_root`, `default_max_workers`, `maximum_workers`, `package_completed_runs`, `maximum_run_output_bytes`, `blocked_capabilities`, `temporary_directory`        |
| `interaction` | `history_enabled`, `similarity_threshold`, `model_name`, `model_debug`                                                                                              |
| `maintenance` | `remote_manifest_url`, `remote_manifest_sha256`, `local_manifest_path`, `minimum_python`, `recommended_python`, `sysinternals_enabled`, `sysinternals_download_url` |
| `logging`     | `level`, `console_enabled`, `color_enabled`, `file_enabled`, `maximum_bytes`, `delete_previous`, `retention_days`                                                   |
| `collectors`  | A mapping from collector ID to that collector's declared settings                                                                                                   |

Relative `runtime.output_root` and `maintenance.local_manifest_path` stay inside the project. `temporary_directory` is `project` or `system`. `blocked_capabilities` is a mapping of capability names to booleans; `true` blocks matching requests. Remote manifests require HTTPS and a lowercase SHA-256 digest. Optional Sysinternals discovery is controlled by `sysinternals_enabled` and its download URL.

## Collector settings

Core collectors reject settings they do not declare. Extension IDs may define their own settings. The supported core settings are:

| Collector                                  | Settings                                                                                                                             |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `core.network.bandwidth_sample`            | `sample_count` 1–10; `interval_seconds` 0.1–60                                                                                       |
| `core.packet.packet_capture`               | `packet_count` 1–10000; `timeout_seconds` 1–60; `retry_window_seconds` 0–60; `interface` text                                        |
| `core.filesystem.system_drive_tree`        | `max_entries` 1–50000; `max_depth` 1–32                                                                                              |
| `core.filesystem.system_drive_listing`     | `max_entries` 1–50000; `max_depth` 1–32                                                                                              |
| `core.filesystem.sensitive_file_inventory` | `root` path; `max_directories` 1–50000; `max_matches` 1–5000                                                                         |
| `core.process.memory_map`                  | `max_regions` 1–100000; `output_limit_bytes` 1024–67108864; `disk_safety_margin_bytes` 0–68719476736; `dump_directory` relative path |

## Complete example

```json
{
  "schema_version": 4,
  "runtime": {
    "output_root": "output/data",
    "default_max_workers": 4,
    "maximum_workers": 16,
    "package_completed_runs": true,
    "maximum_run_output_bytes": 4294967296,
    "temporary_directory": "project",
    "blocked_capabilities": {"network": false, "subprocess": false}
  },
  "interaction": {
    "history_enabled": false,
    "similarity_threshold": 0.55,
    "model_name": "stdlib-sequence-matcher",
    "model_debug": false
  },
  "maintenance": {
    "remote_manifest_url": null,
    "remote_manifest_sha256": null,
    "local_manifest_path": "project.manifest.json",
    "minimum_python": "3.11",
    "recommended_python": "3.11",
    "sysinternals_enabled": true,
    "sysinternals_download_url": "https://download.sysinternals.com/files/SysinternalsSuite.zip"
  },
  "logging": {
    "level": "INFO",
    "console_enabled": true,
    "color_enabled": true,
    "file_enabled": true,
    "maximum_bytes": 4194304,
    "delete_previous": false,
    "retention_days": 30
  },
  "collectors": {
    "core.network.bandwidth_sample": {"sample_count": 3, "interval_seconds": 1},
    "core.packet.packet_capture": {"packet_count": 100, "timeout_seconds": 10, "retry_window_seconds": 0, "interface": "Ethernet"},
    "core.filesystem.system_drive_tree": {"max_entries": 5000, "max_depth": 4},
    "core.filesystem.system_drive_listing": {"max_entries": 5000, "max_depth": 4},
    "core.filesystem.sensitive_file_inventory": {"root": "C:\\Evidence", "max_directories": 1000, "max_matches": 100},
    "core.process.memory_map": {"max_regions": 5000, "output_limit_bytes": 67108864, "disk_safety_margin_bytes": 104857600, "dump_directory": "memory_maps"}
  }
}
```

Validate a changed file before collecting:

```powershell
python -m logicytics preflight --config .\logicytics.yaml
```
