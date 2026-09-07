# Configuration

Logicytics has one authoritative user configuration: [`logicytics.yaml`](../logicytics.yaml) at the application root. The installer creates it when needed, repair workflows may update it, and users may edit it directly. The parser accepts a strict, mapping-only YAML subset; JSON is valid only where it is also valid YAML.

Unknown keys, duplicate keys, unsafe paths, non-finite numbers, unsupported values, and files larger than 2 MiB are rejected before planning. Invocation-only choices such as profiles, collector selection, capabilities, authorization, scheduling, reruns, package policy, and power actions belong to `RunRequest`, not this file.

## Root sections

The root fields are `schema_version`, `runtime`, `interaction`, `maintenance`, `logging`, and `collectors`. The current schema is version `4`.

| Section | Fields |
| --- | --- |
| `runtime` | `output_root`, `default_max_workers`, `maximum_workers`, `package_completed_runs`, `maximum_run_output_bytes` |
| `interaction` | `history_enabled`, `similarity_threshold`, `model_name`, `model_debug` |
| `maintenance` | `remote_manifest_url`, `remote_manifest_sha256`, `local_manifest_path`, `minimum_python`, `recommended_python`, `sysinternals_enabled`, `sysinternals_download_url` |
| `logging` | `level`, `console_enabled`, `color_enabled`, `file_enabled`, `maximum_bytes`, `delete_previous`, `retention_days` |

`runtime.output_root` and `maintenance.local_manifest_path` must stay inside the project when relative. Worker, output, log, and retention values are bounded by the parser. Remote manifests require a matching HTTPS URL and lowercase SHA-256 digest. Sysinternals is enabled by default; set `maintenance.sysinternals_enabled: false` to disable discovery, download, and extraction.

Application events use the same structured data for both sinks. File rows follow `TIME | SEVERITY | CODE SOURCE | MESSAGE`; console-only command results are rendered as grey ASCII boxes and do not pollute the application event log. `logging.console_enabled`, `logging.color_enabled`, and `logging.file_enabled` control those sinks independently.

## Collector settings

The `collectors` mapping is keyed by a validated collector ID. Shipped collectors reject settings they do not declare. Extension IDs may define their own settings. The shipped configurable fields are:

| Collector | Fields and bounds |
| --- | --- |
| `core.network.bandwidth_sample` | `sample_count` 1–10; `interval_seconds` 0.1–60 |
| `core.packet.packet_capture` | `packet_count` 1–10000; `timeout_seconds` 1–60; `retry_window_seconds` 0–60; `interface` text |
| `core.filesystem.system_drive_tree` | `max_entries` 1–50000; `max_depth` 1–32 |
| `core.filesystem.system_drive_listing` | `max_entries` 1–50000; `max_depth` 1–32 |
| `core.filesystem.sensitive_file_inventory` | `root` absolute path; `max_directories` 1–50000; `max_matches` 1–5000 |
| `core.process.memory_map` | `max_regions` 1–100000; `output_limit_bytes` 1024–67108864; `disk_safety_margin_bytes` 0–68719476736; `dump_directory` relative workspace path |

## Complete example

```json
{
  "schema_version": 4,
  "runtime": {
    "output_root": "output/data",
    "default_max_workers": 4,
    "maximum_workers": 16,
    "package_completed_runs": true,
    "maximum_run_output_bytes": 4294967296
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
    "core.network.bandwidth_sample": {
      "sample_count": 3,
      "interval_seconds": 1
    },
    "core.packet.packet_capture": {
      "packet_count": 100,
      "timeout_seconds": 10,
      "retry_window_seconds": 0,
      "interface": "Ethernet"
    },
    "core.filesystem.system_drive_tree": {
      "max_entries": 5000,
      "max_depth": 4
    },
    "core.filesystem.system_drive_listing": {
      "max_entries": 5000,
      "max_depth": 4
    },
    "core.filesystem.sensitive_file_inventory": {
      "root": "C:\\Evidence",
      "max_directories": 1000,
      "max_matches": 100
    },
    "core.process.memory_map": {
      "max_regions": 5000,
      "output_limit_bytes": 67108864,
      "disk_safety_margin_bytes": 104857600,
      "dump_directory": "memory_maps"
    }
  }
}
```

The parser-backed tests validate this example and the field inventory. For installation, migration guidance, and troubleshooting, see the [Logicytics Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki).
