# Logicytics v4 configuration

Logicytics uses one strictly validated schema-version `4` JSON document. The
default location is `logicytics.json` in the project root; `--config PATH` selects
another file. The public Python API accepts the same path through
`load_configuration(project_root, config_path)` and the planning/run helpers.

Configuration is immutable product policy. One-off choices such as a profile,
mode, explicit collector include/exclude list, authorization acknowledgement,
approved capabilities, worker override, output policy, rerun source, and power
action belong to `RunRequest` or CLI flags and never rewrite this file.

## Complete example

Every section is optional except `schema_version`. Omitted values use the defaults
listed below.

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
    "local_manifest_path": "project.manifest.json",
    "minimum_python": "3.11",
    "recommended_python": "3.11"
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
      "interval_seconds": 1.0
    },
    "core.filesystem.system_drive_tree": {
      "max_entries": 5000,
      "max_depth": 12
    }
  }
}
```

## Source selection and precedence

1. An explicit `--config PATH` or API `config_path` is loaded.
2. Otherwise, project-root `logicytics.json` is loaded when present.
3. Otherwise, historical `CODE/config.ini` is translated in memory when present.
4. Otherwise, safe schema-v4 defaults are used.

The modern JSON file always wins over the INI fallback. Loading never modifies
the selected source. A configuration file larger than 2 MiB, malformed UTF-8 or
JSON, duplicate object key, non-finite number, wrong type, unknown engine field,
or unsupported schema fails before planning or collection.

## Root object

| Setting          | Required | Meaning                                                                         |
|------------------|----------|---------------------------------------------------------------------------------|
| `schema_version` | yes      | Integer `4`; schema `3` is accepted only through the documented migration path. |
| `runtime`        | no       | Engine-wide paths, concurrency, packaging, and aggregate-output limits.         |
| `interaction`    | no       | Local semantic matching and opt-in history behavior.                            |
| `maintenance`    | no       | Integrity manifests and supported Python policy.                                |
| `logging`        | no       | Human application-log and console presentation policy.                          |
| `collectors`     | no       | A map from exact collector IDs to isolated collector settings.                  |

Unknown root fields are rejected. In particular, configuration cannot implicitly
enable plugins, MODs, collectors, capabilities, authorization, or a power action.

## Runtime settings

| Setting                    | Default              | Accepted value                                                           | Effect                                                                                                 |
|----------------------------|----------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `output_root`              | `output/data`        | Non-empty absolute path, or a path resolved relative to the project root | Owns run directories, evidence, manifests, packages, and hashes.                                       |
| `default_max_workers`      | `4`                  | Integer from 1 to 64, not greater than `maximum_workers`                 | Worker count used when the request/mode does not force a strategy.                                     |
| `maximum_workers`          | `16`                 | Integer from 1 to 64, not less than `default_max_workers`                | Hard ceiling for CLI and API worker overrides.                                                         |
| `package_completed_runs`   | `true`               | Boolean                                                                  | When false, publish the run tree and manifest without creating a ZIP or ZIP hash.                      |
| `maximum_run_output_bytes` | `4294967296` (4 GiB) | Integer from 1 through `68719476736` (64 GiB)                            | Aggregate artifact budget shared by the run; later collectors fail or skip predictably when exhausted. |

Completed run evidence is retained until the user explicitly removes its run
directory. The logging retention setting below applies only to application logs.

## Interaction settings

| Setting                | Default                   | Accepted value                 | Effect                                                                     |
|------------------------|---------------------------|--------------------------------|----------------------------------------------------------------------------|
| `history_enabled`      | `false`                   | Boolean                        | Persist compressed local match history only when enabled.                  |
| `similarity_threshold` | `0.55`                    | Finite number from 0 through 1 | Minimum direct semantic-match score.                                       |
| `model_name`           | `stdlib-sequence-matcher` | Non-empty single-line string   | Label recorded with matcher results; v4 uses its standard-library matcher. |
| `model_debug`          | `false`                   | Boolean                        | Include matcher diagnostics independently of application debug logging.    |

History records remain local and contain the matched flag, input, accuracy,
timestamp, device name, and per-flag usage. Disabling history prevents new
records; it does not silently delete an existing history file.

## Maintenance settings

| Setting                  | Default                 | Accepted value                                                       | Effect                                                                               |
|--------------------------|-------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `remote_manifest_url`    | none                    | HTTPS URL, configured together with `remote_manifest_sha256`         | Optional integrity-manifest source used by maintenance actions, never by collection. |
| `remote_manifest_sha256` | none                    | Exactly 64 lowercase hexadecimal characters, configured with the URL | Authenticates the exact downloaded manifest bytes before parsing.                    |
| `local_manifest_path`    | `project.manifest.json` | Non-empty relative path without `..`                                 | Project-confined integrity manifest used by debug/developer checks.                  |
| `minimum_python`         | `3.11`                  | `major.minor`                                                        | Oldest supported interpreter reported by diagnostics.                                |
| `recommended_python`     | `3.11`                  | `major.minor`, not older than the minimum                            | Recommended interpreter reported by diagnostics.                                     |

An integrity manifest may contain only its schema version, project version, and
file hashes. It cannot configure execution. Remote retrieval is optional,
authenticated, bounded, and isolated from normal collection.

## Logging settings

| Setting           | Default           | Accepted value                                                                                 | Effect                                                                        |
|-------------------|-------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `level`           | `INFO`            | `DEBUG`, `INTERNAL`, `INFO`, `WARNING`, `ERROR`, `EXCEPTION`, or `CRITICAL` (case-insensitive) | Minimum application-log level.                                                |
| `console_enabled` | `true`            | Boolean                                                                                        | Write redacted human rows to stderr.                                          |
| `color_enabled`   | `true`            | Boolean                                                                                        | Add ANSI colors only when the console is a TTY.                               |
| `file_enabled`    | `true`            | Boolean                                                                                        | Write redacted human rows to `output/logs/Logicytics.log`.                    |
| `maximum_bytes`   | `4194304` (4 MiB) | Integer from 1024 through `67108864` (64 MiB)                                                  | Retain recent complete rows within the application-log byte cap.              |
| `delete_previous` | `false`           | Boolean                                                                                        | Explicitly remove the current application log when the logger starts.         |
| `retention_days`  | `30`              | Integer from 0 through 3650                                                                    | Remove expired rotated `Logicytics*.log` files; does not remove run evidence. |

Engine and collector JSONL event streams are always run-owned and separate from
the human application log. All channels redact sensitive fields before writing.

## Shipped collector settings

The engine validates the following core fields before planning. A collector also
revalidates settings inside its isolated worker. Unknown fields for these shipped
collectors are rejected.

| Collector                                  | Setting                    | Default                   | Accepted value                                            |
|--------------------------------------------|----------------------------|---------------------------|-----------------------------------------------------------|
| `core.network.bandwidth_sample`            | `sample_count`             | `3`                       | Integer 1-10.                                             |
| `core.network.bandwidth_sample`            | `interval_seconds`         | `1.0`                     | Finite number 0.1-60 seconds.                             |
| `core.packet.packet_capture`               | `packet_count`             | `100`                     | Integer 1-10000.                                          |
| `core.packet.packet_capture`               | `timeout_seconds`          | `10`                      | Finite number 1-60 seconds.                               |
| `core.packet.packet_capture`               | `retry_window_seconds`     | `0`                       | Finite number 0-60 seconds.                               |
| `core.packet.packet_capture`               | `interface`                | local host IPv4 address   | Non-empty text without a NUL byte.                        |
| `core.filesystem.system_drive_tree`        | `max_entries`              | `5000`                    | Integer 1-50000.                                          |
| `core.filesystem.system_drive_tree`        | `max_depth`                | `12`                      | Integer 1-32.                                             |
| `core.filesystem.system_drive_listing`     | `max_entries`              | `10000`                   | Integer 1-50000.                                          |
| `core.filesystem.system_drive_listing`     | `max_depth`                | `16`                      | Integer 1-32.                                             |
| `core.filesystem.sensitive_file_inventory` | `root`                     | Windows system-drive root | Non-empty absolute filesystem path.                       |
| `core.filesystem.sensitive_file_inventory` | `max_directories`          | `5000`                    | Integer 1-50000.                                          |
| `core.filesystem.sensitive_file_inventory` | `max_matches`              | `500`                     | Integer 1-5000.                                           |
| `core.process.memory_map`                  | `max_regions`              | `5000`                    | Integer 1-100000.                                         |
| `core.process.memory_map`                  | `output_limit_bytes`       | `67108864` (64 MiB)       | Integer 1024-67108864.                                    |
| `core.process.memory_map`                  | `disk_safety_margin_bytes` | `104857600` (100 MiB)     | Integer 0-68719476736.                                    |
| `core.process.memory_map`                  | `dump_directory`           | `memory_maps`             | Non-empty relative collector-workspace path without `..`. |

Shipped core collectors without a table above do not declare configurable fields,
so a non-empty settings object for one is rejected. Syntactically valid plugin-
and MOD-owned fields remain extensible for their extension contracts, but they do
not bypass capability, workspace, artifact, timeout, or output limits.

## Schema 3 and INI migration

Schema-version `3` JSON is translated in memory and recorded as
`migrated_from_schema: 3` in the run manifest. Root aliases `workers` and
`worker_count` become `runtime.default_max_workers`; `max_workers` becomes
`runtime.maximum_workers`; `output_root`, `package_completed_runs`, and
`maximum_run_output_bytes` move under `runtime`; and `collector_settings` becomes
`collectors`. Historical `ACCESS/RUNS` resolves to `output/data`. Conflicting old
and new names fail closed instead of choosing one silently.

The read-only `CODE/config.ini` bridge translates these historical fields:

- `Settings`: `max_workers`, `log_using_debug`, `delete_old_logs`, and
  `save_preferences`.
- `Flag Settings`: `accuracy_min`, `model_to_use`, and `model_debug`.
- `DumpMemory Settings`: `file_size_limit` and `file_size_safety`.
- `NetWorkPsutil Settings`: `sample_count` and `interval`.
- `PacketSniffer Settings`: `interface`, `packet_count`, `timeout`, and
  `max_retry_time`.

Use a schema-v4 JSON file for all new configuration. See [MIGRATION.md](MIGRATION.md)
for the broader v3-to-v4 runtime and extension compatibility boundary.

## Validation and inspection

Validate discovery, configuration, and host prerequisites without collecting:

```powershell
python -m logicytics --config .\logicytics.json preflight
python -m logicytics --config .\logicytics.json plan --profile standard
```

The run manifest records a redacted configuration snapshot and deterministic
configuration fingerprint. Use `query_run` to inspect the resolved plan,
collector results, artifacts, failures, and hashes after a run.
