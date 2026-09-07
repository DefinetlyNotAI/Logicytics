# v4 output contract

This document defines the stable names, formats, locations, and retention rules
for every shipped collector. `logicytics.output_contracts.core_output_contract`
is the executable source of truth and the worker rejects an artifact whose
relative path or MIME type is outside that contract.

## Canonical paths

For a collector ID `core.<category>.<name>`, a fixed single-file output is named
`<name><suffix>` in its private workspace and stored in the run catalog as
`artifacts/core_<category>_<name>/<name><suffix>`. In a verified ZIP it is
`evidence/<kind>/core_<category>_<name>/<name><suffix>`, where `<kind>` is `raw`
for copied source evidence and `derived` for generated reports.

The suffix is fixed by the declared MIME type:

| MIME type           | Suffix  |
|---------------------|---------|
| `application/json`  | `.json` |
| `application/xml`   | `.xml`  |
| `application/zip`   | `.zip`  |
| `text/csv`          | `.csv`  |
| `text/html`         | `.html` |
| `text/plain`        | `.txt`  |
| `text/vnd.graphviz` | `.dot`  |

This rule covers every shipped collector except the explicitly listed contracts
below. MIME types remain those declared by each collector's `CollectorMetadata`.

| Collector                                  | Stable workspace path pattern   |
|--------------------------------------------|---------------------------------|
| `core.bluetooth.paired_devices`            | `bluetooth_devices.json`        |
| `core.browser.browser_data_backup`         | `browser_data/*/*/*`            |
| `core.filesystem.sensitive_file_inventory` | `sensitive_file_inventory/**`   |
| `core.integration.legacy_code_outputs`     | `legacy_code/**`                |
| `core.media.media_backup`                  | `media_backup/**`               |
| `core.process.memory_map`                  | `**/memory_map.json`            |
| `core.registry.hklm_backup`                | `hklm_backup.reg`               |
| `core.system.windows_system_data_backup`   | `windows_system_data/*/*`       |
| `core.wireless.wifi_profile_keys`          | `wifi_profiles_with_keys/*.xml` |

`core.bluetooth.bluetooth_history` uses the fixed
`bluetooth_history.json` name. Its content carries `collected_at`; the run ID and
package timestamp preserve snapshot history without making the output contract
time-dependent.

## Retention

Collector workspaces are private staging locations and are removed after durable
publication. The run tree, manifest, package, package hash, reports, and logs are
retained together until the user explicitly removes that run. Schema v4 does not
silently expire completed evidence. Copied evidence has no separate lifetime:
raw and derived artifacts follow their owning run. A manifest-only run follows
the same retained-with-run rule but has no ZIP or package-hash sidecar.

Global application, debug, and performance logs live under `output/logs/` and
follow the logging limits described in the configuration documentation; they are
not collector outputs and never enter the evidence catalog.

## Logging presentation

The application log is a bounded, human-readable text log. Each row contains a
local timestamp with milliseconds, severity, source, and message; long messages
and fields continue on aligned rows. Console output uses the same redaction and
humanization rules, with colored severity markers and readable indented fields.
It never emits raw JSON or `key=value` field fragments. Structured run and
collector event logs are stored separately as JSONL for tooling and remain
redacted; they are not printed to the console.
