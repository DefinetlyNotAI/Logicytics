# Logicytics v4

Logicytics v4 is a Windows-focused, run-oriented evidence collection framework.
It is being rebuilt from a clean architecture around an explicit pipeline:

`request -> validated plan -> isolated collectors -> registered artifacts -> manifest -> package`

The project is currently an early v4 implementation. The engine and the first
safe system collector are available; the broader collection catalog is being
rebuilt incrementally.

## Current capabilities

- Typed `core/` and `plugins/` collector contracts.
- Strict preflight validation before a collector can run.
- One process and private workspace per collector.
- Explicit capability approval and collection authorization.
- Structured JSONL engine and collector logs.
- Cooperative cancellation across long walks, captures, queries, memory enumeration,
  and source-copy loops, with unpublished sensitive bytes removed.
- Artifact registration with workspace boundaries, output limits, SHA-256 hashes,
  a run manifest, ZIP package, and package hash.
- Shipped collectors: `core.system.system_info` and the capability-gated
  `core.process.running_processes` and `core.network.network_identity`, plus
  `core.memory.memory_snapshot`, `core.storage.logical_drives`, and deep-profile
  `core.hardware.windows_features`, `core.network.network_adapters`, and
  `core.hardware.battery_status`,
  `core.bluetooth.paired_devices`, plus `core.usb.usb_storage_inventory`.
  Deep-profile inventory also includes `core.bluetooth.bluetooth_addresses`.
  Deep-profile collection also includes `core.system.system_details`,
  `core.system.bios_info`,
  `core.system.operating_system`,
  `core.system.computer_system`,
  `core.system.session_snapshot`,
  `core.system.system_diagnostics`,
  `core.process.detailed_processes`,
  `core.process.process_memory`,
  `core.registry.installed_applications`,
  `core.registry.startup_applications`,
  `core.filesystem.system_drive_tree`,
  `core.filesystem.system_drive_listing`,
  `core.network.active_connections`,
  `core.network.adapter_statistics`,
  `core.network.bandwidth_sample`,
  `core.network.network_interfaces`,
  `core.network.connection_processes`,
  `core.wireless.wifi_profiles`,
  `core.event_log.application_events`,
  `core.event_log.security_events`,
  `core.system.installed_drivers`, and
  `core.system.installed_updates`,
  `core.system.windows_services`,
  `core.system.group_policy`, `core.event_log.system_events`, and
  `core.network.arp_cache`, `core.network.routing_table`, and
  `core.storage.physical_disks`, and `core.storage.mounted_volumes`.
  `core.storage.volume_details` adds filesystem and label metadata in the deep profile.
  `core.encryption.bitlocker_status` is also available in the deep profile.
  `core.encryption.bitlocker_volumes` provides the PowerShell volume view.

## Layout

```text
core/
  system/
    system_info.py       # shipped collectors
plugins/                 # user-created collectors, validated separately
logicytics/              # engine, contracts, planning, runtime, and packaging
tests/                   # core integration tests
output/                  # ignored generated evidence and diagnostics
  logs/
    Logicytics.log       # bounded redacted application log
    debug/               # debug-action diagnostics
    performance/         # stable performance-report location
  data/
    zip/                 # stable package-output location
    hashes/              # stable package-hash location
    run-<id>/            # one self-contained run output tree
      artifacts/         # collector-owned registered evidence store
      collectors/        # private collector workspaces and event channels
      logs/              # engine and performance diagnostics
      reports/           # human-readable run summary
      packages/
        run-<utc>-run-<id>.zip         # verified evidence package
      hashes/
        artifacts.sha256               # packaged-artifact checksum catalog
        run-<utc>-run-<id>.zip.sha256  # matching package SHA-256 sidecar
```

Each verified ZIP uses a versioned, non-overlapping layout. Registered source
material is stored under `evidence/raw/`, generated collector output under
`evidence/derived/`, the human summary under `reports/`, structured diagnostics
under `logs/`, the per-artifact SHA-256 catalog under `hashes/`, and the
machine-readable run manifest under `metadata/`. Manifest schema version `1`
records run identity, action, status, timestamps, the redacted request and
configuration, host facts, resolved plan and fingerprint, collector lifecycle
records, the artifact catalog, errors, and package metadata. Readers fail closed
on absent, non-integer, or unsupported schema versions. The manifest separately
records package layout version `1.0` and every section path; the package SHA-256
remains an external sidecar because an archive cannot contain its own final digest.
Every shipped collector's stable filename or path pattern, MIME format, canonical
package location, and retention behavior is defined in [OUTPUTS.md](OUTPUTS.md)
and enforced by the isolated artifact writer.
Supported legacy inputs, their exact v4 replacements, and removed internal shims
are documented in [MIGRATION.md](MIGRATION.md).
The executable evidence for every supported mode, maintenance action, power
action, cancellation, and permission flow is indexed in [FLOW_MATRIX.md](FLOW_MATRIX.md).

## Requirements

- Python 3.11 or later.
- Windows for the currently shipped collector and future Windows integrations.

The v4 core uses only the Python standard library. A collector may later declare
its own dependency and capability requirements.

Collectors reach Windows through `logicytics.platform_adapters`: guarded process
execution covers PowerShell, WMI/CIM, WMIC, and command tools; dedicated adapters
cover registry reads, host filesystem discovery and staging, socket access, and
Win32 library loading. Privilege and UAC inspection remains in the application
environment boundary. Core collectors may use these mockable service seams and
`logicytics.contracts`, but cannot import orchestration services or bypass the
artifact writer for publication.

Logging uses path-keyed, process-local singleton factories. The CLI and supervisor
share one application logger per canonical `Logicytics.log`; engine and collector
workers share the same event-logger factory while receiving distinct run-owned
JSONL channels. Logger objects are intentionally not shared across process
boundaries, preserving collector isolation and avoiding cross-process mutable state.

## Commands

Run these from the repository root:

```powershell
python -m logicytics preflight
python -m logicytics debug
python -m logicytics dev
python -m logicytics update --launch-action debug --new-window
python -m logicytics collector core.system.system_info --acknowledge-authorization
python -m logicytics plan --profile standard
python -m logicytics run --profile standard --acknowledge-authorization
```

`preflight` reports valid and quarantined collectors. `plan` resolves the exact
collector order without collecting data. `run` requires an explicit authorization
acknowledgement and writes each run below `output/data/run-<id>/`. Its verified
ZIP package and SHA-256 sidecar use the action, UTC request timestamp, and run
ID, and are written into that run's `packages/` and `hashes/` directories;
collector workspaces and `logs/` remain run-scoped. Explicit reruns use a
`rerun-<utc>-run-<id>.zip` package identity.
`collector <exact-id>` runs only that validated collector and any declared
dependencies through the same authorization, isolation, artifact, manifest,
packaging, and cleanup pipeline; it never adds unrelated profile members.
Add `--interactive` to `run` when a transient command window should pause on the
final status; noninteractive and automated runs never prompt.

`debug` writes redacted environment, preflight, Sysinternals, Python-support,
version, and configured file-integrity diagnostics to
`output/logs/debug/debug.json`. `dev` compares the repository with the optional
integrity manifest and checks source organization without running collectors.
Manifest changes are opt-in: use `dev --write-manifest --next-version 4.1.0`, or
`dev --interactive` to review status markers and confirm the write.
For a checkout still using the historical INI fallback, that same confirmed dev
action atomically updates only `[System Settings]` `version` and `files` in
`CODE/config.ini`, preserving comments and unrelated sections. Modern checkouts
continue to write the hashed JSON integrity manifest.
On Windows, an update workflow can explicitly open a fresh visible console for
one allowlisted maintenance action with `update --launch-action preflight
--new-window`. Both options are required together; ordinary update checks never
launch another process.

Collectors that request additional access must be explicitly approved, for
example:

```powershell
python -m logicytics run --profile standard --acknowledge-authorization `
  --allow-capability filesystem_read
```

## Collection profiles

- `minimal`: only collectors that explicitly declare essential local inventory.
- `standard`: shipped core collectors that explicitly declare standard membership.
- `deep`: extended collector-declared inventory; sensitive capabilities still
  require explicit approval.
- `offline`: collector-declared local inventory; network and packet-capture
  capabilities remain forbidden even when explicitly included and approved.

Plugin collectors never join any profile unless explicitly included or enabled
with `--plugins`. Unknown profile names are rejected before collection begins.
Python MODs are workspace-confined unless the user explicitly approves the
`filesystem_write` capability. Native PowerShell, batch, and executable MODs
must declare and receive that approval because their host writes cannot be
enforced through Python audit hooks.

## Execution modes

`python -m logicytics --modes` runs strict collector preflight and prints the
authoritative, versioned machine-readable mode matrix. Its `modes` rows contain
the exact validated `collector_ids` selected by each mode. Its `collectors` rows
represent every discovered core, plugin, and MOD collector exactly once with its
mode assignments, validation state, execution type, and either `manual_only` or
quarantine status. New integrations should use `run --mode <name>`; the
historical flags remain exact compatibility aliases.

| Mode | Profile | Scheduling | Legacy alias | Additional behavior |
| --- | --- | --- | --- | --- |
| `standard` | standard | sequential | `--default` | deterministic standard run |
| `balanced` | standard | bounded parallel | `--threaded` | configured worker pool |
| `quick` | minimal | configured | `--minimal` | essential inventory |
| `thorough` | deep | configured | `--depth` | extended slower inventory |
| `offline` | offline | configured | none | forbids network access |
| `extensions` | standard | configured | `--modded` | enables declared MODS |
| `non-python` | standard | configured | `--nopy` | only non-Python MODS payloads |
| `performance` | standard | sequential | `--performance-check` | duration report |

Explicit `--profile`, `--include`, and `--exclude` remain available for advanced
planning. A named mode cannot be combined with a different explicit profile or
with its contradictory worker strategy.

## Public Python API

Importing `logicytics` does not start collection, create output directories, or
load the runtime supervisor. Application services are loaded only when their
public names are requested:

```python
from pathlib import Path

from logicytics import RunRequest, load_configuration, open_artifact, plan_run, query_run, read_artifact, run_collection

root = Path.cwd()
configuration = load_configuration(root)
request = RunRequest(max_workers=1, acknowledge_authorization=True)
plan = plan_run(root, request, configuration=configuration)
outcome = run_collection(root, request, configuration=configuration)
snapshot = query_run(root, outcome.manifest.run_id, configuration=configuration)
for collector in snapshot.collectors:
    print(collector.collector_id, collector.status, collector.duration_seconds, collector.summary)
    if collector.failure is not None:
        print(collector.failure.operation, collector.failure.remediation)
contents = read_artifact(root, snapshot.run_id, snapshot.artifacts[0].id, configuration=configuration)
# Explicit user-facing actions may open the verified artifact with its Windows handler:
# open_artifact(root, snapshot.run_id, snapshot.artifacts[0].id, configuration=configuration)
```

Planning enforces strict core/plugin preflight and configured worker bounds
without creating evidence. Run queries validate the manifest, collector-owned
artifact catalog, per-collector timestamps, durations, summaries, errors,
actionable failure details, and configured output boundaries. The command-line
run summary prints the same redacted per-collector status, duration, summary,
and failure guidance. Artifact reads are bounded (16 MiB by default, never more
than 64 MiB) and verify the exact registered bytes against their manifest
SHA-256 before returning evidence.
`open_artifact` applies the same run ownership and hash checks with streaming I/O
before asking Windows to open the file through its associated application.

## Configuration

An optional project-root `logicytics.json` uses schema version `4`. Product
policy is split into immutable `runtime`, `interaction`, `maintenance`, and
`logging` sections. Profile membership remains in the typed collector registry,
per-collector options remain under `collectors`, and invocation-only overrides
such as explicit selections, worker strategy, authorization, and capabilities
remain in `RunRequest`; none of those runtime overrides mutate the loaded
configuration.

```json
{
  "schema_version": 4,
  "runtime": {"default_max_workers": 4, "maximum_workers": 8},
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
    "core.filesystem.system_drive_tree": {"max_entries": 5000, "max_depth": 12},
    "core.process.memory_map": {
      "max_regions": 5000,
      "output_limit_bytes": 67108864,
      "disk_safety_margin_bytes": 104857600,
      "dump_directory": "memory_maps"
    }
  }
}
```

Shipped filesystem, network, packet, sensitive-inventory, and metadata-only
memory-map collector options are strictly typed and bounded before planning.
The memory-map output limit, free-space safety margin, and dump directory are
collector-scoped; the directory must be relative to that collector's isolated
workspace.
Unknown engine fields, duplicate JSON keys, unsafe workspace paths, and invalid
collector IDs fail closed; plugin-owned setting fields remain extensible.

An optional remote integrity manifest requires both an HTTPS
`remote_manifest_url` and a pinned lowercase `remote_manifest_sha256`. The exact
downloaded bytes are authenticated before strict JSON parsing. Integrity
manifests can contain only a schema version, project version, and file hashes;
they cannot configure, enable, or add collectors, and collection never depends
on the remote endpoint.

Application logging supports `DEBUG`, `INTERNAL`, `INFO`, `WARNING`, `ERROR`,
`EXCEPTION`, and `CRITICAL`. Human-readable UTC rows are redacted before they
reach colored stderr or `output/logs/Logicytics.log`; per-run and per-collector
JSONL logs remain isolated in each run. The configured byte limit retains recent
complete rows, `delete_previous` removes the prior application log explicitly,
and `retention_days` removes expired rotated application logs.

Schema-version `3` JSON files are migrated in memory without modifying the source
file. Legacy `workers`/`worker_count`, `max_workers`, `output_root`, and
`collector_settings` aliases become their v4 runtime or collector equivalents;
the old `ACCESS/RUNS` default becomes `output/data`. Ambiguous aliases, unsafe
values, unsupported versions, and implicit plugin-selection fields are rejected.
Run manifests record `migrated_from_schema: 3` when this compatibility path runs.
When `logicytics.json` is absent, the historical `CODE/config.ini` is also loaded
as a read-only migration source. Its general logging/worker/history options and
Flag, DumpMemory, NetWorkPsutil, and PacketSniffer sections are converted into
the same bounded typed v4 sections. Modern JSON always takes precedence; an INI
larger than 2 MiB, malformed values, and unsafe limits fail before planning.

Deep-profile compatibility includes `core.integration.legacy_code_outputs`.
When a historical `CODE/` directory exists, it imports only bounded generated
text, CSV, JSON, XML, HTML, DOT, SVG, registry, event-log, log, and ZIP evidence
through the normal artifact catalog. Python/PowerShell/batch source, executables,
models, `config.ini`, caches, virtual environments, and library internals are
never staged or packaged. Clean v4 checkouts skip this collector because they do
not need the legacy directory.

## Collector rules

Core collectors must live at `core/<specialty>/<collector_name>.py`. Each module
must expose exactly one public class named `<CollectorName>Collector`, inherit
from `CoreCollector`, provide documented and typed `metadata`, `validate`,
`collect`, and `cleanup` methods, and use a matching ID/specialty. Plugins follow
the equivalent `PluginCollector` contract.

Every collector explicitly declares the exact MIME types it can register.

## Migration status

Historical Python, PowerShell, batch, and executable scripts enter v4 only as
sidecar-declared MOD compatibility collectors; they never bypass preflight,
planning, isolation, capability approval, or artifact registration. Historical
generated CODE evidence has its own bounded compatibility collector. Every
shipped core collector writes beneath `CollectorContext.workspace`, publishes
through `context.artifacts.register_file`, and avoids mutable `global`/`nonlocal`
state. The run artifact tree is canonical even when a compatibility collector
preserves a legacy evidence filename or nested relative path.
Preflight statically compares literal `register_file(..., media_type=...)` calls
with that declaration, and isolated workers enforce the declaration again on
the returned artifact catalog. The collector folder, ID, class, description,
source category, and primary specialty therefore remain one coherent contract.

Collectors may import the public collector contracts, but not CLI, planning,
runtime, packaging, configuration, or public application-control services. Even
with an approved subprocess capability, workers cannot invoke Logicytics, another
collector, repository/package managers, or system reboot/shutdown commands.
Generated output is registered as derived evidence by default. Collectors that
preserve source material must explicitly register it with
`evidence_kind=EvidenceKind.RAW`; its manifest record and ZIP section retain that
classification across the isolated worker boundary.

Malformed core collectors block a run. Malformed unselected plugins are
quarantined and listed by preflight; malformed explicitly selected plugins block
the requested run. See [TODO.md](TODO.md) for the complete v4 recreation plan.

## Verification

```powershell
python -m unittest discover -v
python -m compileall -q logicytics core tests
```

## Authorization

Use Logicytics only on systems and data you are authorized to inspect. The v4
engine requires an explicit acknowledgement before it starts selected collectors.

## License

See [LICENSE](LICENSE).
