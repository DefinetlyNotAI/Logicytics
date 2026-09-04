# Logicytics v4

Logicytics v4.0 is a complete recreation of the Windows system-data collection
and evidence-packaging tool around an explicit run-oriented pipeline:

`request -> validated plan -> isolated collectors -> registered artifacts -> manifest -> package`

The release ships 66 independently runnable core collectors, typed profiles and
modes, opt-in plugins and MODs, isolated workers, reproducible manifests and
packages, and bounded Windows integrations. See [V4_RELEASE.md](docs/V4_RELEASE.md)
for the release scope and [FEATURE_STATUS.md](docs/FEATURE_STATUS.md) for ownership
and verification evidence.

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
- Sixty-six shipped collectors cover all v4 specialties. The authoritative
  collector catalog and exact mode membership are emitted by
  `python -m logicytics --modes`; representative collectors include
  `core.system.system_info` and the capability-gated
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
  `core.encryption.bitlocker_volumes` provides the PowerShell volume view, while
  `core.system.wmic_inventory` preserves an optional bounded WMIC view when that
  Windows capability is installed.

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
    zip/                 # created compatibility landing directory; canonical packages are run-owned
    hashes/              # created compatibility landing directory; canonical hashes are run-owned
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
package location, and retention behavior is defined in [OUTPUTS.md](docs/OUTPUTS.md)
and enforced by the isolated artifact writer.
Supported legacy inputs, their exact v4 replacements, and removed internal shims
are documented in [MIGRATION.md](docs/MIGRATION.md).
The executable evidence for every supported mode, maintenance action, power
action, cancellation, and permission flow is indexed in [FLOW_MATRIX.md](docs/FLOW_MATRIX.md).

## Requirements

- Python 3.11 or later.
- Windows for core collection and live platform integration checks.

The v4 core uses only the Python standard library. Extension collectors declare
their own dependency and capability requirements.

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

## Installation

Clone the repository on Windows and use Python 3.11 or later. No third-party
runtime package is required:

```powershell
git clone https://github.com/DefinetlyNotAI/Logicytics.git
cd Logicytics
python -m logicytics preflight
```

`preflight` is read-only. It validates configuration, host prerequisites,
collector contracts, optional Windows features, and quarantine state before a
run. A missing optional WMIC, BitLocker, or Sysinternals capability is reported
without invalidating unrelated collectors.

## Quick start

Inspect the exact plan first, then acknowledge authorization explicitly:

```powershell
python -m logicytics plan --profile standard
python -m logicytics run --profile standard --acknowledge-authorization
```

The standard profile is intentionally bounded. Use `--mode thorough` for the full
deep catalog and approve every requested capability deliberately. Generated run
data appears under `output/data/run-<id>/`; the final summary prints collector
status, duration, failures, package path, and package hash.

## CLI reference

Run these from the repository root:

```powershell
python -m logicytics preflight
python -m logicytics debug
python -m logicytics dev
python -m logicytics update --launch-action debug --new-window
python -m logicytics --match "collect quickly"
python -m logicytics usage
python -m logicytics --modes
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

Important global and run controls:

| Control                         | Purpose                                                                            |
|---------------------------------|------------------------------------------------------------------------------------|
| `--config PATH`                 | Load a schema-v4 JSON or supported legacy INI configuration.                       |
| `--match TEXT`                  | Suggest the closest documented action; history persists only when configured.      |
| `--modes`                       | Print the versioned collector/mode matrix.                                         |
| `run --mode NAME`               | Select one canonical execution mode.                                               |
| `--include ID` / `--exclude ID` | Override profile membership by exact collector ID.                                 |
| `--plugins` / `--mods`          | Opt into discovered plugins or declared MODs.                                      |
| `--workers COUNT`               | Override bounded concurrency within the configured maximum.                        |
| `--sequential` / `--parallel`   | Select explicit scheduling without a legacy mode alias.                            |
| `--rerun-from MANIFEST`         | Rerun selected collectors while preserving parent-run provenance.                  |
| `--no-package`                  | Retain the manifest/run tree without a ZIP or ZIP hash.                            |
| `--reboot` / `--shutdown`       | Schedule one mutually exclusive power action only after verified packaging.        |
| `update --apply`                | Explicitly apply the configured Git update; an ordinary update check is read-only. |

Legacy `--default`, `--threaded`, `--minimal`, `--depth`, `--modded`, `--nopy`,
and `--performance-check` flags remain exact aliases. They cannot be combined
with a contradictory mode, profile, or scheduling override.

## Authorization, permissions, and status

Collection requires `--acknowledge-authorization`. Capabilities such as
filesystem reads, registry reads, subprocesses, browser data, private keys,
sensitive files, packet capture, network access, or elevation must also be
approved explicitly when selected metadata requests them. An elevated collector
is rejected during planning unless the process is actually administrative.

Collector outcomes are `succeeded`, `partial`, `skipped`, `cancelled`, or
`failed`. Expected absence and access denial are visible skips when no evidence
can be collected; unexpected command, parsing, timeout, output-limit, or worker
failures remain failures. One collector failure does not stop independent peers,
and the run cannot appear fully successful while a selected collector failed.

Ctrl+C requests cooperative cancellation. Long collectors check the run-owned
cancellation marker, the supervisor terminates unresponsive worker trees within
their timeout policy, and a recoverable partial manifest/package is finalized
when possible. Cleanup removes only worker-owned temporary data.

CLI exit status is `0` for the requested successful/read-only action, `1` for an
unsuccessful collection or maintenance result, and `2` for invalid arguments,
configuration, preflight, authorization, or other Logicytics contract errors.

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

| Mode          | Profile  | Scheduling       | Legacy alias          | Additional behavior           |
|---------------|----------|------------------|-----------------------|-------------------------------|
| `standard`    | standard | sequential       | `--default`           | deterministic standard run    |
| `balanced`    | standard | bounded parallel | `--threaded`          | configured worker pool        |
| `quick`       | minimal  | configured       | `--minimal`           | essential inventory           |
| `thorough`    | deep     | configured       | `--depth`             | extended slower inventory     |
| `offline`     | offline  | configured       | none                  | forbids network access        |
| `extensions`  | standard | configured       | `--modded`            | enables declared MODS         |
| `non-python`  | standard | configured       | `--nopy`              | only non-Python MODS payloads |
| `performance` | standard | sequential       | `--performance-check` | duration report               |

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

The complete field reference, limits, source precedence, migration rules, and a
parser-verified example are in [CONFIGURATION.md](docs/CONFIGURATION.md).

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
the requested run. See [FEATURE_STATUS.md](docs/FEATURE_STATUS.md) for implemented
ownership and [V4_RELEASE.md](docs/V4_RELEASE.md) for the released feature surface.

## Verification

```powershell
python -m unittest discover -v
python -m compileall -q logicytics core tests
```

On Windows, `python -m unittest tests.test_windows_integration -v` adds bounded
live privilege, registry, PowerShell, CIM/WMI, optional WMIC, event-log,
BitLocker, networking, Sysinternals, and artifact-publication probes.

## Troubleshooting

- **A collector is skipped:** inspect its manifest `summary`, `errors`, and
  `failure` fields. Install the named optional Windows feature, run from an
  authorized elevated console when required, or approve the exact capability.
- **Planning rejects capabilities:** rerun `plan` first and add only the listed
  `--allow-capability` values. Sensitive and elevated access is never inferred.
- **PowerShell execution policy is unavailable:** run `preflight`; confirm
  `powershell` is on `PATH`. The engine does not weaken machine policy.
- **WMIC is unavailable:** modern Windows may omit the optional WMIC feature.
  `core.system.wmic_inventory` skips explicitly; CIM/WMI collectors continue.
- **A plugin is quarantined:** use `preflight` to read its static/runtime errors.
  Unselected invalid plugins do not block core collection.
- **A run was interrupted:** inspect the printed run manifest under
  `output/data/run-<id>/metadata/`; cancellation preserves recoverable partial
  evidence and removes only private worker scratch data.
- **A package hash does not match:** do not trust or open the package. Use the
  adjacent `.sha256` sidecar and rerun collection; package verification fails
  closed on modified bytes.
- **Output is unexpectedly large:** lower collector-specific bounds in
  [CONFIGURATION.md](docs/CONFIGURATION.md) or use `minimal`/`standard`; the aggregate
  run budget prevents unbounded publication.

## Documentation

- [CONFIGURATION.md](docs/CONFIGURATION.md) — complete persistent settings contract.
- [OUTPUTS.md](docs/OUTPUTS.md) — every collector path, format, package path, and
  retention rule.
- [MODS.md](docs/MODS.md) — opt-in MOD extension contract.
- [MIGRATION.md](docs/MIGRATION.md) — supported v3 and legacy compatibility boundary.
- [FLOW_MATRIX.md](docs/FLOW_MATRIX.md) — executable evidence for every user flow.
- [FEATURE_STATUS.md](docs/FEATURE_STATUS.md) — TODO section owners and status.
- [V4_RELEASE.md](docs/V4_RELEASE.md) — final v4.0 recreation and release evidence.
- [Logicytics wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki) — v4 user,
  contributor, configuration, evidence, extension, and troubleshooting guides.
- [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — contributor and security policy.

## Authorization

Use Logicytics only on systems and data you are authorized to inspect. The v4
engine requires an explicit acknowledgement before it starts selected collectors.

## License

See [LICENSE](LICENSE).
