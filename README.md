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
  data/
    run-<id>/            # isolated run artifacts, manifest, and logs/
    run-<utc>-run-<id>.zip         # action/timestamp-named evidence package
    run-<utc>-run-<id>.zip.sha256  # matching SHA-256 package sidecar
```

Each verified ZIP uses a versioned, non-overlapping layout. Registered source
material is stored under `evidence/raw/`, generated collector output under
`evidence/derived/`, the human summary under `reports/`, structured diagnostics
under `logs/`, the per-artifact SHA-256 catalog under `hashes/`, and the
machine-readable run manifest under `metadata/`. The manifest records both the
layout version and every section path; the package SHA-256 remains an external
sidecar because an archive cannot contain its own final digest.

## Requirements

- Python 3.11 or later.
- Windows for the currently shipped collector and future Windows integrations.

The v4 core uses only the Python standard library. A collector may later declare
its own dependency and capability requirements.

## Commands

Run these from the repository root:

```powershell
python -m logicytics preflight
python -m logicytics plan --profile standard
python -m logicytics run --profile standard --acknowledge-authorization
```

`preflight` reports valid and quarantined collectors. `plan` resolves the exact
collector order without collecting data. `run` requires an explicit authorization
acknowledgement and writes each run below `output/data/run-<id>/`. Its verified
ZIP package and SHA-256 sidecar use the action, UTC request timestamp, and run
ID, and are written alongside that run directory under the same `output/data/`
root; collector workspaces and `logs/` remain run-scoped. Explicit reruns use a
`rerun-<utc>-run-<id>.zip` package identity.

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

## Public Python API

Importing `logicytics` does not start collection, create output directories, or
load the runtime supervisor. Application services are loaded only when their
public names are requested:

```python
from pathlib import Path

from logicytics import RunRequest, load_configuration, plan_run, query_run, read_artifact, run_collection

root = Path.cwd()
configuration = load_configuration(root)
request = RunRequest(max_workers=1, acknowledge_authorization=True)
plan = plan_run(root, request, configuration=configuration)
outcome = run_collection(root, request, configuration=configuration)
snapshot = query_run(root, outcome.manifest.run_id, configuration=configuration)
contents = read_artifact(root, snapshot.run_id, snapshot.artifacts[0].id, configuration=configuration)
```

Planning enforces strict core/plugin preflight and configured worker bounds
without creating evidence. Run queries validate the manifest, collector-owned
artifact catalog, and configured output boundaries. Artifact reads are bounded
(16 MiB by default, never more than 64 MiB) and verify the exact registered
bytes against their manifest SHA-256 before returning evidence.

## Configuration

An optional project-root `logicytics.json` uses schema version `4` and separates
engine-wide `runtime` options from per-collector `collectors` settings:

```json
{
  "schema_version": 4,
  "runtime": {"default_max_workers": 4, "maximum_workers": 8},
  "collectors": {
    "core.filesystem.system_drive_tree": {"max_entries": 5000, "max_depth": 12},
    "core.process.memory_map": {"max_regions": 5000, "dump_directory": "memory_maps"}
  }
}
```

Shipped filesystem, network, packet, sensitive-inventory, and metadata-only
memory-map collector options are strictly typed and bounded before planning.
Unknown engine fields, duplicate JSON keys, unsafe workspace paths, and invalid
collector IDs fail closed; plugin-owned setting fields remain extensible.

Schema-version `3` JSON files are migrated in memory without modifying the source
file. Legacy `workers`/`worker_count`, `max_workers`, `output_root`, and
`collector_settings` aliases become their v4 runtime or collector equivalents;
the old `ACCESS/RUNS` default becomes `output/data`. Ambiguous aliases, unsafe
values, unsupported versions, and implicit plugin-selection fields are rejected.
Run manifests record `migrated_from_schema: 3` when this compatibility path runs.

## Collector rules

Core collectors must live at `core/<specialty>/<collector_name>.py`. Each module
must expose exactly one public class named `<CollectorName>Collector`, inherit
from `CoreCollector`, provide documented and typed `metadata`, `validate`,
`collect`, and `cleanup` methods, and use a matching ID/specialty. Plugins follow
the equivalent `PluginCollector` contract.

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
