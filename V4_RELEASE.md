# Logicytics 4.0.0 release

Release date: 2026-08-31

Logicytics 4.0.0 is the complete recreation of the feature surface defined in
[TODO.md](TODO.md). It preserves the supported Windows system-data collection,
interaction, maintenance, extension, output, and packaging capabilities while
replacing the historical shared-script design with a typed, run-oriented,
isolated architecture.

The repository implementation, versioned documentation, and separate
[GitHub wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki) are
release-complete. The wiki's v4 user, contributor, configuration, evidence,
extension, migration, security, and troubleshooting pages were synchronized in
wiki commit `eaf7628`.

## Release identity

- Package version: `4.0.0`.
- Collector contract: `4.0`.
- Configuration schema: `4` with bounded schema-3 JSON and `CODE/config.ini`
  migration.
- Run manifest schema: `1`.
- Package layout: `1.0`.
- Runtime: Python 3.11 or later on Windows; standard library only for the core.

No v3 engine path remains. Supported legacy inputs are translations into the v4
request, collector, artifact, and manifest contracts described in
[MIGRATION.md](MIGRATION.md).

## Preserved feature surface

The shipped catalog contains 66 independently runnable core collectors. Their
collection and reporting surface covers `system`, `hardware`, `process`,
`memory`, `filesystem`, `network`, `packet`, `wireless`, `bluetooth`, `usb`,
`browser`, `registry`, `event_log`, `storage`, `encryption`, `media`, `ssh`,
`diagnostics`, `reporting`, and `integration`. Reporting is owned by the run
manifest, summaries, performance results, interaction statistics, and package
catalog rather than a standalone evidence collector.

Windows access includes bounded PowerShell, CIM/WMI, optional WMIC, registry,
event-log, BitLocker, network, filesystem, process, Win32 privilege, and optional
Sysinternals integrations behind mockable platform adapters. Optional host
features produce explicit availability or skip results and never stop unrelated
collectors.

The canonical profiles are `minimal`, `standard`, `deep`, and `offline`. The
canonical modes are `standard`, `balanced`, `quick`, `thorough`, `offline`,
`extensions`, `non-python`, and `performance`; all historical mode flags remain
validated aliases. The machine-readable `python -m logicytics --modes` output is
the authoritative collector inclusion matrix.

Core evidence contracts include `application/json`, `application/octet-stream`,
`application/xml`, `application/zip`, `text/csv`, `text/html`, `text/plain`, and
`text/vnd.graphviz`. [OUTPUTS.md](OUTPUTS.md) defines every stable workspace
pattern, package path, format, evidence kind, and retained-with-run policy.

Persistent settings cover runtime output/concurrency/package limits, local
interaction/history policy, authenticated maintenance manifests, application
logging, and bounded collector-specific fields. [CONFIGURATION.md](CONFIGURATION.md)
is the complete parser-backed schema reference. Profiles, explicit selections,
extensions, capabilities, authorization, scheduling, reruns, output policy, and
power actions remain invocation-only.

Plugins and MODs remain opt-in extension points. Python, PowerShell, batch, and
executable MOD payloads require sidecars and use the same preflight, planning,
capability, isolation, artifact, failure, and packaging boundaries as core
collectors. Historical generated `CODE` evidence has one bounded compatibility
collector; source and executable material is excluded.

## Predictable and inspectable results

Each request resolves to a deterministic collector/dependency order and plan
fingerprint before collection. Every selected collector receives a private
process, workspace, event channel, cancellation marker, configuration snapshot,
and artifact writer. Terminal states are explicit: `succeeded`, `partial`,
`skipped`, `cancelled`, or `failed`.

Every run owns a UTC/request/run-ID package name, redacted manifest, human
summary, structured logs, artifact checksum catalog, verified ZIP, and external
ZIP SHA-256 sidecar under `output/data/run-<id>/`. A rerun records its parent and
uses a separate package identity. Manifest-only output is explicit. Finished
evidence remains with its run until the user removes that run; worker scratch
data is removed only after durable publication.

Artifact publication is workspace-confined, streamed where possible, size/file
bounded, MIME-checked, and SHA-256 cataloged. Packages contain registered evidence
and generated metadata only. Readers validate schema, run ownership, paths,
sizes, and hashes before returning or opening evidence.

## Authorization, failure, and cancellation

Collection cannot start without explicit authorization acknowledgement.
Sensitive, network, packet, browser, private-key, filesystem, subprocess,
registry, and elevation capabilities are declared per collector and require
approval. Elevated selection also requires a live administrator check.

One collector failure does not abort independent work, but its failure cannot be
hidden by a successful package. The manifest records operation, platform error,
remediation, retry safety, lifecycle timestamps, progress, worker exit, and
termination reason. Permission denial and optional feature absence remain visible
typed outcomes.

Long traversals, copies, queries, captures, and enumerations check cooperative
cancellation. The supervisor also enforces time, memory, output, artifact-count,
heartbeat, and process-tree boundaries. Cancelled runs finalize recoverable
partial evidence when possible and clean only their own temporary state.

## Verification evidence

The v4 release gates are:

```powershell
python -m unittest discover -v
python -m compileall -q logicytics core tests
python -m logicytics preflight
python -m unittest tests.test_windows_integration -v
git diff --check
```

The suite statically and dynamically validates all 66 core collectors, direct and
orchestrated execution, lifecycle/cancellation behavior, mocked Windows failures,
live Windows boundaries, stable outputs, golden structured bytes, profiles and
modes, extension isolation, API reads, configuration migration, run/package/hash
integrity, and the complete flow matrix in [FLOW_MATRIX.md](FLOW_MATRIX.md).

[FEATURE_STATUS.md](FEATURE_STATUS.md) assigns every TODO section an owner,
status, and implementation/test evidence. The TODO checkboxes remain the
authoritative item-level completion record.

## Upgrade and support

Read [MIGRATION.md](MIGRATION.md) before reusing a v3 configuration, legacy flag,
MOD, or generated `CODE` output. Read [README.md](README.md) for installation and
operation, [CONTRIBUTING.md](CONTRIBUTING.md) for change requirements, and
[SECURITY.md](SECURITY.md) for supported versions and vulnerability reporting.
