# Migrating to Logicytics v4

The v4 engine has one canonical data model: a typed request produces an isolated
run tree, registered artifacts, a manifest, and (unless disabled) a verified ZIP
with a SHA-256 sidecar. Compatibility code may translate an old input into that
model, but it may not create a second execution or output pipeline.

## Supported compatibility bridges

These bridges are intentionally supported for the v4 release and are not
internal shims:

| Historical interface | v4 replacement | Retention policy |
| --- | --- | --- |
| `--default` | `--mode standard` | Supported v4 CLI alias |
| `--threaded` | `--mode balanced` | Supported v4 CLI alias |
| `--minimal` | `--mode quick` | Supported v4 CLI alias |
| `--depth` | `--mode thorough` | Supported v4 CLI alias |
| `--nopy` | `--mode non-python` | Supported v4 CLI alias |
| `--modded` | `--mode extensions` | Supported v4 CLI alias |
| `--performance-check` | `--mode performance` | Supported v4 CLI alias |
| schema-3 JSON settings | schema-4 `logicytics.json` | Read and migrated in memory |
| `CODE/config.ini` | schema-4 `logicytics.json` | Read-only fallback when JSON is absent |
| generated evidence in `CODE/` | `core.integration.legacy_code_outputs` | Deep-profile import only |
| `.py`, `.ps1`, `.bat`, `.exe` files in `MODS/` | typed plugin collector | Isolated compatibility adapter |

Legacy inputs never enable plugins or MODS implicitly. Ambiguous settings,
duplicate keys, unsafe paths, undeclared capabilities, and unsupported fields
fail closed. Migration provenance is recorded in the run manifest.

## Output migration

There are no global `ACCESS/`, `RUNS/`, `LOGS/`, or `PACKAGES/` compatibility
copies. Their v4 replacements are respectively the run tree under
`output/data/run-<id>/`, `output/logs/`, `output/data/zip/`, and
`output/data/hashes/`. The run artifact tree and verified package layout in
[OUTPUTS.md](OUTPUTS.md) are canonical.

The legacy CODE evidence collector preserves a historical file's relative name
only inside its collector-owned `legacy_code/` subtree. This is the sole legacy
output-name bridge because those names identify user-generated evidence. It
excludes source code, configuration, manifests, models, caches, virtual
environments, and library internals. No compatibility copy is written beside the
canonical artifact.

## Removed internal shims

The v4 public API replaces process-global run state, current-directory lookup,
direct collector `exit()` calls, shared cleanup, unregistered output scanning,
and direct packaging of `CODE/`. Collector code must use `CollectorContext`,
`ArtifactWriter`, typed results, and the supervisor lifecycle. These obsolete
internal behaviors are not exposed as aliases and must not be reintroduced.

New integrations should implement the v4 collector contract directly. A bridge
is accepted only when it has a named owner, bounded input schema, explicit
replacement, isolation tests, and a documented removal or support policy.
