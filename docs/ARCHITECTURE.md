# Repository architecture

## Top-level folders

| Folder        | Responsibility                                                                                                                    |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `logicytics/` | Public package, contracts, CLI, configuration, discovery, planning, runtime, logging, manifests, packaging, and platform adapters |
| `core/`       | Shipped read-only collector implementations grouped by specialty                                                                  |
| `plugins/`    | Opt-in user-owned `PluginCollector` implementations                                                                               |
| `MODS/`       | Opt-in Python scripts enabled by metadata sidecars                                                                                |
| `tests/`      | Unit, contract, resilience, security, and Windows integration tests                                                               |
| `docs/`       | This user, operator, and developer manual                                                                                         |
| `output/`     | Local run output; do not commit evidence                                                                                          |
| `.github/`    | Contribution templates, security automation, and documentation publishing workflow                                                |

## Engine modules

- `contracts.py` defines enums, metadata, request, context, artifact, result, and collector interfaces.
- `configuration.py` loads and validates `logicytics.yaml`.
- `discovery.py` finds candidates, performs static checks, probes metadata, and manages the disposable validation cache.
- `planner.py` resolves profiles, selectors, dependencies, capabilities, ordering, and run fingerprints.
- `runtime.py` owns worker processes, quotas, cancellation, retry rules, progress, and terminal statuses.
- `artifacts.py` is the workspace-bound artifact writer.
- `manifest.py` writes the durable run record.
- `packaging.py` builds and verifies ZIP packages and hash sidecars.
- `api.py` exposes planning, collection, run queries, and bounded artifact reads.
- `cli/commands.py` maps command-line arguments to the public engine operations.

## Boundary rules

Collectors may depend on contracts and approved platform adapters. They should not reach into supervisor internals,
write outside their workspace, alter global configuration, or create unbounded output. The runtime owns cleanup and
process termination. The manifest is the source for later inspection; do not infer success from a console line alone.

## Core source layout

The core ID follows `core.<specialty>.<name>` and normally maps to `core/<specialty>/<name>.py`. A module contains one
collector class with a PascalCase name ending in `Collector`, a `metadata()` class method, lifecycle methods, and no
import-time collection. The catalog in [Core Collectors](CORE_COLLECTORS.md) is grouped by the live source tree.
