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
  `core.memory.memory_snapshot` and `core.storage.logical_drives`.

## Layout

```text
core/
  system/
    system_info.py       # shipped collectors
plugins/                 # user-created collectors, validated separately
logicytics/              # engine, contracts, planning, runtime, and packaging
tests/                    # core integration tests
ACCESS/                  # ignored generated runs and packages
```

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
acknowledgement and writes results below `ACCESS/RUNS/`, with ZIP packages and
SHA-256 sidecars under `ACCESS/PACKAGES/`.

Collectors that request additional access must be explicitly approved, for
example:

```powershell
python -m logicytics run --profile standard --acknowledge-authorization `
  --allow-capability filesystem_read
```

## Collector rules

Core collectors must live at `core/<specialty>/<collector_name>.py`. Each module
must expose exactly one public class named `<CollectorName>Collector`, inherit
from `CoreCollector`, provide documented and typed `metadata`, `validate`,
`collect`, and `cleanup` methods, and use a matching ID/specialty. Plugins follow
the equivalent `PluginCollector` contract.

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
