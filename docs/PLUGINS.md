# Plugin support

Plugins are the only supported extension mechanism. A plugin is a typed Python `PluginCollector` stored below `plugins/`; it is never selected by a normal core-only request. Script sidecars, batch, PowerShell, and executable extension paths are not supported.

## Enable a reviewed plugin

1. Read its source and declared metadata.
2. Run preflight with plugins enabled.
3. Build a plan and review the exact ID, capabilities, sensitivity, limits, and dependencies.
4. Run only with explicit authorization.

```powershell
.\.venv\Scripts\python.exe -m logicytics preflight --plugins
.\.venv\Scripts\python.exe -m logicytics plan --mode standard --plugins
.\.venv\Scripts\python.exe -m logicytics run --mode standard --plugins --acknowledge-authorization
```

For a sensitive plugin, prefer an exact `--include plugin.name` request and document why the capability set is authorized. A blocked capability always overrides a plugin declaration.

## Discovery and identity

Plugin modules live under `plugins/`. Each valid module exposes exactly one `PluginCollector` subclass. A plugin ID begins with `plugin.` and must match its file or folder ownership: a normal `plugins/example.py` exposes `plugin.example`; a `main.py` entry point uses its containing folder name.

Discovery statically validates the source first, then probes metadata in a short-lived restricted process. Invalid unselected plugins are quarantined so an unrelated optional plugin cannot break normal core collection. A selected or globally enabled invalid plugin blocks the request.

## Contract requirements

The complete field-by-field contract is in [Contracts](CONTRACTS.md) and a runnable minimal pattern is in [Plugin Authoring](PLUGIN_AUTHORING.md). In short, a plugin must:

- have no import-time collection, network, registry, subprocess, or output side effects;
- construct deterministic metadata and explicitly declare access, privilege, network reach, sensitive categories, limits, output MIME types, and contract version;
- perform bounded work only in its private workspace and temporary directory;
- register complete artifacts through `context.artifacts`, never by writing directly to an output or package path;
- check cancellation during long work and return typed `succeeded`, `partial`, `skipped`, `cancelled`, or `failed` outcomes; and
- keep all host access within declared capabilities and supported platform adapters.

Plugins do not gain authority merely because they are enabled. The runtime applies the same isolated worker, quota, logging, artifact, packaging, cancellation, and capability-policy rules used for core collectors.
