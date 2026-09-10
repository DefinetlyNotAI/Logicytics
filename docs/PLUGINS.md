# Plugins and extensions

There are two extension routes. A `PluginCollector` is a typed Python collector in `plugins/`; a MOD is a Python script
in `MODS/` with a JSON sidecar. Both are opt-in, preflighted, isolated, quota-bound, and included in manifest-led
packaging when they publish evidence.

## Plugins

Plugins use the full collector contract and are enabled with `--plugins`. They can be selected by exact ID with
`--include plugin.name`. Read [Plugin Authoring](PLUGIN_AUTHORING.md) for the complete lifecycle.

## MODS

MODS accepts Python `.py` files only. A runnable file requires an adjacent `<file>.py.mod.json` sidecar. Lowercase
snake_case filenames are required; names beginning with `_` are ignored. Normal runs never execute MODS.

Example layout:

```text
MODS/
  inventory/
    local_report.py
    local_report.py.mod.json
```

Required sidecar fields include `id` (`mod.local_report`), `name`, `version`, `specialty`, `description`, `author`,
`supported_platforms`, `capabilities`, `privilege_level`, `sensitive_data_categories`, `network_access`,
`estimated_cost`, `timeout_seconds`, `maximum_output_bytes`, `maximum_artifact_files`, `output_media_types`,
`minimum_contract_version`, and `default_profiles`.

MODS scripts run with `LOGICYTICS_WORKSPACE` as their private working directory. Non-empty files, allowed MIME types,
standard output, and standard error can become artifacts. Lines formatted as `LEVEL: message` become structured events.
A MOD failure affects only that MOD.

The `subprocess` capability is required for MOD execution; other capabilities must also be declared. A blocked
capability still overrides the declaration. Review third-party scripts before enabling them.
