# MODS compatibility contract

`MODS/` is the opt-in compatibility area for Python collectors. Normal runs
never execute these files. Use `run --modded` to run the standard core profile
followed by valid MODS payloads. Only Python `.py` scripts are accepted.

## Layout and discovery

Mods are discovered recursively. Runnable filenames must be lowercase
`snake_case` and use `.py`. PowerShell, batch, and executable payloads are
rejected during preflight. Files and directories whose names begin with `_` are
ignored. Every runnable file needs an adjacent
sidecar named `<filename>.<extension>.mod.json`; for example:

```text
MODS/
  inventory/
    local_report.py
    local_report.py.mod.json
```

The sidecar must declare the complete immutable collector metadata contract.
This example is intentionally bounded and non-sensitive:

```json
{
  "id": "mod.local_report",
  "name": "Local report",
  "version": "1.0.0",
  "specialty": "integration",
  "description": "Creates a bounded local compatibility report.",
  "author": "Example author",
  "supported_platforms": ["win32"],
  "capabilities": ["subprocess", "filesystem_write"],
  "privilege_level": "standard",
  "sensitive_data_categories": [],
  "network_access": "none",
  "estimated_cost": "low",
  "timeout_seconds": 30,
  "maximum_output_bytes": 1048576,
  "maximum_artifact_files": 20,
  "output_media_types": ["text/plain", "application/json"],
  "minimum_contract_version": "4.0",
  "default_profiles": ["standard"]
}
```

The ID must be `mod.<filename_stem>`. MODS scripts must request the
`subprocess` capability. Other access must also be declared and remains
subject to the active blocked-capability policy. Python MODs are audit-confined
to their private workspace by default.
Python MODS need `filesystem_write` only when intentionally writing outside
their private workspace. Invalid or missing sidecars quarantine the MOD;
selecting all MODS then fails preflight before any collector launches.

## Execution and output

Core collectors run before mods. Each mod receives its own worker process,
private workspace, temporary directory, event channel, timeout, memory limit,
artifact quota, cancellation signal, and result record. The source is copied
into the private workspace before execution and uses the configured Python
interpreter.

Write generated evidence beneath the current working directory, exposed as
`LOGICYTICS_WORKSPACE`. Only non-empty files whose MIME type appears in
`output_media_types` are registered. Standard output and error are captured as
text artifacts when non-empty. Lines formatted as `LEVEL: message` are also
converted into structured events for `DEBUG`, `INFO`, `WARNING`, `ERROR`,
`CRITICAL`, `INTERNAL`, and `EXCEPTION`.

A mod failure, crash, timeout, cancellation, or malformed output affects only
that mod. Independent collectors continue, and the terminal result appears in
the run manifest and summary. Registered mod evidence is included in the normal
manifest-led run package and in a separately named `mods-*.zip` with its own
SHA-256 sidecar.

## Security boundary

Mods are untrusted, opt-in local code. Worker/process isolation, a private
workspace, declared-capability enforcement, bounded resources, packaging
allowlists, and process-tree termination contain engine state and evidence
publication. Python MOD mutation attempts outside the workspace fail unless
`filesystem_write` was declared. A configured or command-line block still
overrides a declaration. Review a MOD and its declared capabilities before
executing it.
