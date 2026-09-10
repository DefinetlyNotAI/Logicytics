# Troubleshooting

Start with the command's exit code and the run `manifest.json`. A console summary is useful for navigation, but the
manifest records the plan, terminal statuses, artifact catalog, and actionable collector-level failure details.

## The CLI says it must run in a virtual environment

The normal CLI deliberately rejects a system interpreter. Bootstrap or repair the environment with the installer, then
use the explicit managed interpreter:

```powershell
python -m logicytics.cli.installer  # or .\.venv\Scripts\Activate.ps1 to activate if already installed
python -m logicytics preflight
```

If PowerShell blocks `.\.venv\Scripts\Activate.ps1`, activation is optional.
Keep using `.\.venv\Scripts\python.exe` rather than changing execution policy.

## Preflight reports invalid collectors or extensions

Read the named source and diagnostic row. Then run a fresh validation after the source is corrected:

```powershell
python -m logicytics preflight --invalidate-cache
```

Do not bypass preflight or manually mark a source valid. Syntax and import errors, unsafe names, missing or
nondeterministic metadata, unsupported platforms, undeclared capabilities, and import-time side effects are contract
violations. An optional Windows facility that is merely unavailable should be recorded distinctly from an invalid
source.

## The plan is empty or a collector is not selected

First confirm the exact dotted ID in [Core Collector Catalog](CORE_COLLECTORS.md) and inspect the selected mode with
`--modes`. Then rebuild the plan with the same mode, profile, includes, excludes, extension switches, and capability
blocks as the intended run. A collector can be excluded by a selector, not belong to the mode, be invalid in preflight,
or require a capability you blocked.

Plugins require `--plugins`; MODS require `--mods`. Their absence from a normal core-only plan is intentional. Review
their sidecars and source before enabling them.

## A collector is skipped, partial, failed, or canceled

Read its manifest record's summary, structured error details, remediation, and retry-safe indication. Skipped commonly
means a required command, optional Windows feature, device, permission, or prerequisite was unavailable. Partial means
some registered artifacts may still be valuable. A failed or canceled collector does not normally erase independent
collector results.

To narrow a reproducible problem, run the exact collector or use sequential execution:

```powershell
python -m logicytics collector core.system.system_info `
  --acknowledge-authorization

python -m logicytics run --mode standard --sequential `
  --acknowledge-authorization
```

Do not retry a sensitive or high-impact collector until its scope is still authorized. Elevation, if required by the
collector, must be granted through your normal administrative process.

## Nothing was written or packaging is absent

Verify the configured `runtime.output_root`, project permissions, free disk space, and `maximum_run_output_bytes`.
Relative output paths are kept inside the project and unsafe paths are rejected. A `--no-package` request intentionally
omits a ZIP; the manifest-backed run directory remains the result.

Use the diagnostic command for a bounded environment report:

```powershell
python -m logicytics debug
```

Inspect the manifest before deleting or rerunning anything. Packaging is not a success proxy: a ZIP can coexist with
partial results.

## Timeout, memory, or output limits

The runtime records the termination reason. Reduce the request to the smallest needed collector, make scheduling
deterministic with `--sequential`, and inspect the collector's documented bounds. Adjust collector settings only when
the larger collection is authorized and the destination can safely hold the output. Do not raise limits blindly for a
source that may traverse, copy, or expose sensitive data.

## Safe support material

Use `debug` and a sanitized manifest excerpt. Remove credentials, key material, browser files, usernames, hostnames,
paths, addresses, SIDs, and raw evidence. Never attach an entire output directory or package to a public issue.
See [Evidence Review](EVIDENCE_REVIEW.md) for the handoff checklist.
