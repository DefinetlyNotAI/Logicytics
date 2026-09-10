# Getting started

This walkthrough makes one ordinary, authorized Windows inventory collection. It does not repair the host or remove data. It does create evidence files, which can contain hostnames, account information, network details, and other sensitive local state. If that is not within your scope, stop here and read [Safety](SAFETY.md).

## 1. Prepare the repository

Open PowerShell at the repository root. The installer is the only Logicytics entry point intended to run with a system Python; it creates or reuses `.venv` and writes the default configuration only when it is absent.

```powershell
python -m logicytics.cli.installer
```

Use the managed interpreter for every remaining command. Activation is optional: the explicit path is more reproducible and also works when PowerShell execution policy blocks `Activate.ps1`.

```powershell
python -m logicytics --help
python -m logicytics preflight
```

`preflight` is read-only with respect to collection: it validates sources and prerequisites before a collector can be planned. An unavailable optional Windows facility is normally reported as unavailable or skipped; an invalid collector, unsafe extension, or configuration error must be fixed before proceeding.

## 2. Build and review a plan

The plan is the authorization checkpoint. It resolves a mode into exact collector IDs, dependencies, capabilities, and resource limits without starting workers or gathering evidence.

```powershell
python -m logicytics plan --mode standard
```

Read the plan and answer these questions:

- Are the selected IDs within the agreed scope? The complete descriptions are in [Core Collector Catalog](CORE_COLLECTORS.md).
- Do any declared capabilities include sensitive files, browser data, private keys, elevation, packet capture, or network access?
- Is `runtime.output_root` on storage approved for evidence? See [Configuration](CONFIGURATION.md).
- Should a source be removed, or should a capability be blocked across the request?

For example, this preserves a standard plan while preventing collectors that need sensitive files or packet capture from entering it:

```powershell
python -m logicytics plan --mode standard `
  --block-capability sensitive_files --block-capability packet_capture
```

If the plan is broader than necessary, refine it before collection. Selectors are repeatable, and an exact collector run is often a better diagnostic than a large profile:

```powershell
python -m logicytics collector core.system.system_info `
  --acknowledge-authorization
```

## 3. Run with explicit authorization

The acknowledgement confirms that the operator reviewed the requested evidence; it is not an automatic permission grant. Use it only after the preceding review.

```powershell
python -m logicytics run --mode standard `
  --acknowledge-authorization
```

Use `--sequential` for a deterministic troubleshooting run, `--no-package` when you need only the manifest-backed run directory, and `--interactive` when a console window should remain open at the end. Do not add `--plugins` or `--mods` just to make a command work: those switches opt into user-owned executable code.

## 4. Decide whether the result answers the question

Start with the run's `manifest.json`, not the ZIP file. Check the overall status, then every skipped, partial, failed, or cancelled collector record and its actionable details. A partial run may contain valid artifacts; a ZIP can exist even when not every collector succeeded.

The console gives the manifest location and, when packaging succeeded, the ZIP and SHA-256 sidecar. Follow [Evidence Review](EVIDENCE_REVIEW.md) for the review and handoff workflow, or [Troubleshooting](TROUBLESHOOTING.md) when a result is not usable.
