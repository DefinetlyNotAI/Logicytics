# Troubleshooting

## “Must run inside a virtual environment”

Use `.\.venv\Scripts\Activate.ps1` or call `.\.venv\Scripts\python.exe` directly. The installer can create or repair the environment.

## Preflight reports invalid collectors

Run with `--invalidate-cache`, read the diagnostic rows, and fix the named source or sidecar. Do not bypass preflight or manually mark a collector valid. A syntax/import error, unsafe filename, missing metadata, unsupported platform, undeclared capability, or import-time side effect must be corrected at the source.

## A collector is skipped

Skipped usually means a Windows command, optional feature, device, permission, or prerequisite was unavailable. Read the manifest's summary and errors. Run the exact collector with `collector ID` if you need a focused diagnostic; elevation may be required for the declared collector.

## A run is partial or failed

Open `manifest.json`, then the run/application JSONL logs. A partial run may contain useful artifacts. A failed collector should include an operation, platform error, remediation, and retry-safe flag when the engine can determine them. Do not treat a ZIP's existence alone as proof that every collector succeeded.

## Nothing is written

Check `runtime.output_root`, project permissions, free disk space, and the configured maximum run output. Use `debug` to write a diagnostic report. Relative paths are resolved beneath the project; unsafe paths are rejected.

## A plugin or MOD is not selected

Plugins require `--plugins`; MODS require `--mods`. Verify the exact ID, filename, sidecar, profile, capabilities, and preflight status. A valid extension is intentionally absent from a normal core-only plan.

## Cancellation, timeout, or memory limit

The manifest records cancellation and termination reason. Reduce scope, lower output limits, use `--sequential` to isolate the problem, or adjust the collector's bounded settings. Do not increase limits blindly when the source may contain sensitive or unbounded data.

## Sharing diagnostics

Use `debug` and sanitized manifest excerpts. Remove credentials, key material, browser files, usernames, hostnames, paths, addresses, SIDs, and raw evidence. Never attach an entire output directory to a public issue.
