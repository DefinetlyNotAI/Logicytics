# Getting started

## What the program does

Logicytics reads local Windows state, turns each requested source into a bounded evidence artifact, records what happened, and optionally creates a verified ZIP package. It is a collection and reporting tool, not a repair tool: normal collectors are designed to read state and do not change the system.

## Before you begin

1. Work from a copy or checkout of the repository.
2. Use a supported Windows Python installation and the repository's `.venv`.
3. Decide what you are authorized to collect. Process lists, account names, browser data, registry data, Wi-Fi keys, SSH material, and files may be sensitive.
4. Ensure the output directory is on storage you trust and that the evidence will be handled appropriately.

The installer is the only entry point intended to run outside the managed environment:

```powershell
python -m logicytics.cli.installer
```

Activate the environment, or use the explicit interpreter for every command:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m logicytics preflight
```

If PowerShell blocks activation, use the explicit interpreter instead. Do not copy a system Python command over a managed-environment command when the latter is shown.

## The safe first run

```powershell
.\.venv\Scripts\python.exe -m logicytics preflight
.\.venv\Scripts\python.exe -m logicytics plan --mode standard
.\.venv\Scripts\python.exe -m logicytics run --mode standard --acknowledge-authorization
```

`preflight` checks code and prerequisites. `plan` shows the selected collectors without collecting. `run` performs the isolated work. The authorization flag is an explicit confirmation; it is not a substitute for reviewing the plan.

## After the run

The final console summary gives the run manifest path and, when packaging succeeds, the ZIP path and SHA-256 sidecar. Open the manifest first. It tells you which collectors succeeded, were skipped, or failed and lists every registered artifact. See [Results](OUTPUTS.md) for the run tree and [Formats](FORMATS.md) for how to consume each file.

## Small, focused examples

Run only one low-risk collector:

```powershell
.\.venv\Scripts\python.exe -m logicytics collector core.system.system_info --acknowledge-authorization
```

Make a plan that excludes one source:

```powershell
.\.venv\Scripts\python.exe -m logicytics plan --mode standard --exclude core.system.system_details
```

Run deterministically for troubleshooting:

```powershell
.\.venv\Scripts\python.exe -m logicytics run --mode standard --sequential --acknowledge-authorization
```

Never enable plugins or MODS merely to make a command succeed. They are opt-in because they are user-owned code.
