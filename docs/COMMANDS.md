# Command reference

Run commands from the repository root. Except for the installer, use the managed interpreter exactly as shown:

```text
.\.venv\Scripts\python.exe -m logicytics [global options] <command> [command options]
```

An option may appear before or after a command when the command supports it. Repeated options may be supplied more than once. `plan` is the safe checkpoint: it resolves scope without collecting evidence.

## Global options

| Flag | What it does |
| --- | --- |
| `--config PATH` | Selects the authoritative YAML configuration. The path is validated before planning. |
| `--usb [DRIVE]` | Targets a Windows installation on removable storage; omit `DRIVE` to scan, or provide a drive letter. |
| `--usage` | Displays local interaction statistics and writes the usage graph. |
| `--modes` | Prints and writes the generated execution-mode inclusion matrix. |
| `--match TEXT` | Suggests the closest documented action from a natural-language description. |
| `-h`, `--help` | Shows parser help without collecting evidence. |

## Installer

```powershell
python -m logicytics.cli.installer [--environment PATH] [--overwrite-config]
```

The installer is the only entry point intended for a system Python. It creates or reuses the virtual environment and creates `logicytics.yaml` if absent.

| Flag | What it does |
| --- | --- |
| `--environment PATH` | Uses this virtual-environment directory instead of `.venv`; a relative path is resolved from the repository root. |
| `--overwrite-config` | Replaces the root configuration template. Back up a customized file first. |

## `preflight`

```powershell
.\.venv\Scripts\python.exe -m logicytics preflight [flags]
```

Validates core and, when enabled, plugin sources before collection. It reports valid collectors, quarantined invalid optional plugins, and blocking selected failures.

| Flag | What it does |
| --- | --- |
| `--config PATH`, `--usb [DRIVE]` | Applies the global configuration or removable-media source selection. |
| `--profile MODE` | Uses one named mode for selected-plugin validation context. |
| `--include ID` | Repeats to select exact collector IDs; selected invalid plugins become blocking failures. |
| `--exclude ID` | Repeats to remove IDs from the request. |
| `--plugins` | Enables valid `PluginCollector` implementations from `plugins/` for the request. |
| `--invalidate-cache` | Discards cached validation metadata and performs a fresh source/runtime probe. |
| `--workers COUNT` | Supplies the bounded worker count carried into request validation. |
| `--block-capability CAP` | Repeats to prohibit a declared capability. See [Safety](SAFETY.md). |

## `plan`

```powershell
.\.venv\Scripts\python.exe -m logicytics plan --mode standard [flags]
```

Builds and prints a deterministic, dependency-ordered plan. It starts no collector worker and writes no evidence artifacts.

| Flag | What it does |
| --- | --- |
| `--mode {quick,balanced,standard,offline,thorough}` | Selects the request mode and its scheduling behavior. |
| `--profile MODE` | Alternative named mode selection. Do not combine it with `--mode`. |
| `--include ID` / `--exclude ID` | Repeats to refine the resolved profile with exact dotted collector IDs. |
| `--plugins` | Allows valid opt-in plugins to be considered; plugins remain opt-in even when their profile matches. |
| `--invalidate-cache` | Revalidates collector metadata before planning. |
| `--workers COUNT` | Sets a positive bounded worker count for the resulting request. |
| `--block-capability CAP` | Rejects a plan containing a collector that declares the capability. |
| `--config PATH`, `--usb [DRIVE]` | Selects configuration or removable-media context. |

## `run`

```powershell
.\.venv\Scripts\python.exe -m logicytics run --mode standard --acknowledge-authorization [flags]
```

Runs the reviewed plan in isolated workers. Read the final manifest even after an exit code of `0`; a package is not proof that every collector succeeded.

| Flag | What it does |
| --- | --- |
| `--mode MODE` | Selects a named execution mode. |
| `--default`, `--threaded`, `--minimal`, `--depth` | Historical aliases for `standard`, `balanced`, `quick`, and `thorough`; exactly one mode selector is allowed. |
| `--profile MODE` | Alternative named mode selector; cannot be combined with a mode selector. |
| `--include ID` / `--exclude ID` | Repeats to add or remove exact collector IDs. |
| `--plugins` | Enables valid opt-in plugins for this run. |
| `--workers COUNT` | Caps concurrent isolated workers. Sequential work requires `1`; parallel work requires at least `2`. |
| `--sequential` / `--parallel` | Mutually exclusive scheduling override. Use sequential execution for deterministic diagnosis. |
| `--performance-check` | Forces serial measurement and writes per-collector duration evidence. It cannot be combined with `--parallel`. |
| `--block-capability CAP` | Repeats to prohibit declared access categories. |
| `--rerun-from PATH` | Re-runs explicitly included IDs from a finalized compatible manifest or run directory. |
| `--no-package` | Finalizes the manifest-backed run directory without writing a ZIP package. |
| `--reboot` / `--shutdown` | Mutually exclusive post-run host actions, scheduled only after verified package publication. |
| `--acknowledge-authorization` | Required confirmation that the operator has authority for the reviewed scope. |
| `--interactive` | Keeps the interactive command window open after final status. |
| `--config PATH`, `--usb [DRIVE]` | Selects configuration or removable-media context. |

## `collector`

```powershell
.\.venv\Scripts\python.exe -m logicytics collector core.system.system_info --acknowledge-authorization [flags]
```

Runs one exact collector ID independently. It is the preferred way to diagnose one source after reviewing its capabilities.

| Flag | What it does |
| --- | --- |
| `COLLECTOR_ID` | Required positional dotted ID for the one collector to run. |
| `--acknowledge-authorization` | Required authorization confirmation. |
| `--no-package` | Keeps the manifest-backed result without a ZIP. |
| `--interactive` | Keeps an interactive command window open at completion. |
| `--workers COUNT` | Sets the bounded worker limit; the direct collector still resolves one selected source. |
| `--block-capability CAP` | Rejects the collector when it declares a blocked capability. |
| `--config PATH`, `--usb [DRIVE]` | Selects configuration or removable-media context. |

`collector` cannot be combined with `--profile`, `--include`, `--exclude`, or `--plugins`; its positional ID is the complete selection.

## `debug`

```powershell
.\.venv\Scripts\python.exe -m logicytics debug [flags]
```

Writes a bounded diagnostic JSON report containing environment, interpreter, configuration, maintenance, optional-tool, and preflight information. Sanitize it before sharing.

| Flag | What it does |
| --- | --- |
| `--config PATH`, `--usb [DRIVE]` | Uses the selected configuration and target context while gathering diagnostics. |
| `--profile MODE`, `--include ID`, `--exclude ID`, `--plugins` | Applies the request selection used for preflight diagnostics. |
| `--workers COUNT`, `--block-capability CAP` | Carries execution and policy context into the diagnostic request. |

## `update`

```powershell
.\.venv\Scripts\python.exe -m logicytics update [--apply] [--launch-action ACTION --new-window]
```

Checks Git, the repository, origin configuration, and remote reachability. It never pulls unless `--apply` is explicit.

| Flag | What it does |
| --- | --- |
| `--apply` | Runs `git pull` after successful repository and remote checks. |
| `--launch-action {preflight,debug,dev}` | Selects an allowlisted maintenance action to launch after update. Requires `--new-window`. |
| `--new-window` | Starts the allowlisted action in a visible separate Windows console. |
| `--config PATH`, `--usb [DRIVE]` | Accepts common context flags; update behavior itself operates on the checkout. |

## `dev`

```powershell
.\.venv\Scripts\python.exe -m logicytics dev [--write-manifest --next-version VERSION] [--interactive]
```

Runs repository integrity and organization checks. It is a developer maintenance command, not an evidence collection command.

| Flag | What it does |
| --- | --- |
| `--write-manifest` | Writes the reviewed local integrity manifest; requires `--next-version` unless interactive mode supplies it. |
| `--next-version VERSION` | Provides the semantic version recorded in a newly written manifest. |
| `--interactive` | Shows checks and prompts before an integrity-manifest write. |
| `--config PATH`, `--usb [DRIVE]` | Accepts common context flags. |

## Exit codes

| Code | Meaning | Next action |
| --- | --- | --- |
| `0` | Command completed successfully. | Review the manifest or generated report. |
| `1` | A collection finished with non-success status. | Preserve the manifest and inspect partial, skipped, failed, or cancelled records. |
| `2` | Argument, configuration, preflight, planning, or command error. | Read the rendered error and [Error reference](ERRORS.md); correct the cause rather than bypassing validation. |
| `130` | Interrupted. | Locate the durable manifest and determine what safely finalized before retrying. |
