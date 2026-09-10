# Command reference

The installer is the only command intended to run outside the managed environment. All other examples use the explicit
interpreter.

## Global form

```text
python -m logicytics [global options] <command> [command options]
```

Global options:

| Option          | Meaning                                                                              |
|-----------------|--------------------------------------------------------------------------------------|
| `--config PATH` | Use a specific authoritative YAML file                                               |
| `--usb [DRIVE]` | Locate a Windows installation on removable storage; optionally choose a drive letter |
| `--usage`       | Show local interaction statistics and create a usage graph                           |
| `--modes`       | Show the typed execution-mode inclusion matrix                                       |
| `--match TEXT`  | Suggest the closest documented action for natural-language text                      |
| `-h`, `--help`  | Show help                                                                            |

## Installer

```powershell
python -m logicytics.cli.installer
```

The menu prepares the virtual environment and root YAML. It is intentionally separate from collection.

## `preflight`

```powershell
python -m logicytics preflight [options]
```

Checks all selected source trees and prerequisites without running collection. Options: `--profile`,
`--invalidate-cache`, repeated `--include ID`, repeated `--exclude ID`, `--plugins`, `--mods`, repeated
`--block-capability CAP`, `--config`, and `--usb [DRIVE]`.

## `plan`

```powershell
python -m logicytics plan --mode MODE [options]
```

Creates and prints a plan without starting workers. Options are `--mode {quick,balanced,standard,offline,thorough}`,
`--profile`, `--invalidate-cache`, `--include`, `--exclude`, `--plugins`, `--mods`, `--workers COUNT`, repeated
`--block-capability`, `--config`, and `--usb`.

## `run`

```powershell
python -m logicytics run --mode MODE [options]
```

The mode options are mutually exclusive: `--mode MODE`, `--default`, `--threaded`, `--minimal`, or `--depth`.
`--profile` is the named profile form and should not conflict with the mode selection.

All run options: `--include ID`, `--exclude ID`, `--plugins`, `--mods`, `--workers COUNT`, `--block-capability CAP`,
`--rerun-from PATH`, `--sequential`, `--parallel`, `--performance-check`, `--no-package`, `--reboot`, `--shutdown`,
`--acknowledge-authorization`, `--interactive`, `--config`, and `--usb`.

`--reboot` and `--shutdown` are mutually exclusive. `--sequential` and `--parallel` are mutually exclusive. Post-run
power actions require packaged output.

## `collector`

```powershell
python -m logicytics collector core.system.system_info --acknowledge-authorization
```

Runs one exact collector ID independently. It accepts `--profile`, repeated `--include`/`--exclude`, `--plugins`,
`--mods`, `--workers`, repeated `--block-capability`, `--no-package`, `--acknowledge-authorization`, `--interactive`,
`--config`, and `--usb`.

## `debug`

```powershell
python -m logicytics debug
```

Writes a diagnostic JSON report containing environment, Python, configuration, maintenance, optional tool, and preflight
information. It accepts the common config and USB options plus selection and capability options.

## `update`

```powershell
python -m logicytics update [--apply] [--launch-action { preflight, debug, dev }] [--new-window]
```

Checks Git and remote reachability. `--apply` explicitly runs `git pull`. `--launch-action` must be paired with
`--new-window`; the action is allowlisted.

## `dev`

```powershell
python -m logicytics dev [--write-manifest] [--next-version VERSION] [--interactive]
```

Runs developer integrity and contribution checks. Writing the local integrity manifest requires a valid semantic version
and the explicit write option.

## Modes and profiles

The available named modes are `quick`, `balanced`, `standard`, `offline`, and `thorough`. Historical aliases are
`--default`, `--threaded`, `--minimal`, and `--depth`. Use `--modes` to inspect the live inclusion matrix; do not guess
which collector belongs to a profile.

## Exit codes

`0` means the requested command completed successfully. `1` means a collection completed with a non-success run status.
`2` means command, configuration, preflight, or planning failure. `130` means interruption. Always use the manifest and
diagnostic JSON to understand a non-zero result.
