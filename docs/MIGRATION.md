# Migration

Logicytics v4 uses `logicytics.yaml` as its sole settings source and keeps application execution code under `logicytics/module/`. Commands live under `logicytics/cli/`; globally available Windows ctypes infrastructure lives under `logicytics/global/`.

Run `python -m logicytics.cli.installer` to prepare the virtual environment and root YAML configuration, then run `python -m logicytics preflight` from that environment.

## Command aliases

The old mode flags remain explicit compatibility aliases. Their v4 equivalents are:

| Legacy flag           | v4 mode              |
|-----------------------|----------------------|
| `--default`           | `--mode standard`    |
| `--threaded`          | `--mode balanced`    |
| `--minimal`           | `--mode quick`       |
| `--depth`             | `--mode thorough`    |
| `--performance-check` | `--mode performance` |

Only one mode or alias may be selected per invocation. Use `--mods` to opt into
Python MODS sidecars. Prefer named `--mode` values in new automation.

## File and collector bridges

- `CODE/config.ini` is read only as a legacy compatibility input; v4 writes canonical manifests and does not mutate the file.
- `core.integration.legacy_code_outputs` remains the explicit bridge for approved legacy `CODE` output. It is not an implicit scan of the repository.
- `MODS/` is an opt-in Python extension area. MOD sidecars declare the script, capabilities, output paths, and media types before a run can select them.
- There are no global `ACCESS/`, `RUNS/`, `LOGS/`, or `PACKAGES/` stores. Run-owned evidence is written to `runtime.output_root/run/<fingerprint-prefix>/`; verified ZIPs and their hashes are published to `runtime.output_root/zip/<fingerprint-prefix>.zip` and `runtime.output_root/hashes/<fingerprint-prefix>.zip.sha256`. Prefixes start at eight characters and extend only to resolve a collision.

Do not copy v3 JSON settings into the v4 YAML path. Translate supported values using [CONFIGURATION.md](CONFIGURATION.md), then run `python -m logicytics preflight` before collecting.

Technical migration guidance is maintained in the [Logicytics Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki).
