# Moving from older layouts

Use `logicytics.yaml` as the single settings source. Execution code lives under `logicytics/module/`, command parsing
under `logicytics/cli/`, and the public API under `logicytics/module/api.py`.

Older mode flags remain compatibility aliases: `--default` selects `standard`, `--threaded` selects `balanced`,
`--minimal` selects `quick`, and `--depth` selects `thorough`. New scripts should use `--mode` or `--profile`.

Legacy `CODE/config.ini` is read only as a compatibility input. The explicit `core.integration.legacy_code_outputs`
collector imports approved generated output; the engine does not implicitly scan the repository. `MODS/` is opt-in and
requires sidecars. Run evidence is owned by its manifest-backed run folder. There are no global `ACCESS/`, `RUNS/`,
`LOGS/`, or `PACKAGES/` stores.

Translate supported settings using [Configuration](CONFIGURATION.md), then run `preflight` and `plan` before collecting.
