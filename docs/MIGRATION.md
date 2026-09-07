# Migration

Logicytics v4 uses `logicytics.yaml` as its sole settings source and keeps application execution code under `logicytics/module/`. Commands live under `logicytics/cli/`; globally available Windows ctypes infrastructure lives under `logicytics/global/`.

Run `python -m logicytics.cli.installer` to prepare the virtual environment and root YAML configuration, then run `python -m logicytics preflight` from that environment.

Technical migration guidance is maintained in the [Logicytics Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki).
