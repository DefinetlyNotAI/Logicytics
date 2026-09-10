# MODS

This page is the short compatibility entry point for the optional Python-script extension area.
Read [Plugins and extensions](PLUGINS.md) for discovery, sidecars, execution, output registration, and security rules,
then read [Plugin Authoring](PLUGIN_AUTHORING.md) when writing code.

Enable MODS only deliberately:

```powershell
python -m logicytics preflight --mods
python -m logicytics run --mode standard --mods --acknowledge-authorization
```

Python files require an adjacent metadata sidecar and run in isolated private workspaces. A normal run does not execute
them.
