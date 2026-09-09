# Logicytics documentation

This is the complete user and developer manual for Logicytics, a Windows evidence-collection engine. Read only [Getting Started](GETTING_STARTED.md) to perform a safe first run; use the rest as a reference library.

## Choose a path

| Goal | Read |
| --- | --- |
| I have never used the tool | [Getting Started](GETTING_STARTED.md), [Safety](SAFETY.md), [Results](OUTPUTS.md) |
| I need every command | [Command Reference](COMMANDS.md) |
| I want to understand the engine | [Engine](ENGINE.md), [Architecture](ARCHITECTURE.md) |
| I need to change settings | [Configuration](CONFIGURATION.md) |
| I need to find an evidence source | [Core Collector Catalog](CORE_COLLECTORS.md) |
| I need to read JSON, CSV, HTML, or ZIP output | [Results](OUTPUTS.md) and [Formats](FORMATS.md) |
| I want to run a plugin | [Plugins](PLUGINS.md) |
| I want to write a collector | [Plugin Authoring](PLUGIN_AUTHORING.md), [Contracts](CONTRACTS.md) |
| Something went wrong | [Troubleshooting](TROUBLESHOOTING.md) |
| I am maintaining the repository | [Development](DEVELOPMENT.md), [Verification](VERIFICATION.md) |

## Safety rule of thumb

Logicytics can collect sensitive local evidence. Only run it on systems and data you are authorized to examine. Start with `preflight`, review the plan, use the smallest profile that answers the question, and inspect the selected capabilities before authorizing a run.

## Documentation conventions

- A **collector ID** is the exact dotted identifier such as `core.system.system_info`.
- A **profile** selects a documented group of collectors; `--include` and `--exclude` refine it.
- A **capability** is an access category a collector must declare before it can request that access.
- A **run** is one planned, isolated execution with a manifest and optional package.
- Paths in examples are Windows paths. Keep the managed interpreter path exactly as shown when using PowerShell.

The repository workflow publishes this directory to the GitHub repository wiki after documentation changes land on the default branch.
