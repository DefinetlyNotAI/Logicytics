# Logicytics documentation

Logicytics is a Windows-focused, run-oriented evidence collector. It plans a bounded request, gives every selected
collector an isolated workspace, registers the resulting files, and records the outcome in a durable manifest. This
manual is for operators who need a defensible collection workflow and developers who need to preserve the engine's
contracts.

Start with [Installation](INSTALLATION.md), then follow [Getting Started](GETTING_STARTED.md). For an actual
investigation or administrative collection, keep [Operations](OPERATIONS.md) open beside the terminal and finish
with [Evidence Review](EVIDENCE_REVIEW.md).

## Choose a path

| Goal                                          | Read                                                                                           |
|-----------------------------------------------|------------------------------------------------------------------------------------------------|
| I have never used the tool                    | [Installation](INSTALLATION.md), [Getting Started](GETTING_STARTED.md), [Safety](SAFETY.md)    |
| I need to run a collection responsibly        | [Operations](OPERATIONS.md), [Safety](SAFETY.md), [Core Collector Catalog](CORE_COLLECTORS.md) |
| I need to review or hand over evidence        | [Evidence Review](EVIDENCE_REVIEW.md), [Results](OUTPUTS.md), [Formats](FORMATS.md)            |
| I need every command                          | [Command Reference](COMMANDS.md)                                                               |
| I want to understand the engine               | [Engine](ENGINE.md), [Architecture](ARCHITECTURE.md)                                           |
| I need to change settings                     | [Configuration](CONFIGURATION.md)                                                              |
| I need to find an evidence source             | [Core Collector Catalog](CORE_COLLECTORS.md)                                                   |
| I need to read JSON, CSV, HTML, or ZIP output | [Results](OUTPUTS.md) and [Formats](FORMATS.md)                                                |
| I want to run a plugin                        | [Plugins](PLUGINS.md)                                                                          |
| I want to write a collector                   | [Plugin Authoring](PLUGIN_AUTHORING.md), [Contracts](CONTRACTS.md)                             |
| Something went wrong                          | [Troubleshooting](TROUBLESHOOTING.md), [Error Reference](ERRORS.md)                            |
| I am maintaining the repository               | [Development](DEVELOPMENT.md), [Verification](VERIFICATION.md)                                 |

## Safety rule of thumb

Logicytics can collect sensitive local evidence. Only run it on systems and data you are authorized to examine. Start
with `preflight`, review the plan, use the smallest profile that answers the question, and inspect the selected
capabilities before authorizing a run.

## A normal operator workflow

1. Install or repair the managed environment with the installer.
2. Run `preflight`; resolve invalid sources before collecting.
3. Create a `plan` for the exact mode and selectors you intend to use.
4. Review the collector IDs, declared capabilities, output location, and any sensitive sources. Add exclusions or
   `--block-capability` where needed.
5. Run only after adding `--acknowledge-authorization` deliberately.
6. Read `manifest.json` before relying on or sharing the output. A package is evidence delivery, not a success signal by
   itself.

The engine intentionally records skipped, partial, failed, and cancelled work instead of silently treating it as
success. Those statuses are useful evidence: read them before retrying or expanding scope.

## Documentation conventions

- A **collector ID** is the exact dotted identifier such as `core.system.system_info`.
- A **profile** selects a documented group of collectors; `--include` and `--exclude` refine it.
- A **capability** is an access category a collector must declare before it can request that access.
- A **run** is one planned, isolated execution with a manifest and optional package.
- Paths in examples are Windows paths. Keep the managed interpreter path exactly as shown when using PowerShell.

The repository workflow publishes this directory to the GitHub repository wiki after documentation changes land on the
default branch.
