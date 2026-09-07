# Configuration

Logicytics has one authoritative user configuration: [`logicytics.yaml`](../logicytics.yaml) at the application root. The installer creates it when needed, repair workflows may update it, and users may edit it directly.

The file uses a strict, mapping-only YAML subset. Invalid indentation, duplicate keys, unsupported settings, unsafe paths, non-finite numbers, and invalid values stop planning before collectors run.

Key controls include the output root, bounded worker limits, console and file-log policy, collector-specific safe limits, and Sysinternals setup. Sysinternals is enabled by default; set `maintenance.sysinternals_enabled: false` to disable discovery, download, and extraction entirely.

For the complete configuration reference and examples, use the [Logicytics Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki).
