# Engine and plugin contracts

`logicytics/contracts.py` is the stable vocabulary shared by the CLI, planner, runtime, core collectors, and the only supported extension type: `PluginCollector`. A contract violation is intentionally rejected during construction, preflight, planning, worker execution, artifact registration, or packaging rather than being silently repaired.

## Identity and ownership

Core IDs use `core.<specialty>.<name>`; plugin IDs use `plugin.<name>`. IDs are lowercase dotted identifiers, while labels such as profiles and categories are lowercase snake case. A core module must agree with its `core/<specialty>/<name>.py` location. A plugin must agree with its plugin file or `main.py` folder ownership.

`CollectorKind` has only `core` and `plugin`. Plugins may use a custom lowercase specialty; core collectors must use the closed `Specialty` enum. All metadata construction must be deterministic and side-effect-free.

## `CollectorMetadata`

Metadata is policy, not display text. It is evaluated before a collector may enter a plan.

| Field | Requirement and effect |
| --- | --- |
| `id`, `name`, `description`, `author` | Non-empty single-line strings. `id` must match its owner and location. |
| `version` | Semantic version, for example `1.2.0`. |
| `specialty` | A core `Specialty` value or a plugin's custom lowercase category. |
| `supported_platforms` | Non-empty tuple of lowercase platform labels; a mismatched current platform rejects the plan. |
| `capabilities` | Tuple of declared `Capability` values. Declared access is visible to planning and can be blocked. |
| `privilege_level` | `standard` or `elevated`; it must agree exactly with the `elevated_privileges` capability. |
| `network_access` | `none`, `local`, or `remote`; `network` capability requires local or remote access. |
| `sensitive_data_categories` | Lowercase labels describing sensitive output. Minimal mode cannot include a sensitive collector. |
| `default_profiles` | Non-empty mode-profile membership. Standard mode admits only routine configuration categories. |
| `dependencies` | Unique collector IDs, never the collector's own ID. Dependencies are ordered and policy-checked. |
| `output_media_types` | Non-empty unique MIME types. Every registered artifact must use one of these types. |
| `timeout_seconds` | Positive worker time limit. |
| `maximum_memory_bytes`, `maximum_output_bytes` | Positive per-collector resource ceilings. |
| `maximum_artifact_bytes`, `maximum_artifact_files` | Positive artifact limits; a missing per-artifact byte limit defaults to the output limit and cannot exceed it. |
| `maximum_retries`, `retry_delay_seconds` | Bounded retry policy: retries are 0–3 and delay is finite from 0–30 seconds. |
| `minimum_contract_version` | A valid `major.minor` contract version; the current contract is `4.0`. |
| `parallel_safe`, `resource_class` | Scheduling declaration. A resource class prevents unsafe overlap with conflicting work. |

### Capability values

| Capability | Declare it when the collector needs to |
| --- | --- |
| `filesystem_read` / `filesystem_write` | Read host files or write outside its supplied private workspace. |
| `registry_read` | Read Windows registry state. |
| `subprocess` | Invoke a host executable or shell-free command. |
| `network` / `packet_capture` | Reach the network or capture packet metadata. |
| `browser_data`, `sensitive_files`, `private_keys` | Handle those sensitive data categories. |
| `elevated_privileges` | Require an administrator context; metadata must also set `privilege_level=elevated`. |

The planner rejects selected collectors with a request-blocked capability. Offline mode additionally rejects `network` and `packet_capture` collectors. Capability declaration does not grant permission to evade the worker boundary.

## Collector lifecycle

```python
class ExamplePluginCollector(PluginCollector):
    @classmethod
    def metadata(cls) -> CollectorMetadata: ...
    def validate(self, context: CollectorContext) -> ValidationResult: ...
    @staticmethod
    def prepare(context: CollectorContext) -> ValidationResult: ...
    def collect(self, context: CollectorContext) -> CollectorResult: ...
    @staticmethod
    def finalize(context: CollectorContext, result: CollectorResult) -> CollectorResult: ...
    @staticmethod
    def estimate(context: CollectorContext) -> CollectionEstimate: ...
    @classmethod
    def dependencies(cls) -> tuple[str, ...]: ...
    def cleanup(self, context: CollectorContext) -> None: ...
```

`metadata` is declarative. `validate` is side-effect-free and checks prerequisites. `prepare` performs bounded collector-local setup. `collect` does the bounded evidence work. `finalize` returns the publishable result. `cleanup` releases collector-owned resources even when work failed; it must not delete engine-owned run data. `estimate` and `dependencies` are optional hooks when meaningful.

Preflight also enforces the public class shape. Do not add import-time work, dynamic code loading, hidden public methods, global mutable run state, nested worker pools, `print`-driven control flow, or collector-to-collector calls.

## Context, results, and progress

`CollectorContext` supplies the run ID, collector ID, private `workspace`, private `temporary_directory`, configuration settings, artifact writer, event logger, and cancellation state. Use the typed setting helpers (`setting_int`, `setting_float`, and `setting_str`) rather than coercing arbitrary values. Check `context.is_cancelled` before and during every traversal, copy, query, capture, loop, or long subprocess.

`CollectorResult` uses one terminal `CollectorStatus`:

| Status | Use it when |
| --- | --- |
| `succeeded` | The requested work completed and registered its declared output. |
| `partial` | Useful registered evidence exists, but a meaningful portion could not complete. |
| `skipped` | A normal prerequisite, device, feature, or permission is unavailable. |
| `cancelled` | The engine or user cancellation boundary was reached. |
| `failed` | The collector encountered an execution or contract error. |

Always provide a non-empty summary. Attach structured errors and scalar metrics where they aid review. Do not describe a partial or skipped result as success.

## Artifact contract

`context.artifacts.register_file(...)` is the only publication path. The source must be complete, non-empty, beneath the worker workspace, owned by the current collector, and use a declared MIME type. The writer validates its path, content size, SHA-256, file-count and byte quotas, and provenance before placing it under the run-owned artifact store.

An `Artifact` contains a stable ID, POSIX relative path, lowercase SHA-256, size, MIME type, collector ID, source category, timezone-aware collection timestamp, transformations, evidence kind, safe filename, and `registered` status. Artifact paths cannot escape the owner directory; packages are built only from the finalized manifest catalog.

## Plugin author checklist

1. Put one well-named `PluginCollector` implementation under `plugins/`.
2. Make metadata complete, honest, deterministic, and compatible with contract `4.0`.
3. Make validation explain absent prerequisites without collecting data.
4. Bound every loop, file, command, memory allocation, and output path.
5. Use platform adapters and the provided context rather than direct global host access.
6. Register only complete declared artifacts, return a typed result, and clean up collector-owned resources.
7. Add focused tests, run `preflight --plugins`, then inspect the plan before enabling the plugin in a collection.

See [Plugin Authoring](PLUGIN_AUTHORING.md) for a minimal example and [Errors](ERRORS.md) for remediation of contract failures.
