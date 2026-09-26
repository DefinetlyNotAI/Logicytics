# Error reference

This reference explains every user-facing error family emitted by the CLI, planner, preflight, runtime, artifact writer, package publisher, and public API. The final detail can include a Windows error, file path, collector ID, or exception text from the host; preserve that detail in the manifest or sanitized diagnostic because it identifies the exact failing source.

## Read an error before retrying

1. Record the command, exit code, and exact rendered message.
2. For a collection, open the referenced `manifest.json`; a non-success run can retain valid registered artifacts.
3. Read the error family below, correct the underlying input, scope, capability, source, or environment condition.
4. Re-run `preflight` or `plan` before re-authorizing collection. Do not bypass a gate by changing a status or deleting diagnostic files.

## CLI and request errors

| Rendered error or family                                                                                       | Meaning                                                                                    | Correction                                                                                  |
|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `unrecognized arguments` / parser usage                                                                        | A flag is unsupported, misspelled, or placed after an incompatible command.                | Use the supported tables in [Commands](COMMANDS.md).                                        |
| `--profile cannot be combined with a named collection mode`                                                    | Two mode selectors were supplied.                                                          | Use one of `--profile`, `--mode`, or a legacy mode alias.                                   |
| `--mode cannot be combined with a legacy mode alias` / `legacy collection mode aliases are mutually exclusive` | Conflicting mode selectors were supplied.                                                  | Select exactly one mode.                                                                    |
| `performance checking requires sequential execution`                                                           | `--performance-check` was combined with parallel scheduling.                               | Remove `--parallel`; performance measurement is serial.                                     |
| `sequential execution requires --workers=1`                                                                    | A sequential request asks for more than one worker.                                        | Omit `--workers` or set it to `1`.                                                          |
| `parallel execution requires at least two configured workers`                                                  | Parallel execution cannot run with fewer than two workers.                                 | Increase the configured or explicit worker count.                                           |
| `collector execution cannot combine its ID with profile or selection flags`                                    | Direct `collector ID` selection was mixed with profile/include/exclude/plugin selection.   | Run only the positional ID, or use `plan`/`run` for a multi-collector request.              |
| `--rerun-from requires at least one explicit --include collector ID`                                           | A rerun has no precise source selection.                                                   | Supply one or more `--include` IDs from the original finalized plan.                        |
| `original run manifest cannot be loaded`                                                                       | The supplied rerun path is unreadable, not JSON, or not a manifest.                        | Use the prior run directory or its `manifest.json`; preserve the original file.             |
| `original run manifest must ...` / `uses unsupported schema_version`                                           | The rerun manifest is malformed, unfinished, incompatible, or lacks a valid resolved plan. | Rerun from a finalized compatible v4 manifest.                                              |
| `rerun collectors were not present in the original run`                                                        | An include ID was not part of the prior resolved plan.                                     | Choose only IDs recorded by that manifest, or make a new reviewed request.                  |
| `must run inside a virtual environment`                                                                        | The normal CLI was invoked with system Python.                                             | Run the installer, then use `.\.venv\Scripts\python.exe -m logicytics ...`.                 |
| `Command cancelled` / exit `130`                                                                               | The process received an interrupt.                                                         | Locate the manifest, inspect finalized records, and retry only if scope remains authorized. |

## Configuration errors

Configuration failures are `PlanError` conditions and exit before collection. They protect a single authoritative YAML configuration.

| Error family                                                                            | Meaning                                                                                       | Correction                                                                                                    |
|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `configuration file ...` / `cannot read configuration`                                  | The selected file is missing, unreadable, oversized, or not valid UTF-8.                      | Use an accessible YAML file below the size limit.                                                             |
| `configuration must contain ...` / `unknown configuration key`                          | The root or nested mapping has the wrong shape or an unsupported key.                         | Compare the file against [Configuration](CONFIGURATION.md); remove typos rather than relying on ignored keys. |
| `duplicate key`                                                                         | YAML repeats a key, making intent ambiguous.                                                  | Keep one value for the key.                                                                                   |
| `unsupported configuration schema_version`                                              | The file is not schema version `4`.                                                           | Migrate deliberately to the documented schema.                                                                |
| `must be ...` / `must not ...` for runtime, logging, interaction, or maintenance fields | A value has the wrong type, range, label, URL, digest, or path form.                          | Use the field's documented type and bound; do not coerce booleans into numbers.                               |
| `path escapes project` / `unsafe path`                                                  | A configured relative or absolute path leaves its approved root or uses a forbidden location. | Choose a permitted project-relative output, manifest, dump, or temporary path.                                |
| `collector settings ...`                                                                | A core collector received unknown, missing, or out-of-range settings.                         | Use only that collector's settings and bounds from [Configuration](CONFIGURATION.md).                         |

## Preflight errors

Preflight combines static source checks and an isolated runtime metadata probe. A core failure always blocks. An invalid plugin is quarantined until explicitly selected or enabled, at which point it blocks.

| Diagnostic family                                                                        | Meaning                                                                                            | Correction                                                                    |
|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `collector preflight failed`                                                             | One or more selected/core candidates are invalid. The appended path and diagnostics identify them. | Fix each named source, then run `preflight --invalidate-cache`.               |
| `unsafe filename`, `forbidden top-level statement`, `import-time ...`                    | Source layout or import behavior would perform work before isolation.                              | Use a regular lowercase Python module and move work into lifecycle methods.   |
| `collector class requires a docstring` / `metadata must construct one CollectorMetadata` | The source does not expose the required inspectable class shape.                                   | Follow [Contracts](CONTRACTS.md) and [Plugin Authoring](PLUGIN_AUTHORING.md). |
| `plugin metadata must explicitly declare ...`                                            | Plugin metadata omits a required security, quota, or output field.                                 | Declare every named field honestly in `metadata()`.                           |
| `invalid validation response` / `collector contract version is unsupported`              | The isolated probe returned invalid metadata or an unsupported contract.                           | Correct `metadata()` and use minimum contract version `4.0`.                  |
| `collector ID must start with its owner kind` / location mismatch                        | Metadata ID does not match a core/plugin module's ownership.                                       | Rename the module, folder, or ID so they agree.                               |
| `preflight cache ...`                                                                    | Cached validation state is unreadable or stale.                                                    | Run with `--invalidate-cache`; never edit the cache to force validity.        |

## Planning and policy errors

| Error family                                                           | Meaning                                                                                          | Correction                                                                      |
|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `unknown collection profile`                                           | The request profile is not one of the built-in profiles.                                         | Use the modes shown by `--modes`.                                               |
| `max_workers must be positive` / `request max_workers ...`             | Worker count is outside its supported positive bound.                                            | Set a valid count and honor sequential/parallel constraints.                    |
| `requested collectors are unavailable`                                 | An include ID was not discovered as valid.                                                       | Check the exact ID, plugin enablement, preflight result, and configuration.     |
| `selected collector depends on ...`                                    | A dependency is unavailable, excluded, or forms a cycle.                                         | Restore the dependency, remove the exclusion, or correct metadata dependencies. |
| `plugin dependency ... must be explicitly selected or plugins enabled` | A selected collector depends on an opt-in plugin that was not enabled.                           | Review and enable plugins, or explicitly include the dependency.                |
| `selected collector policy validation failed`                          | The detailed rows identify blocked capability, unsupported platform, or offline-policy conflict. | Reduce scope, remove the block only when authorized, or use a supported host.   |
| `selected collectors require an administrator account`                 | Metadata requires elevation and the process is not elevated.                                     | Use the normal approved administrative process, or exclude the collector.       |
| `authorization acknowledgement is required`                            | A state-changing collection request omitted explicit authorization acknowledgement.              | Review the plan, then add `--acknowledge-authorization` if authorized.          |

## Metadata and contract errors

These `ValueError` messages are emitted during metadata, request, artifact, estimate, or result construction. They identify a programmer or plugin-authoring contract failure, not a condition to suppress.

| Error family                                                                   | Meaning                                                                                     | Correction                                                                                    |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `metadata ... must be a non-empty single-line string`                          | Required identity text is missing or contains a newline.                                    | Provide one non-empty single-line value.                                                      |
| `metadata id/specialty/version/minimum_contract_version has an invalid schema` | Identity/version format is invalid.                                                         | Use supported core/plugin ID, specialty, semantic version, and `major.minor` contract format. |
| `metadata ... must be a tuple ...` / `must not contain duplicates`             | A metadata sequence has the wrong type, invalid labels, or duplicates.                      | Use immutable tuples of valid unique values.                                                  |
| `sensitive collectors must not belong to the minimal profile`                  | Sensitive evidence was placed in the smallest profile.                                      | Remove `minimal` from the profile list.                                                       |
| `standard collectors may contain only routine ... categories`                  | Standard mode was assigned sensitive categories outside its approved set.                   | Move it to a narrower/deeper explicit profile.                                                |
| `privilege_level must match ...` / `network_access must declare ...`           | Access declarations contradict their capability.                                            | Align elevation/network values with capability declarations.                                  |
| `maximum_artifact_bytes must not exceed maximum_output_bytes`                  | An individual artifact could exceed total collector output.                                 | Lower the artifact ceiling or increase the total bound responsibly.                           |
| `request include and exclude selections must not overlap`                      | The same collector was simultaneously selected and removed.                                 | Keep it on one side only.                                                                     |
| `request ... must be boolean` / `must be a tuple of ...`                       | A programmatic `RunRequest` uses wrong types.                                               | Construct the immutable request with documented types.                                        |
| `artifact ...`                                                                 | Artifact identity, hash, timestamp, MIME type, owner, path, or transformations are invalid. | Publish through the supplied writer; do not manually forge artifacts.                         |
| `collector result ...` / `estimated_...`                                       | A collector returned an invalid status, summary, result shape, or estimate.                 | Return a typed constructor result with non-empty summary and valid metrics.                   |

## Runtime, artifact, and capability errors

| Error family                                       | Meaning                                                                       | Correction                                                                                             |
|----------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `CAPABILITY_BLOCKED`                               | A declared capability is prohibited by request or configuration policy.       | Keep the collector out of the plan or remove the block only after authorization review.                |
| `CAPABILITY_UNDECLARED`                            | Collector code attempted host access it did not declare.                      | Correct metadata and implementation; do not add capability declarations merely to silence the failure. |
| `artifact source ...` / `artifact ... escapes ...` | A source, destination, symlink, or owner path violates workspace confinement. | Produce files under the provided workspace and register them through the writer.                       |
| `artifact source changed during registration`      | The file changed while being copied and hashed.                               | Finish writing before registration; use an atomic collector-local staging pattern.                     |
| `artifact ... exceeds ... limit`                   | Per-file, artifact-count, reserved-output, or run-output quota was exceeded.  | Reduce collection or define a justified bounded output policy.                                         |
| `collector worker crashed`                         | The isolated worker raised unexpectedly.                                      | Read the collector record's error and trace; fix the source or prerequisite, then preflight.           |
| `collector timed out`, `memory limit`, `cancelled` | Runtime containment ended the work.                                           | Reduce scope, use sequential diagnosis, and correct the bounded collector behavior.                    |
| `unsupported collector execution type`             | Internal worker payload was not a supported collector payload.                | This is an engine bug; retain diagnostics and report it with sanitized context.                        |

## Manifest, package, and API errors

| Error family                                                                         | Meaning                                                                            | Correction                                                                                            |
|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `run manifest is missing schema_version` / `unsupported run manifest schema_version` | A caller tried to read an incompatible or malformed manifest.                      | Use a finalized v4 manifest.                                                                          |
| `no output directory exists for run`                                                 | The requested run ID has no matching retained output directory.                    | Verify the ID and retention location; do not create a replacement directory.                          |
| `artifact is not registered` / `artifact exceeds maximum read size`                  | Public API artifact read is not manifest-backed or exceeds its bounded read limit. | Read only catalogued artifacts and request an appropriate bounded limit.                              |
| `package contents do not match ...` / `package integrity verification failed`        | ZIP members, checksums, or catalog differ from the finalized manifest.             | Treat the package as invalid; preserve it for investigation and rerun packaging from intact evidence. |
| `package metadata ... invalid` / `packaged metadata does not match ...`              | Package metadata cannot be decoded or differs from the manifest.                   | Do not distribute it; investigate tampering or a packaging defect.                                    |
| `requested performance report is missing` / `performance report escapes ...`         | A performance-mode run lacks its required report or the path is unsafe.            | Preserve the run and investigate runtime output ownership before retrying.                            |
| `no unique output fingerprint prefix is available`                                   | Existing output paths exhaust all fingerprint prefixes.                            | Preserve existing evidence, choose a new output root, and investigate collisions.                     |

## Maintenance and update errors

| Error family                                                                          | Meaning                                                                          | Correction                                                                           |
|---------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `git is unavailable`, `not a repository`, `origin ...`, `remote ...`                  | Update/development checks cannot establish a valid checkout or reachable remote. | Install Git, use the intended checkout, configure origin, or resolve network access. |
| `--write-manifest requires --next-version`                                            | A persistent integrity-manifest write has no explicit semantic target.           | Provide `--next-version X.Y.Z` or use interactive review.                            |
| `unsupported new-window action` / `new command windows are supported only on Windows` | An update launch action is not allowlisted or host support is absent.            | Use `preflight`, `debug`, or `dev` with `--new-window` on Windows.                   |

When an error is not represented verbatim above, keep its exact message and the manifest/debug JSON. Host API, optional Windows tool, and collector-specific error text is intentionally preserved as the final detail needed to diagnose that source.
