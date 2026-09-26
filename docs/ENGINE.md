# How the engine works

The engine is a staged pipeline:

```text
request -> configuration -> preflight -> plan -> isolated workers -> artifacts -> manifest -> package
```

## 1. Request

The CLI or Python API creates an immutable `RunRequest`. It contains the profile, exact includes/excludes, plugin and
plugin enablement, worker limit, blocked capabilities, performance policy, rerun source, output policy, and optional post-run
action. Invalid combinations are rejected before collection.

## 2. Configuration

`logicytics.yaml` is parsed as a strict mapping. Unknown keys, duplicate keys, unsafe paths, invalid numbers,
unsupported collector settings, and oversized files are rejected. The validated `AppConfig` produces a configuration
fingerprint used to prevent stale validation data from being reused.

## 3. Preflight

Discovery walks the source trees, checks filenames and class shape, validates imports and side effects, and runs a
side-effect-free validation worker. A successful cached probe is accepted only when source hash, relative path,
collector kind, interpreter version, contract version, and full configuration identity still match. Invalid candidates
are never cached as successful.

## 4. Plan

The planner selects by profile, exact IDs, dependencies, capability policy, platform, privilege, and resource rules. It
orders dependencies before dependants and chooses bounded parallel or serial execution. A plan has a fingerprint; it is
the stable description of what the run intends to do.

## 5. Worker execution

The supervisor starts eligible collectors in separate processes. A worker receives a `CollectorContext`, calls
validation/prepare/collect/finalize, and returns a typed `CollectorResult`. Heartbeats and progress are persisted.
Timeouts, memory limits, cancellation, crashes, malformed results, and capability violations become explicit statuses
rather than false success.

## 6. Artifact publication

Collectors cannot publish arbitrary files directly. They call `context.artifacts.register_file(...)`. The writer
enforces workspace ownership, declared MIME types, size and file quotas, safe relative paths, SHA-256 calculation, and
evidence kind. Worker scratch space is removed after terminal publication; durable run evidence remains.

## 7. Manifest and package

The manifest records run identity, request, plan, timestamps, configuration identity, collector records, artifacts,
failures, progress, and package metadata. Packaging verifies allowlisted paths, checksums, archive members, and the ZIP
sidecar before reporting a successful package. A manifest-only run is valid when `--no-package` is intentional.

## Status meanings

Collector statuses are `succeeded`, `partial`, `skipped`, `cancelled`, and `failed`. Run statuses are `planned`,
`running`, `succeeded`, `partial`, `failed`, and `cancelled`. A skipped collector is not the same as a failed collector;
read its summary and errors for the reason.
