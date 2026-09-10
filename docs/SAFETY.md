# Safety, privacy, and authorization

## Consent and scope

You are responsible for authorization. Do not collect another person's machine, private keys, saved Wi-Fi keys, browser
profiles, or sensitive files without a clear legal and organizational basis. Treat the run directory and ZIP package as
potentially confidential.

Before a run:

1. Read the plan.
2. Check the collector IDs and profile.
3. Check declared capabilities and sensitive-data categories.
4. Check output location and available disk space.
5. Add `--block-capability` for access you do not want used.
6. Add `--acknowledge-authorization` only after those checks.

## Isolation model

Every selected collector gets an independent worker process, a private workspace, a private temporary directory, an
event channel, a cancellation boundary, and declared resource limits. One failure is recorded for that collector and
does not normally stop independent collectors. The supervisor terminates only the worker process tree it owns when a
timeout or memory limit is exceeded.

Core collectors are discovered from `core/` and selected by profiles. Plugin collectors in `plugins/` and Python MODS in
`MODS/` never run unless explicitly enabled. A plugin or MOD must pass static and runtime preflight before it can enter
a plan.

## Capability gates

Capabilities include `filesystem_read`, `filesystem_write`, `registry_read`, `subprocess`, `network`, `packet_capture`,
`browser_data`, `sensitive_files`, `private_keys`, and `elevated_privileges`. A declaration describes what a collector
may need; it does not grant permission to exceed the worker boundary. A blocked capability wins over a declaration.

## Data handling

Application messages are redacted before they reach console or file sinks. Evidence itself is not a log message: it is
written through the artifact writer and recorded in the manifest with path, MIME type, size, timestamp, and SHA-256. Do
not upload raw evidence to an issue. Sanitize hostnames, usernames, paths, addresses, account identifiers, tokens, and
private data before sharing diagnostics.

The engine does not silently expire completed run evidence. Delete a run only after confirming that its manifest,
package, hash, and logs are no longer needed.

## High-impact options

- `--reboot` and `--shutdown` are mutually exclusive and are scheduled only after durable package publication.
- `--performance-check` forces serial measurement and is intended for measurement, not fastest collection.
- `--plugins` and `--mods` execute additional user-owned code.
- `--usb` changes the source Windows installation and rejects unsafe output/cache/temp placement on that disk.
- `--update --apply` performs `git pull`; use it only when repository state and remote are understood.
