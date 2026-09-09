# Development guide

## Setup

Use the repository's managed environment and read `CONTRIBUTING.md`, `SECURITY.md`, and [Contracts](CONTRACTS.md) before changing engine or collector code. Keep the standard-library-first installer path intact.

## Change boundaries

Engine changes belong under `logicytics/module/`; contract changes belong in `logicytics/contracts.py`; shipped collection belongs under the matching `core/<specialty>/` directory; user extensions belong in `plugins/` or `MODS/`; documentation changes belong in `docs/`. Do not commit collected evidence, caches, credentials, or local manifests.

## Adding a core collector

Choose one specialty and exact ID, implement the lifecycle contract, declare all access and limits, add focused tests, update the catalog and configuration guide if settings are introduced, and verify preflight plus packaging. Core collectors are automatically discoverable; do not add ad hoc registration tables unless the architecture requires it.

## Documentation changes

Use stable concepts instead of release-specific marketing. Every command, field, capability, collector ID, and output claim should be traceable to live parser, contract, metadata, or test evidence. The wiki workflow publishes the contents of `docs/` after a default-branch documentation change.

## Commits

Use a focused Conventional Commit such as `docs: replace operator manual` or `ci: publish documentation to wiki`. Inspect `git diff` and `git status` before staging so unrelated work remains untouched.
