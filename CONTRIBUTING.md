# Contributing to Logicytics

Logicytics v4 is a Windows-focused evidence collector with strict authorization,
isolation, output, and compatibility contracts. Keep changes focused and preserve
those contracts. Use the [issue tracker](https://github.com/DefinetlyNotAI/Logicytics/issues)
for reproducible bugs and scoped feature proposals.

## Development setup

Requirements:

- Windows for live collector and platform integration checks.
- Python 3.11 or later. The v4 engine has no third-party runtime dependency.
- Git for developer integrity and explicit update actions.

From a fresh checkout:

```powershell
python -m logicytics preflight
python -m unittest discover -v
python -m compileall -q logicytics core tests
```

`preflight` must report no invalid core collector. Some live Windows features such
as WMIC, BitLocker, or Sysinternals are optional; absence must produce an explicit
skip or availability result rather than breaking unrelated collection.

## Architecture boundaries

The canonical pipeline is:

`request -> validated plan -> isolated collectors -> registered artifacts -> manifest -> package`

- `logicytics/cli.py` parses and renders. It does not collect evidence.
- `logicytics/planner.py` resolves one deterministic plan from immutable
  `RunRequest` policy and validated metadata.
- `logicytics/runtime.py` owns per-run and per-worker lifecycle, cancellation,
  retries, timeouts, failure aggregation, and post-run actions.
- `logicytics/platform_adapters.py` owns host command, process, registry,
  filesystem, network, privilege, and Win32 access. Collectors must use these
  injectable seams instead of importing host APIs directly.
- `logicytics/artifacts.py` is the only publication path from collector
  workspaces into the run artifact catalog.
- `logicytics/packaging.py` consumes the finalized catalog; it never scans source
  directories for arbitrary files.
- Maintenance, debug, update, developer, and usage behavior stays separate from
  normal collection.

Importing `logicytics` must not start collection, load the supervisor, or create
files. Do not add global mutable run state, implicit current-directory behavior,
collector-to-collector calls, or a second execution/output path.

## Core collector changes

Each `core/<specialty>/<collector_name>.py` module owns exactly one public
`<CollectorName>Collector` and one primary job. Split a feature into a new ID when
it needs a different capability, privilege level, network reach, sensitive-data
category, timeout/cost policy, or output contract.

A collector must:

- inherit `CoreCollector` and provide fully typed, documented `metadata`,
  `validate`, `prepare`, `collect`, `finalize`, and `cleanup` behavior;
- use an ID, class name, file name, and `Specialty` that agree;
- declare platform, capabilities, privilege, network access, sensitivity,
  profiles, dependencies, scheduling policy, timeouts, retries, memory/output
  limits, artifact count, and every MIME type it can publish;
- check `context.is_cancelled` before work and during every long loop, query,
  capture, copy, or traversal;
- write only beneath `context.workspace`, report structured progress, register
  every result through `context.artifacts`, and return a typed result;
- turn expected absence or permission denial into an actionable `skipped` or
  `partial` result without hiding an actual failure;
- avoid `print`, `exit`, nested worker pools, mutable globals, repository writes,
  package-wide cleanup, secrets in logs, and unbounded reads or subprocess output.

Run the collector directly and through the orchestrator. Add mocked Windows
responses, cancellation coverage, and output-contract evidence. If structured
bytes change intentionally, update the relevant golden file in `tests/golden/`
and explain why.

## Plugins and MODs

Plugins implement the typed `PluginCollector` contract and remain opt-in. MODs
use a sidecar declaration and may be Python, PowerShell, batch, or executable
payloads. Neither extension type may bypass preflight, planning, capability
approval, isolated workspaces, artifact registration, or package filtering.

Read [MODS.md](docs/MODS.md) before changing discovery or extension behavior. Read
[MIGRATION.md](docs/MIGRATION.md) before changing a legacy flag, schema migration,
historical `CODE` evidence import, or MOD adapter. Compatibility code must remain
a bounded translation into the canonical v4 model.

## Configuration changes

Configuration changes must preserve the strict schema in
[CONFIGURATION.md](docs/CONFIGURATION.md). Update the field reference and its
parser-backed tests in the same commit as any setting, default, bound, migration
alias, or source-precedence change.

Profiles, modes, include/exclude selections, plugin/MOD enablement, capability
approval, authorization acknowledgement, scheduling overrides, reruns, package
policy, and post-run power actions are invocation-only `RunRequest` behavior.
They do not belong in persistent configuration.

## Evidence, security, and compatibility

- Update [OUTPUTS.md](docs/OUTPUTS.md) when a filename, MIME type, package path,
  evidence kind, or retention rule changes.
- Keep source, executables, models, configuration secrets, caches, and library
  internals out of evidence packages.
- Preserve reproducible package hashes and manifest schema validation.
- Add explicit capabilities for sensitive or elevated access. Never weaken
  authorization to make a test pass.
- Follow [SECURITY.md](SECURITY.md) for vulnerability reports. Do not put secrets
  or real private evidence in issues, fixtures, logs, or commits.

## Testing expectations

Use the narrowest relevant tests while iterating, then run the complete gates
before opening a pull request:

```powershell
python -m unittest discover -v
python -m compileall -q logicytics core tests
python -m logicytics preflight
git diff --check
```

On Windows, also run:

```powershell
python -m unittest tests.test_windows_integration -v
```

The suite includes static preflight, lifecycle/cancellation checks, mocked
collector responses, golden output bytes, flow/mode matrices, package/hash
reproduction, documentation contracts, and bounded live Windows probes. A build
or compile check alone is not sufficient evidence.

## Documentation changes

Keep user and developer documentation synchronized with behavior:

- [README.md](README.md): installation, quick start, CLI, permissions, and
  troubleshooting.
- [CONFIGURATION.md](docs/CONFIGURATION.md): every persistent setting and migration.
- [OUTPUTS.md](docs/OUTPUTS.md): artifact and retention contracts.
- [MODS.md](docs/MODS.md): extension contract.
- [MIGRATION.md](docs/MIGRATION.md): supported compatibility boundary.
- [FLOW_MATRIX.md](docs/FLOW_MATRIX.md): executable flow evidence.
- [FEATURE_STATUS.md](docs/FEATURE_STATUS.md): TODO ownership and completion evidence.
- [V4_RELEASE.md](docs/V4_RELEASE.md): v4 recreation scope and release verification.

The repository wiki is complementary documentation, not a substitute for the
versioned contract files required to review a change.

## Issues and pull requests

A useful bug report includes the Logicytics version, Windows edition/build,
Python version, command and sanitized configuration, expected and actual result,
exit code, relevant redacted logs, and exact reproduction steps. State whether
the process was elevated and whether an optional Windows feature was installed.

Pull requests should:

- solve one coherent problem and avoid unrelated edits;
- use conventional commit subjects such as `feat:`, `fix:`, `refactor:`,
  `test:`, or `docs:` with a detailed body;
- include tests and documentation proportional to the changed contract;
- preserve unrelated work and never include generated evidence or secrets;
- pass the complete verification gates above; and
- comply with the [Developer Certificate of Origin](DCO.md),
  [Code of Conduct](CODE_OF_CONDUCT.md), and repository license.

By contributing code, you agree to license it under the [MIT License](LICENSE).
Documentation contributions use the repository's stated documentation license.
