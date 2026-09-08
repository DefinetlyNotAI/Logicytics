# Logicytics

<p align="center">
  <strong>Reliable Windows evidence collection, organized around one verified run at a time.</strong>
</p>

<div style="text-align:center;" align="center">
    <a href="https://github.com/DefinetlyNotAI/Logicytics/issues"><img src="https://img.shields.io/github/issues/DefinetlyNotAI/Logicytics" alt="GitHub Issues"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/tags"><img src="https://img.shields.io/github/v/tag/DefinetlyNotAI/Logicytics" alt="GitHub Tag"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/t/DefinetlyNotAI/Logicytics" alt="GitHub Commit Activity"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/languages"><img src="https://img.shields.io/github/languages/count/DefinetlyNotAI/Logicytics" alt="GitHub Language Count"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/actions"><img src="https://img.shields.io/github/check-runs/DefinetlyNotAI/Logicytics/main" alt="GitHub Branch Check Runs"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics"><img src="https://img.shields.io/github/repo-size/DefinetlyNotAI/Logicytics" alt="GitHub Repo Size"></a>
</div>
<div style="text-align:center;" align="center">
    <a href="https://www.codefactor.io/repository/github/definetlynotai/logicytics"><img src="https://www.codefactor.io/repository/github/definetlynotai/logicytics/badge" alt="GitHub Repo CodeFactor Rating"></a>
    <a href="https://qlty.sh/gh/DefinetlyNotAI/projects/Logicytics"><img src="https://qlty.sh/gh/DefinetlyNotAI/projects/Logicytics/maintainability.svg" alt="Maintainability" /></a>
    <a href="https://api.securityscorecards.dev/projects/github.com/DefinetlyNotAI/Logicytics"><img src="https://api.securityscorecards.dev/projects/github.com/DefinetlyNotAI/Logicytics/badge"  alt="OpenSSF Best Practices Score"/></a>
    <a href="https://www.bestpractices.dev/projects/9451"><img src="https://www.bestpractices.dev/projects/9451/badge" alt="OpenSSF Best Practices Badge"></a>
</div>

Logicytics collects authorized Windows evidence through isolated collectors and leaves a manifest-backed run folder that can be inspected, packaged, and verified. It is built for clear outcomes: a collector can succeed, skip, partially complete, or fail without hiding what happened to the rest of the run.

> Use Logicytics only on systems and data you are authorized to inspect.

## Get started

The installer is the only Logicytics command designed to run outside a virtual environment. It prepares `.venv` and creates the single root configuration file, `logicytics.yaml`.

```powershell
python -m logicytics.cli.installer
.\.venv\Scripts\Activate.ps1
python -m logicytics preflight
```

When preflight is clean, plan before collecting:

```powershell
python -m logicytics plan --profile standard
python -m logicytics run --profile standard --acknowledge-authorization
```

Use the bundled runner for the complete dynamically discovered test suite:

```powershell
.\.venv\Scripts\Activate.ps1
python -m logicytics.cli.tests
```

## CLI reference

Run these commands from the repository root. The normal CLI must run inside
the virtual environment created by the installer. Both module entry points are
equivalent:

```powershell
.venv\Scripts\python.exe -m logicytics --help
.venv\Scripts\python.exe -m logicytics.cli --help
```

The CLI has three standalone utility actions and seven subcommands:

| Invocation                                  | Purpose                                                         |
|---------------------------------------------|-----------------------------------------------------------------|
| python -m logicytics --usage                | Show a formatted interaction summary and create flag_usage.svg. |
| python -m logicytics --modes                | Show a formatted mode summary and save the JSON matrix to disk. |
| python -m logicytics --match TEXT           | Show a formatted match result for natural-language input.       |
| python -m logicytics preflight              | Discover and validate collectors.                               |
| python -m logicytics debug                  | Show formatted diagnostics and write debug.json.                |
| python -m logicytics plan                   | Resolve a dependency-safe plan without collecting evidence.     |
| python -m logicytics run                    | Execute the plan in isolated workers and publish a run.         |
| python -m logicytics collector COLLECTOR_ID | Execute one exact collector independently.                      |
| python -m logicytics update                 | Check Git and optionally apply an update.                       |
| python -m logicytics dev                    | Run repository integrity checks and manage the local manifest.  |

Standalone actions cannot be combined with a subcommand. Use --help after any
action for the parser-generated help.

### Global flags

These flags go before the action name:

| Flag          | Description                                                               |
|---------------|---------------------------------------------------------------------------|
| -h, --help    | Show help and exit.                                                       |
| --config PATH | Load configuration from PATH instead of the root logicytics.yaml.         |
| --usage       | Show local interaction statistics and create the usage graph.             |
| --modes       | Show the mode summary; save the complete matrix to logs/debug/modes.json. |
| --match TEXT  | Show a formatted match result using the local matcher.                    |

All normal command output uses the Logicytics console presentation: a severity
marker, a short message, and readable indented fields. Structured JSON is kept
in the documented artifact files and is never dumped directly to the console.
Each user-facing command clears an interactive terminal before it starts and
leaves one final newline when it returns; redirected and worker-protocol output
is left untouched.
INFO text is white, DEBUG text is gray, warnings are yellow, and failures are
red. Long messages wrap to the terminal width; application log fields are
written as readable continuation lines with millisecond timestamps. Argument
errors, startup failures, installer output, and test-runner output use the same
line-based presentation. Run and collector JSONL files remain private,
redacted machine-readable artifacts for tooling.

### Shared collector flags

These flags are parser-exposed on every subcommand. They affect collector
selection and planning on preflight, debug, plan, run, and collector; update
and dev accept them for CLI consistency but do not execute collection.

| Flag                                      | Description                                                                        |
|-------------------------------------------|------------------------------------------------------------------------------------|
| --profile {minimal,standard,deep,offline} | Select built-in collector membership and access policy.                            |
| --include COLLECTOR_ID                    | Include one exact dotted collector ID; repeat as needed.                           |
| --exclude COLLECTOR_ID                    | Exclude one exact dotted collector ID; repeat as needed.                           |
| --plugins                                 | Enable all valid opt-in plugins for the selected profile.                          |
| --mods                                    | Enable valid sidecar-declared scripts from MODS.                                   |
| --workers COUNT                           | Bound concurrent isolated workers; configuration supplies the default.             |
| --block-capability CAPABILITY             | Block one capability requested by selected collectors; repeat for each capability. |

Collector metadata declares the capabilities each script requests. Declared
capabilities run by default unless blocked with `--block-capability` or the
`runtime.blocked_capabilities` configuration setting. The offline profile still
enforces its network restrictions, and authorization is still required before
collection starts:

| Capability          | Permission                                                    |
|---------------------|---------------------------------------------------------------|
| filesystem_read     | Read files through the filesystem boundary.                   |
| filesystem_write    | Write outside a collector private workspace where permitted.  |
| registry_read       | Read Windows registry data through the registry adapter.      |
| subprocess          | Launch a child process through the process adapter.           |
| network             | Use declared local or remote network access.                  |
| packet_capture      | Capture raw packets; network declaration is not enough.       |
| browser_data        | Read declared local browser profile data.                     |
| sensitive_files     | Read the selected sensitive-file surface.                     |
| private_keys        | Read declared private-key material.                           |
| elevated_privileges | Select collectors requiring administrator privileges.         |

If planning finds a blocked capability, it reports every affected collector and
prints the blocking policy in the error. A script that accesses a capability
without declaring it in `CollectorMetadata.capabilities` is refused by the
isolated worker with the unique `CAPABILITY_DECLARATION_MISMATCH` diagnostic.

### Profiles

| Profile  | Selection policy                                                                                        |
|----------|---------------------------------------------------------------------------------------------------------|
| minimal  | Essential local system, memory, and storage inventory.                                                  |
| standard | Shipped core collectors declaring standard membership.                                                  |
| deep     | Extended declared inventory; sensitive and privileged work still needs explicit selection and approval. |
| offline  | Local-only inventory; network and packet-capture collectors are rejected.                               |

## Preflight

Preflight is read-only. It validates collector filenames, AST boundaries,
metadata, lifecycle contracts, and isolated runtime metadata, then classifies
valid, invalid, and quarantined extensions.

```powershell
python -m logicytics preflight
python -m logicytics preflight --include plugin.example --plugins
python -m logicytics preflight --mods
```

It returns 0 when there are no blocking invalid collectors and 2 when a
blocking validation failure exists. It does not collect evidence.

## Planning

Planning resolves profile membership, explicit selections, dependencies,
capabilities, platform support, offline policy, and administrator requirements.
It does not launch collection workers or create a run-owned evidence tree.

```powershell
python -m logicytics plan --profile minimal
python -m logicytics plan --profile standard
python -m logicytics plan --profile standard --include core.system.system_info --exclude core.network.network_identity
python -m logicytics plan --profile standard --plugins
python -m logicytics plan --profile standard --mods
python -m logicytics plan --profile offline
```

The output is line-based and lists collectors in dependency-safe order.

## Collection runs

run performs preflight, builds the plan, starts isolated workers, retains
manifests, logs, and artifacts, and normally creates a verified ZIP package.

| Run flag                    | Description                                                         |
|-----------------------------|---------------------------------------------------------------------|
| --mode MODE                 | Select a named execution mode.                                      |
| --sequential                | Run one isolated worker at a time.                                  |
| --parallel                  | Use bounded parallel execution; at least two workers are required.  |
| --rerun-from PATH           | Rerun explicit IDs from a finalized run directory or manifest.json. |
| --performance-check         | Run serially and write per-collector duration measurements.         |
| --no-package                | Finalize evidence without creating a ZIP package.                   |
| --reboot                    | Schedule a reboot only after successful packaged output.            |
| --shutdown                  | Schedule a shutdown only after successful packaged output.          |
| --acknowledge-authorization | Confirm authorization; required before collection starts.           |
| --interactive               | Pause at the final status in an interactive command window.         |

Conflicting execution modes are rejected. Reboot and shutdown are mutually
exclusive.

### Named execution modes

| Mode        | Profile  | Scheduling                     | Extensions                 |
|-------------|----------|--------------------------------|----------------------------|
| standard    | standard | Sequential                     | None                       |
| balanced    | standard | Configured bounded parallelism | None                       |
| quick       | minimal  | Configured bounded parallelism | None                       |
| thorough    | deep     | Configured bounded parallelism | None                       |
| offline     | offline  | Configured bounded parallelism | None                       |
| extensions  | standard | Configured bounded parallelism | All valid plugins and MODs |
| non-python  | standard | Configured bounded parallelism | Non-Python MODs only       |
| performance | standard | Sequential                     | Duration report enabled    |

```powershell
python -m logicytics run --mode quick --acknowledge-authorization
python -m logicytics run --mode balanced --acknowledge-authorization
python -m logicytics run --mode performance --acknowledge-authorization
```

### Legacy run aliases

| Alias               | Equivalent mode |
|---------------------|-----------------|
| --default           | standard        |
| --threaded          | balanced        |
| --minimal           | quick           |
| --depth             | thorough        |
| --modded            | extensions      |
| --nopy              | non-python      |
| --performance-check | performance     |

### Direct collector runs

collector runs one exact ID and cannot be combined with profile, include,
exclude, plugin, or MOD selection flags.

```powershell
python -m logicytics collector core.system.system_info --acknowledge-authorization
python -m logicytics collector core.network.network_adapters --acknowledge-authorization
python -m logicytics collector core.network.network_identity --acknowledge-authorization
```

Direct collector runs also accept --block-capability, --no-package, and
--interactive, and retain the same manifest and artifact contracts.

### Reruns

Reruns require a finalized prior run and at least one explicit --include. The
included IDs must exist in the original resolved plan:

```powershell
python -m logicytics run --rerun-from output/data/run-<run-id> --include core.system.system_info --acknowledge-authorization
```

The rerun receives a new run ID and separate evidence directory.

## Diagnostics and maintenance

### Debug

debug shows a formatted diagnostic summary and writes the complete JSON report
to output/logs/debug/debug.json. The report includes configuration,
environment, Python/runtime details, Sysinternals state, preflight counts, and
maintenance checks.

```powershell
python -m logicytics debug
python -m logicytics --config C:\path\to\logicytics.yaml debug
```

### Update

update checks Git availability and repository state. It is read-only unless
--apply is supplied.

| Flag                                  | Description                                                                         |
|---------------------------------------|-------------------------------------------------------------------------------------|
| --apply                               | Run git pull after repository checks.                                               |
| --launch-action {preflight,debug,dev} | Choose a safe action to launch after a successful update check.                     |
| --new-window                          | Launch that action in a separate visible Windows console; requires --launch-action. |
| --performance-check                   | Compatibility flag accepted by the parser; update does not collect evidence.        |

```powershell
python -m logicytics update
python -m logicytics update --apply
python -m logicytics update --launch-action preflight --new-window
```

### Developer checks

dev runs repository organization and integrity checks without collecting
evidence.

| Flag                   | Description                                                |
|------------------------|------------------------------------------------------------|
| --write-manifest       | Write the reviewed local integrity manifest.               |
| --next-version VERSION | Version to record; required for non-interactive writes.    |
| --interactive          | Show checks and prompt before changing the local manifest. |

```powershell
python -m logicytics dev
python -m logicytics dev --write-manifest --next-version 4.0.0
python -m logicytics dev --interactive
```

## Companion CLI scripts

### Installer

The installer is the only Logicytics command designed to run outside a virtual
environment. It creates or reuses the environment and writes the root
logicytics.yaml template.

```powershell
python -m logicytics.cli.installer
python -m logicytics.cli.installer --environment .venv
python -m logicytics.cli.installer --environment C:\Tools\logicytics-venv
python -m logicytics.cli.installer --overwrite-config
```

| Flag               | Description                                                                              |
|--------------------|------------------------------------------------------------------------------------------|
| --environment PATH | Virtual environment directory; default is .venv. Relative paths use the repository root. |
| --overwrite-config | Replace the root YAML template with default values.                                      |

After installation, activate the environment before running a normal command:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the environment is already present, every normal command prints this same
activation instruction instead of running outside it. If `.venv` is absent or
incomplete, it directs you to the installer instead.

### Test runner

The test runner discovers the complete tests package, logs suite lifecycle,
and returns CI-friendly status. It requires the managed virtual environment.

```powershell
.\.venv\Scripts\Activate.ps1
python -m logicytics.cli.tests
python -m logicytics.cli.tests --verbosity 0
python -m logicytics.cli.tests --verbosity 1
python -m logicytics.cli.tests --verbosity 2
```

| Flag                | Description                                      |
|---------------------|--------------------------------------------------|
| --verbosity {0,1,2} | Forward unittest output verbosity; default is 2. |

The test runner returns 0 for a passing suite, 1 for failures/errors, and 2
when the environment or test presentation cannot be prepared.

## What to expect

Each run follows a simple, auditable path:

`request → validated plan → isolated collectors → normalized artifacts → manifest → package`

The console keeps status and progress easy to scan. Lifecycle events are shown with readable messages, while application events are written separately in an aligned, human-readable log with duration and execution context. Evidence and package metadata remain with the run they describe.

Sysinternals is enabled by default. Logicytics detects an existing installation or securely downloads and extracts the official archive to its dedicated application location. Set `maintenance.sysinternals_enabled: false` in `logicytics.yaml` to opt out completely.

## Learn more

The [Logicytics Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki) is the home for setup, usage, configuration, collector development, architecture, outputs, troubleshooting, and security guidance.

For contribution and security policies, see [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Logicytics is released under the [project license](LICENSE).
