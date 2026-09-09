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

Logicytics is a Windows evidence collection framework. It validates every collector before use, runs each one in isolation, and keeps the result in a manifest-backed run folder. A collector can succeed, skip, or fail without obscuring the rest of the verified run.

> Use Logicytics only on systems and data you are authorized to inspect.

## Start here

The installer is the only command intended to run outside the managed virtual environment. Run it once from the repository root:

```powershell
python -m logicytics.cli.installer
```

Then activate the environment and check the installation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m logicytics preflight
```

When preflight reports no invalid collectors, make a plan and run it:

```powershell
python -m logicytics plan --profile standard
python -m logicytics run --profile standard --acknowledge-authorization
```

If a normal command says the environment is missing, run the installer. If it says the environment is not active, run `.\.venv\Scripts\Activate.ps1` first.

## Choose a run

Every run validates collectors, records a manifest, and packages the result unless `--no-package` is supplied.

| Need | Command |
|---|---|
| Fast local inventory | `python -m logicytics run --mode quick --acknowledge-authorization` |
| Everyday collection | `python -m logicytics run --mode balanced --acknowledge-authorization` |
| Deterministic sequential collection | `python -m logicytics run --mode standard --acknowledge-authorization` |
| Local-only collection | `python -m logicytics run --mode offline --acknowledge-authorization` |
| Extended collection | `python -m logicytics run --mode thorough --acknowledge-authorization` |
| Duration report | `python -m logicytics run --mode performance --acknowledge-authorization` |

`thorough` can include administrator-only collectors. Start an elevated shell when the plan reports that requirement. See every available mode with `python -m logicytics --modes`.

## Where results go

Each run receives its own directory under `output/data/`:

```text
output/data/run/<fingerprint>/
  manifest.json          # status, collector results, and artifact catalog
  artifacts/             # collected evidence
  logs/                  # run and collector JSONL events
  reports/               # generated summaries

output/data/zip/<fingerprint>.zip
output/data/hashes/<fingerprint-first-8>.zip.sha256
```

The console is intentionally brief. Use `manifest.json` to inspect a run, the package hash to verify a package, and `output/logs/Logicytics.log` for the human-readable application log. The fingerprint is a SHA-256 identity derived from the immutable run ID. Set `logging.level: DEBUG` in `logicytics.yaml` when you need detailed worker lifecycle information and file call sites.

## Useful commands

```powershell
# Revalidate every collector instead of reusing cached preflight probes
python -m logicytics preflight --invalidate-cache

# Inspect a plan without collecting evidence
python -m logicytics plan --profile standard

# Run one collector only
python -m logicytics collector core.system.system_info --acknowledge-authorization

# Run the complete test suite
python -m logicytics.cli.tests

# See diagnostics, configuration, and maintenance state
python -m logicytics debug
```

Use `python -m logicytics --help` or append `--help` to any command for its full flags.

## Configuration and extensions

`logicytics.yaml` is the single user configuration file. It controls output locations, worker limits, logging, optional Sysinternals setup, and declared collector settings. Keep credentials and secrets out of it.

Core collectors are shipped and validated as part of the application. Plugins and MODs are opt-in and must pass the same validation boundary before they can run.

- [Configuration reference](docs/CONFIGURATION.md)
- [Output contract](docs/OUTPUTS.md)
- [MODS guide](docs/MODS.md)
- [Migration guide](docs/MIGRATION.md)
- [Flow matrix](docs/FLOW_MATRIX.md)

## Help and contributing

The [Logicytics Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki) covers setup, troubleshooting, collector development, architecture, and security in more depth.

For changes to Logicytics, read [CONTRIBUTING.md](CONTRIBUTING.md). Please also review [SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Logicytics is released under the [project license](LICENSE).
