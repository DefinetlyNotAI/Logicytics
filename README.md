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
python -m logicytics preflight
```

When preflight is clean, plan before collecting:

```powershell
python -m logicytics plan --profile standard
python -m logicytics run --profile standard --acknowledge-authorization
```

Use the bundled runner for the complete dynamically discovered test suite:

```powershell
python -m logicytics.cli.tests
```

## What to expect

Each run follows a simple, auditable path:

`request → validated plan → isolated collectors → normalized artifacts → manifest → package`

The console keeps status and progress easy to scan. Detailed command output is boxed; application events are written separately in an aligned, human-readable log. Evidence and package metadata remain with the run they describe.

Sysinternals is enabled by default. Logicytics detects an existing installation or securely downloads and extracts the official archive to its dedicated application location. Set `maintenance.sysinternals_enabled: false` in `logicytics.yaml` to opt out completely.

## Learn more

The [Logicytics Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki) is the home for setup, usage, configuration, collector development, architecture, outputs, troubleshooting, and security guidance.

For contribution and security policies, see [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Logicytics is released under the [project license](LICENSE).
