# Installation and environment

Logicytics is developed and exercised as a Windows-focused application. The v4 runtime has no required third-party
Python dependency, but it deliberately uses a repository-local virtual environment so commands run with a known
interpreter and do not depend on unrelated packages installed for another project.

## Requirements

- A local checkout of this repository.
- Python 3.11 or later, available as `python` for the bootstrap command.
- Windows for live collector behavior and Windows integration checks.
- Enough trusted disk space for the configured output location and any ZIP package. Evidence output is not a cache; plan
  its storage accordingly.

Git is needed for the `update` and `dev` maintenance actions, but not for a normal already-checked-out collection run.

## Bootstrap or repair the environment

From the repository root, run the installer:

```powershell
python -m logicytics.cli.installer
```

It creates `.venv` if it does not exist, otherwise it reuses it. It also creates the root `logicytics.yaml` template if
it is absent. To intentionally replace that template, use the explicit destructive configuration option:

```powershell
python -m logicytics.cli.installer --overwrite-config
```

Back up or version-control a customized configuration before using `--overwrite-config`; it replaces the root template.

## Run the supported interpreter

Every normal application command requires the managed environment.
The most reliable form is the explicit interpreter path:

```powershell
.\.venv\Scripts\python.exe -m logicytics preflight
```

You may activate the environment for an interactive shell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m logicytics preflight
```

If activation is blocked by PowerShell policy, do not weaken the policy just for this tool.
Continue using `.\.venv\Scripts\python.exe`; it produces the same managed-environment execution.

## Confirm the installation

Run the following in order:

```powershell
python -m logicytics --help
python -m logicytics preflight
```

The first command verifies CLI invocation.
The second verifies the selected collector sources and prerequisites without collecting evidence.
Resolve any configuration or invalid-source result before attempting a plan or run.

## Common environment mistakes

| Symptom                                 | Cause and correction                                                                                                                                     |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| "Must run inside a virtual environment" | Use `.\.venv\Scripts\python.exe` for the normal CLI or activate via `.\.venv\Scripts\Activate.ps1`. Only the installer runs from the system interpreter. |
| `python` is not recognized              | Install a supported Python version or invoke its known executable for the installer; after bootstrap use the local `.venv` path.                         |
| Activation is denied                    | Do not rely on activation. Use the explicit interpreter path.                                                                                            |
| A custom config disappeared             | The installer was run with `--overwrite-config`; restore the backed-up configuration and preflight it before use.                                        |
| Preflight finds invalid extension code  | Correct or remove the named extension source. Extensions are intentionally not trusted just because their files exist.                                   |

For configuration fields and safe path rules, read [Configuration](CONFIGURATION.md).
For collection scope after installation, continue to [Getting Started](GETTING_STARTED.md).
