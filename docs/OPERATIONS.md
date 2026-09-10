# Operations guide

This guide turns the command reference into an operating procedure. It is not a substitute for organizational authorization, incident procedures, or evidence retention policy. It helps an authorized operator make the collection request small, reviewable, and reproducible.

## Operating model

Every request follows the same path:

```text
scope decision -> preflight -> plan review -> explicit authorization -> isolated run -> manifest review -> evidence handoff
```

The command-line interface does not infer authorization from a profile. The `--acknowledge-authorization` flag is required for collection because the final scope decision belongs to the operator. It must follow, not replace, a review of the plan.

## Choose the smallest useful request

Use a mode only when its complete membership fits the question. Inspect the live matrix with:

```powershell
python -m logicytics --modes
```

Then create a non-collecting plan. The five modes are `quick`, `balanced`, `standard`, `offline`, and `thorough`.

```powershell
python -m logicytics plan --mode standard
```

Use a direct collector for a single known question, such as host inventory:

```powershell
python -m logicytics collector core.system.system_info --acknowledge-authorization
```

Use repeated `--include` and `--exclude` when the standard profile is close but not exact. Keep an exported plan or the terminal transcript with the case notes so another reviewer can reproduce the scope.

## Review capabilities before authorization

Capabilities describe access a collector may request. They are not just labels: a requested block prevents matching collectors from entering the plan. The available CLI capability values are:

- `filesystem_read`, `filesystem_write`, `registry_read`, and `subprocess`
- `network` and `packet_capture`
- `browser_data`, `sensitive_files`, and `private_keys`
- `elevated_privileges`

For a restricted collection, request explicit blocks and plan again:

```powershell
python -m logicytics plan --mode thorough `
  --block-capability browser_data `
  --block-capability sensitive_files `
  --block-capability private_keys `
  --block-capability packet_capture
```

Configuration can also define persistent blocks. Invocation blocks add to those configured restrictions; they do not weaken them. Read [Safety](SAFETY.md) and the catalog entry for every sensitive collector you leave selected.

## Run and observe

Run the reviewed request with acknowledgement:

```powershell
python -m logicytics run --mode standard `
  --acknowledge-authorization
```

The runtime gives each collector its own worker process, workspace, temporary directory, event channel, and resource boundary. An independent collector failure is recorded rather than normally terminating unrelated collectors. That is why an apparently successful run must still be inspected collector by collector.

For diagnosis, make concurrency explicit:

```powershell
python -m logicytics run --mode standard --sequential `
  --acknowledge-authorization
```

`--performance-check` performs serial measurement and saves per-collector durations; it is a measurement mode, not a shortcut. `--no-package` leaves the durable run directory and manifest but intentionally omits the ZIP. `--reboot` and `--shutdown` are high-impact post-run actions: they are mutually exclusive and only occur after successful package publication.

## Extensions and removable media

Plugins and MODS are opt-in because they are user-owned code. Include them only after their source, metadata or sidecar, and capability declarations have been reviewed:

```powershell
python -m logicytics preflight --plugins --mods
python -m logicytics plan --mode standard --plugins --mods
```

Use `--usb [DRIVE]` only when deliberately targeting a Windows installation on removable storage. The engine rejects unsafe output, cache, and temporary paths on that target. Confirm the chosen drive and keep output on separately approved storage before authorizing the run.

## Handle the outcome

Exit code is a routing signal, not the full result:

| Exit code | Meaning | Operator action |
| --- | --- | --- |
| `0` | Requested command completed successfully | Review the manifest and artifacts before handoff. |
| `1` | Collection completed with a non-success run status | Review failed, partial, skipped, or cancelled records; preserve useful artifacts. |
| `2` | Command, configuration, preflight, or planning failure | Correct the indicated input or source; do not bypass the gate. |
| `130` | Interrupted | Locate the manifest and determine what was safely finalized before retrying. |

Finish every collection with [Evidence Review](EVIDENCE_REVIEW.md). If an error is not self-explanatory, follow [Troubleshooting](TROUBLESHOOTING.md) and retain only sanitized diagnostic material for support.
