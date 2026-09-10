# Execution flow checklist

| Flow                        | How to exercise it                                                                  |
|-----------------------------|-------------------------------------------------------------------------------------|
| Preflight                   | `python -m logicytics preflight`                                                    |
| Plan without collection     | `python -m logicytics plan --mode standard`                                         |
| One collector               | `python -m logicytics collector ID --acknowledge-authorization`                     |
| Bounded parallel run        | `python -m logicytics run --mode balanced --parallel --acknowledge-authorization`   |
| Deterministic run           | `python -m logicytics run --mode standard --sequential --acknowledge-authorization` |
| Optional extensions         | Add `--plugins` or `--mods` after reviewing them                                    |
| Performance measurement     | Add `--performance-check` to `run`                                                  |
| Manifest-only output        | Add `--no-package` to `run` or `collector`                                          |
| Rerun selected IDs          | `run --include ID --rerun-from PATH`                                                |
| Cancellation                | Interrupt an active run and inspect its manifest                                    |
| Permission/optional feature | Inspect the skipped record and remediation                                          |
| Maintenance                 | `debug`, `update`, or `dev`                                                         |

The executable regression anchors are:

- `test_typed_mode_registry_maps_every_user_mode_and_legacy_alias`
- `test_explicit_execution_modes_control_isolated_worker_overlap`
- `test_performance_report_is_finalized_before_automatic_packaging`
- `test_mods_require_sidecars_and_run_as_isolated_registered_artifacts`
- `test_dev_writes_explicit_manifest_and_debug_persists_diagnostics`
- `test_update_can_explicitly_launch_an_allowlisted_action_in_a_new_window`
- `test_semantic_flag_matching_history_usage_and_graph_are_local_and_opt_in`
- `test_post_run_actions_are_typed_exclusive_and_require_verified_packaging`
- `test_cancelled_run_writes_a_recoverable_package_and_manifest`
- `test_access_denied_is_skipped_not_failed`

The automated suite is the authoritative regression evidence for these flows. Use [Verification](VERIFICATION.md) for
the standard commands.
