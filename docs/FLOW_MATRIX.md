# v4 execution-flow verification matrix

This matrix names the executable regression evidence for every supported v4
mode, maintenance action, post-run action, and terminal safety flow. The test
suite verifies that every referenced test remains present and then executes the
tests normally.

| Flow                | Executable evidence                                                        |
|---------------------|----------------------------------------------------------------------------|
| default / standard  | `test_typed_mode_registry_maps_every_user_mode_and_legacy_alias`           |
| threaded / balanced | `test_explicit_execution_modes_control_isolated_worker_overlap`            |
| minimal / quick     | `test_typed_mode_registry_maps_every_user_mode_and_legacy_alias`           |
| deep / thorough     | `test_typed_mode_registry_maps_every_user_mode_and_legacy_alias`           |
| non-Python          | `test_nopy_and_modded_modes_select_declared_mod_types_without_helpers`     |
| performance         | `test_performance_report_is_finalized_before_automatic_packaging`          |
| modded / extensions | `test_mods_require_sidecars_and_run_as_isolated_registered_artifacts`      |
| debug               | `test_dev_writes_explicit_manifest_and_debug_persists_diagnostics`         |
| update              | `test_update_can_explicitly_launch_an_allowlisted_action_in_a_new_window`  |
| usage               | `test_semantic_flag_matching_history_usage_and_graph_are_local_and_opt_in` |
| shutdown and reboot | `test_post_run_actions_are_typed_exclusive_and_require_verified_packaging` |
| cancellation        | `test_cancelled_run_writes_a_recoverable_package_and_manifest`             |
| permission denied   | `test_access_denied_is_skipped_not_failed`                                 |

The mode registry test covers named modes and every historical flag alias. The
overlap test executes real isolated workers. MOD and non-Python tests exercise
sidecar discovery and native adapters. Maintenance tests keep update/debug out of
normal collection. Power actions require a verified package. Cancellation and
permission tests prove these states cannot be misreported as success.
