"""Keep the documented release flow matrix connected to executable tests."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


class FlowMatrixTests(unittest.TestCase):
    def test_every_required_flow_has_executable_regression_evidence(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        test_names: set[str] = set()
        for path in (project_root / "tests").glob("test_*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                if node.name.startswith("test_"):
                    test_names.add(node.name)
        required = {
            "default": "test_typed_mode_registry_maps_every_user_mode_and_legacy_alias",
            "threaded": "test_explicit_execution_modes_control_isolated_worker_overlap",
            "minimal": "test_typed_mode_registry_maps_every_user_mode_and_legacy_alias",
            "deep": "test_typed_mode_registry_maps_every_user_mode_and_legacy_alias",
            "non-python": "test_nopy_and_modded_modes_select_declared_mod_types_without_helpers",
            "performance": "test_performance_report_is_finalized_before_automatic_packaging",
            "modded": "test_mods_require_sidecars_and_run_as_isolated_registered_artifacts",
            "debug": "test_dev_writes_explicit_manifest_and_debug_persists_diagnostics",
            "update": "test_update_can_explicitly_launch_an_allowlisted_action_in_a_new_window",
            "usage": "test_semantic_flag_matching_history_usage_and_graph_are_local_and_opt_in",
            "shutdown/reboot": "test_post_run_actions_are_typed_exclusive_and_require_verified_packaging",
            "cancellation": "test_cancelled_run_writes_a_recoverable_package_and_manifest",
            "permission-denied": "test_access_denied_is_skipped_not_failed",
        }
        documented = (project_root / "FLOW_MATRIX.md").read_text(encoding="utf-8")
        for flow, test_name in required.items():
            with self.subTest(flow=flow):
                self.assertIn(test_name, test_names)
                self.assertIn(f"`{test_name}`", documented)


if __name__ == "__main__":
    unittest.main()
