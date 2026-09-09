"""Regression coverage for interaction history and rendered analysis artifacts."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logicytics.cli import CLI, main
from logicytics.module.configuration import (
    load_config,
)
from logicytics.module.errors import PlanError
from logicytics.module.interaction import (
    load_history,
    match_flag,
    usage_statistics,
    write_usage_graph,
)


class InteractionTests(unittest.TestCase):
    """Local semantic matching, history, and usage behavior."""

    def test_semantic_flag_matching_history_usage_and_graph_are_local_and_opt_in(self) -> None:
        """Natural-language actions use configured matching and persist only with consent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()
            with (
                patch.object(CLI, "project_root", return_value=root),
                patch(
                    "sys.stderr",
                    output,
                ),
            ):
                self.assertEqual(0, main(["--match", "run a quick basic collection"]))
            rendered = output.getvalue()
            self.assertIn("Match result", rendered)
            self.assertIn("Matched flag: minimal", rendered)
            self.assertIn("History persisted: no", rendered)
            self.assertNotIn("{", rendered)
            self.assertNotIn('"matched_flag"', rendered)
            self.assertFalse((root / ".cache" / "interaction_history.json.gz").exists())

            (root / "logicytics.yaml").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "interaction": {
                            "history_enabled": True,
                            "similarity_threshold": 0.5,
                            "model_name": "stdlib-test-model",
                            "model_debug": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch.object(CLI, "project_root", return_value=root),
                patch(
                    "sys.stderr",
                    output,
                ),
            ):
                self.assertEqual(0, main(["--match", "an exhaustive slow scan"]))
            rendered = output.getvalue()
            self.assertIn("Match result", rendered)
            self.assertIn("Matched flag: depth", rendered)
            self.assertIn("Model: stdlib-test-model", rendered)
            self.assertIn("History persisted: yes", rendered)
            self.assertNotIn("{", rendered)
            history_path = root / ".cache" / "interaction_history.json.gz"
            history = load_history(history_path)
            self.assertEqual(1, len(history))
            self.assertIn("timestamp", history[0])
            self.assertIn("device_name", history[0])

            output = io.StringIO()
            with (
                patch.object(CLI, "project_root", return_value=root),
                patch(
                    "sys.stderr",
                    output,
                ),
            ):
                self.assertEqual(0, main(["--usage"]))
            rendered = output.getvalue()
            self.assertIn("Interaction usage", rendered)
            self.assertIn("Total interactions: 1", rendered)
            self.assertIn("depth: 1", rendered)
            self.assertNotIn("{", rendered)
            graph_path = root / ".cache" / "flag_usage.svg"
            self.assertTrue(graph_path.exists())
            self.assertIn("<svg", graph_path.read_text(encoding="utf-8"))

    def test_semantic_matching_can_fall_back_to_prior_inputs(self) -> None:
        """A weak direct match may reuse a stronger local historical association."""
        history = [{"input": "collect the strange moon report", "matched_flag": "debug"}]
        result = match_flag(
            "collect strange moon report",
            threshold=0.99,
            model_name="stdlib-sequence-matcher",
            history=history,
        )
        self.assertEqual("debug", result.matched_flag)
        self.assertEqual("history", result.source)
        statistics = usage_statistics([{**history[0], "accuracy": 0.9, "device_name": "host"}])
        self.assertEqual("collect the strange moon report", statistics["common_input"])

    def test_usage_graph_contains_only_current_nonzero_command_and_mode_counts(
        self,
    ) -> None:
        """The graph must match recorded usage rather than an obsolete fixed flag list."""
        with tempfile.TemporaryDirectory() as temporary:
            graph_path = Path(temporary) / "flag_usage.svg"
            write_usage_graph(
                graph_path,
                {"per_flag_frequency": {"config": 3, "quick": 2, "thorough": 1}},
            )

            graph = graph_path.read_text(encoding="utf-8")
            self.assertIn("Logicytics command and mode usage", graph)
            self.assertIn(">config<", graph)
            self.assertIn(">quick<", graph)
            self.assertIn(">thorough<", graph)
            self.assertNotIn(">default<", graph)

    def test_interaction_configuration_rejects_unsafe_or_ambiguous_values(self) -> None:
        """Matching and persistence policy must be typed before any history is opened."""
        invalid_sections = (
            {"similarity_threshold": 1.1},
            {"history_enabled": "yes"},
            {"model_debug": 1},
            {"model_name": "bad\nname"},
            {"unknown": True},
        )
        for section in invalid_sections:
            with self.subTest(section=section), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "logicytics.yaml").write_text(
                    json.dumps({"schema_version": 4, "interaction": section}),
                    encoding="utf-8",
                )
                with self.assertRaises(PlanError):
                    load_config(root)


if __name__ == "__main__":
    unittest.main()
