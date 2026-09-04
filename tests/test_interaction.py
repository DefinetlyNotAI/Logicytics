from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logicytics.cli import CLI, main
from logicytics.configuration import (
    load_config,
)
from logicytics.errors import PlanError
from logicytics.interaction import load_history, match_flag, usage_statistics


class InteractionTests(unittest.TestCase):
    """Local semantic matching, history, and usage behavior."""

    def test_semantic_flag_matching_history_usage_and_graph_are_local_and_opt_in(self) -> None:
        """Natural-language actions use configured matching and persist only with consent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()
            with patch.object(CLI, "project_root", return_value=root), patch(
                    "sys.stdout",
                    output,
            ):
                self.assertEqual(0, main(["--match", "run a quick basic collection"]))
            payload = json.loads(output.getvalue())
            self.assertEqual("minimal", payload["matched_flag"])
            self.assertFalse(payload["history_persisted"])
            self.assertFalse((root / "output" / "data" / "interaction_history.json.gz").exists())

            (root / "logicytics.json").write_text(
                json.dumps({
                    "schema_version": 4,
                    "interaction": {
                        "history_enabled": True,
                        "similarity_threshold": 0.5,
                        "model_name": "stdlib-test-model",
                        "model_debug": True,
                    },
                }),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch.object(CLI, "project_root", return_value=root), patch(
                    "sys.stdout",
                    output,
            ):
                self.assertEqual(0, main(["--match", "an exhaustive slow scan"]))
            payload = json.loads(output.getvalue())
            self.assertEqual("depth", payload["matched_flag"])
            self.assertEqual("stdlib-test-model", payload["model_debug"]["model_name"])
            history_path = root / "output" / "data" / "interaction_history.json.gz"
            history = load_history(history_path)
            self.assertEqual(1, len(history))
            self.assertIn("timestamp", history[0])
            self.assertIn("device_name", history[0])

            output = io.StringIO()
            with patch.object(CLI, "project_root", return_value=root), patch(
                    "sys.stdout",
                    output,
            ):
                self.assertEqual(0, main(["--usage"]))
            usage = json.loads(output.getvalue())
            self.assertEqual(1, usage["total_interactions"])
            self.assertEqual(1, usage["per_flag_frequency"]["depth"])
            graph_path = Path(usage["graph_path"])
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
                (root / "logicytics.json").write_text(
                    json.dumps({"schema_version": 4, "interaction": section}),
                    encoding="utf-8",
                )
                with self.assertRaises(PlanError):
                    load_config(root)


if __name__ == "__main__":
    unittest.main()
