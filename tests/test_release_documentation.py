"""Executable release-documentation and feature-ownership contracts for v4.0."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

from logicytics.contracts import Specialty
from logicytics.discovery import preflight
from logicytics.modes import EXECUTION_MODES


class ReleaseDocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parent.parent
        cls.readme = (cls.root / "README.md").read_text(encoding="utf-8")
        cls.release = (cls.root / "V4_RELEASE.md").read_text(encoding="utf-8")
        cls.status = (cls.root / "FEATURE_STATUS.md").read_text(encoding="utf-8")
        cls.todo = (cls.root / "TODO.md").read_text(encoding="utf-8")

    def test_release_identity_is_final_and_no_provisional_claim_remains(self) -> None:
        metadata = tomllib.loads((self.root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual("4.0.0", metadata["project"]["version"])
        self.assertIn("complete recreation", self.release.casefold())
        self.assertIn("complete recreation", self.readme.casefold())
        for provisional in (
            "being rebuilt",
            "early v4 implementation",
            "broader collection catalog is being rebuilt",
            "to be released",
        ):
            with self.subTest(provisional=provisional):
                self.assertNotIn(provisional, self.readme.casefold())
                self.assertNotIn(
                    provisional,
                    (self.root / "SECURITY.md").read_text(encoding="utf-8").casefold(),
                )

    def test_release_covers_every_specialty_mode_and_output_media_type(self) -> None:
        report = preflight(self.root)
        self.assertEqual((), report.invalid)
        core = [candidate for candidate in report.valid if candidate.kind.value == "core"]
        self.assertEqual(66, len(core))
        for specialty in Specialty:
            with self.subTest(specialty=specialty.value):
                self.assertIn(f"`{specialty.value}`", self.release)
        for mode in EXECUTION_MODES:
            with self.subTest(mode=mode):
                self.assertIn(f"`{mode}`", self.release)
        media_types = {
            media_type
            for candidate in core
            if candidate.metadata is not None
            for media_type in candidate.metadata.output_media_types
        }
        for media_type in media_types:
            with self.subTest(media_type=media_type):
                self.assertIn(f"`{media_type}`", self.release)

    def test_every_todo_section_has_exactly_one_owner_status_and_evidence_row(self) -> None:
        headings = [
            line.lstrip("#").strip()
            for line in self.todo.splitlines()
            if re.match(r"^#{2,4} ", line)
        ]
        rows: dict[str, tuple[str, str, str]] = {}
        for line in self.status.splitlines():
            if not line.startswith("| ") or line.startswith("| ---"):
                continue
            columns = [column.strip() for column in line.strip("|").split("|")]
            if columns[0] == "TODO section":
                continue
            self.assertEqual(4, len(columns), line)
            self.assertNotIn(columns[0], rows)
            rows[columns[0]] = (columns[1], columns[2], columns[3])
        self.assertEqual(headings, list(rows))
        for heading, (owner, status, evidence) in rows.items():
            with self.subTest(heading=heading):
                self.assertTrue(owner)
                self.assertEqual("Complete", status)
                self.assertTrue(evidence)

    def test_every_todo_item_is_complete_and_wiki_evidence_is_recorded(self) -> None:
        unchecked = [
            line.removeprefix("- [ ] ")
            for line in self.todo.splitlines()
            if line.startswith("- [ ] ")
        ]
        self.assertEqual([], unchecked)
        self.assertIn("GitHub wiki", self.release)
        self.assertIn("eaf7628", self.release)
        self.assertIn("eaf7628", self.status)

    def test_user_and_contributor_guides_cover_the_release_entry_points(self) -> None:
        documents = (
            "CONFIGURATION.md", "CONTRIBUTING.md", "OUTPUTS.md", "MODS.md",
            "MIGRATION.md", "FLOW_MATRIX.md", "FEATURE_STATUS.md", "V4_RELEASE.md",
            "SECURITY.md", "CODE_OF_CONDUCT.md",
        )
        for document in documents:
            with self.subTest(document=document):
                self.assertIn(document, self.readme)
        for option in (
            "--config", "--match", "--modes", "--profile", "--include", "--exclude",
            "--plugins", "--mods", "--workers", "--allow-capability", "--rerun-from",
            "--sequential", "--parallel", "--no-package", "--reboot", "--shutdown",
            "--acknowledge-authorization", "--interactive", "--apply",
        ):
            with self.subTest(option=option):
                self.assertIn(option, self.readme)
        contributor = (self.root / "CONTRIBUTING.md").read_text(encoding="utf-8")
        for requirement in (
            "Development setup", "Architecture boundaries", "Core collector changes",
            "Plugins and MODs", "Configuration changes", "Testing expectations",
            "conventional commit", "Developer Certificate of Origin",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, contributor)

    def test_output_location_documentation_matches_run_owned_packaging(self) -> None:
        migration = (self.root / "MIGRATION.md").read_text(encoding="utf-8")
        self.assertIn("canonical packages are run-owned", self.readme)
        self.assertIn("each run's `packages/` and", migration)
        self.assertIn("not\npopulated with duplicate evidence", migration)


if __name__ == "__main__":
    unittest.main()
