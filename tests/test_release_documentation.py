"""Executable release-documentation and feature-ownership contracts for v4.0."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from logicytics.contracts import Specialty
from logicytics.module.discovery import preflight
from logicytics.module.modes import EXECUTION_MODES


class ReleaseDocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parent.parent
        cls.readme = (cls.root / "README.md").read_text(encoding="utf-8")
        cls.release = (cls.root / "docs" / "V4_RELEASE.md").read_text(encoding="utf-8")

    def test_release_identity_is_final_and_no_provisional_claim_remains(self) -> None:
        metadata = tomllib.loads((self.root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual("4.0.0", metadata["project"]["version"])
        self.assertIn("complete recreation", self.release.casefold())
        self.assertIn("verified run", self.readme.casefold())
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
            media_type for candidate in core if candidate.metadata is not None for media_type in candidate.metadata.output_media_types
        }
        for media_type in media_types:
            with self.subTest(media_type=media_type):
                self.assertIn(f"`{media_type}`", self.release)

    def test_release_evidence_is_recorded(self) -> None:
        self.assertIn("GitHub wiki", self.release)
        self.assertIn("eaf7628", self.release)

    def test_user_and_contributor_guides_cover_the_release_entry_points(self) -> None:
        documents = (
            "CONFIGURATION.md",
            "CONTRIBUTING.md",
            "OUTPUTS.md",
            "MODS.md",
            "MIGRATION.md",
            "FLOW_MATRIX.md",
            "V4_RELEASE.md",
            "SECURITY.md",
            "CODE_OF_CONDUCT.md",
        )
        for document in documents:
            with self.subTest(document=document):
                location = (
                    self.root / "docs" / document
                    if document.endswith(".md")
                    and document
                    not in {
                        "CONTRIBUTING.md",
                        "SECURITY.md",
                        "CODE_OF_CONDUCT.md",
                    }
                    else self.root / document
                )
                self.assertTrue(location.is_file())
        self.assertIn("Logicytics Wiki", self.readme)
        contributor = (self.root / "CONTRIBUTING.md").read_text(encoding="utf-8")
        for requirement in (
            "Development setup",
            "Architecture boundaries",
            "Core collector changes",
            "Plugins and MODs",
            "Configuration changes",
            "Testing expectations",
            "conventional commit",
            "Developer Certificate of Origin",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, contributor)

    def test_output_location_documentation_matches_run_owned_packaging(self) -> None:
        migration = (self.root / "docs" / "MIGRATION.md").read_text(encoding="utf-8")
        self.assertIn("manifest-backed run folder", self.readme)
        self.assertIn("logicytics.yaml", migration)
        self.assertIn("logicytics/module/", migration)


if __name__ == "__main__":
    unittest.main()
