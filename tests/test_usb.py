"""Tests for USB-mode Windows discovery and storage isolation."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from logicytics.cli import cli_methods
from logicytics.module.errors import PlanError
from logicytics.module.usb import UsbContext, ensure_usb_storage, find_windows_installation


class UsbTests(unittest.TestCase):
    """Drive discovery remains deterministic and never permits Windows-disk writes."""

    def test_usb_auto_discovers_the_first_windows_drive_in_alphabetical_order(self) -> None:
        """Automatic selection checks drive letters from A through Z and stops at the first Windows directory."""
        checked: list[str] = []

        def windows_directory(path: Path) -> bool:
            checked.append(str(path))
            return str(path).casefold().endswith("c:\\windows")

        with patch.object(Path, "is_dir", autospec=True, side_effect=windows_directory):
            context = find_windows_installation()

        self.assertEqual("C", context.windows_drive)
        self.assertEqual(("A:", "B:", "C:"), context.scanned_drives)
        self.assertEqual(3, len(checked))

    def test_usb_explicit_drive_skips_auto_selection_and_validates_windows(self) -> None:
        """An explicit drive supports systems with more than one Windows installation."""
        with patch.object(Path, "is_dir", autospec=True, return_value=True) as is_directory:
            context = find_windows_installation("f:")

        self.assertEqual("F", context.windows_drive)
        self.assertEqual(("F:",), context.scanned_drives)
        self.assertEqual(1, is_directory.call_count)

    def test_usb_storage_rejects_the_discovered_windows_drive(self) -> None:
        """Output, cache, and temporary roots cannot be placed on the selected Windows disk."""
        context = UsbContext("C", ("C:",))
        with self.assertRaisesRegex(PlanError, "selected Windows disk C"):
            ensure_usb_storage(context, output_root=Path("C:/output"), cache_directory=Path("E:/.cache"))

        ensure_usb_storage(context, output_root=Path("E:/output"), cache_directory=Path("E:/.cache"))

    def test_usb_is_available_globally_and_after_supported_subcommands(self) -> None:
        """USB mode is available for planning and collection command forms."""
        parser = cli_methods.parser()
        self.assertEqual("D", parser.parse_args(["--usb=D", "plan"]).usb)
        self.assertEqual("E", parser.parse_args(["run", "--usb", "E"]).usb)


if __name__ == "__main__":
    unittest.main()
