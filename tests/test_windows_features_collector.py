"""Permission-handling checks for the optional Windows-features collector."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parent.parent / "core" / "hardware" / "windows_features.py"
    spec = importlib.util.spec_from_file_location("windows_features_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WindowsFeaturesCollectorTests(unittest.TestCase):
    """Ensure expected PowerShell permission restrictions do not fail a whole run."""

    def test_power_shell_dism_access_error_is_recognized(self) -> None:
        """DISM's 'access to ... is denied' wording must map to a skipped collector."""
        module = _module()
        self.assertTrue(module._is_access_denied("Set current directory failed: Access to the path is denied."))


if __name__ == "__main__":
    unittest.main()
