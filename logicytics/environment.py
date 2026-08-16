"""Read-only Windows environment checks used before v4 planning and execution."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import asdict, dataclass
from shutil import which


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Local privilege, UAC, and PowerShell-policy state."""

    is_administrator: bool | None
    uac_enabled: bool | None
    powershell_execution_policy: str | None

    def to_dict(self) -> dict[str, bool | str | None]:
        """Return a JSON-safe environment report."""
        return asdict(self)


def inspect_environment() -> EnvironmentReport:
    """Read local environment state without changing policy or elevation."""
    try:
        is_administrator = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        is_administrator = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System") as key:
            uac_enabled = bool(winreg.QueryValueEx(key, "EnableLUA")[0])
    except (ImportError, OSError):
        uac_enabled = None
    policy = None
    if which("powershell") is not None:
        try:
            completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-ExecutionPolicy"], capture_output=True, check=False, text=True, timeout=15)
            if completed.returncode == 0:
                policy = completed.stdout.strip() or None
        except OSError:
            pass
    return EnvironmentReport(is_administrator, uac_enabled, policy)
