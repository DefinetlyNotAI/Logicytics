"""Read-only Windows environment checks used before v4 planning and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from logicytics.platform_adapters import (
    process_adapter,
    registry_adapter,
    which,
    windows_api_adapter,
)


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
    is_administrator = windows_api_adapter.is_administrator()
    try:
        with registry_adapter.OpenKey(
                registry_adapter.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System",
        ) as key:
            uac_enabled = bool(registry_adapter.QueryValueEx(key, "EnableLUA")[0])
    except OSError:
        uac_enabled = None
    policy = None
    powershell = which("powershell")
    if powershell is not None:
        try:
            completed = process_adapter.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command", "Get-ExecutionPolicy"],
                capture_output=True,
                check=False,
                text=True,
                timeout=15,
            )
            if completed.returncode == 0:
                policy = completed.stdout.strip() or None
        except OSError:
            pass
    return EnvironmentReport(is_administrator, uac_enabled, policy)
