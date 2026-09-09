"""USB execution safeguards for locating an offline Windows installation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase

from logicytics.module.errors import PlanError


@dataclass(frozen=True, slots=True)
class UsbContext:
    """The Windows installation selected for a USB-hosted execution."""

    windows_drive: str
    scanned_drives: tuple[str, ...]

    @property
    def windows_root(self) -> Path:
        """Return the selected offline Windows directory."""
        return Path(f"{self.windows_drive}:\\Windows")


def _normalize_drive(value: str) -> str:
    """Accept exactly one Windows drive letter, with an optional colon."""
    normalized = value.strip().upper().removesuffix(":")
    if len(normalized) != 1 or normalized not in ascii_uppercase:
        raise PlanError("--usb expects one drive letter from A to Z")
    return normalized


def find_windows_installation(requested_drive: str | None = None) -> UsbContext:
    """Locate Windows from A through Z, or validate one explicitly selected drive."""
    drives = (_normalize_drive(requested_drive),) if requested_drive else tuple(ascii_uppercase)
    scanned: list[str] = []
    for drive in drives:
        scanned.append(f"{drive}:")
        if (Path(f"{drive}:\\") / "Windows").is_dir():
            return UsbContext(drive, tuple(scanned))
    if requested_drive is not None:
        raise PlanError(f"--usb {drives[0]} did not contain a Windows directory")
    raise PlanError("--usb could not find a Windows installation while scanning drives A through Z")


def ensure_usb_storage(context: UsbContext, **paths: Path) -> None:
    """Reject every write root that would place USB-mode data on the Windows disk."""
    windows_drive = context.windows_drive.casefold()
    unsafe = sorted(
        name
        for name, path in paths.items()
        if Path(path).drive.rstrip(":").casefold() == windows_drive
    )
    if unsafe:
        raise PlanError(
            "--usb cannot write to the selected Windows disk "
            f"{context.windows_drive}: ({', '.join(unsafe)})"
        )
