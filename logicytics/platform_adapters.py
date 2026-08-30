"""Injectable, mockable boundaries for host-specific operations used by collectors."""

from __future__ import annotations

import subprocess
import socket as _socket
import ctypes
import shutil
import os
from pathlib import Path
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    import winreg as _winreg
except ImportError:  # pragma: no cover - exercised by non-Windows package imports
    _winreg = None


class ProcessAdapter:
    """Run one explicit, shell-free host command through a central policy seam."""

    def run(self, command: Sequence[str], **options: Any) -> subprocess.CompletedProcess[str]:
        """Delegate to the guarded stdlib runner while retaining its familiar result contract."""
        normalized = tuple(str(argument) for argument in command)
        if not normalized:
            raise ValueError("command must contain at least one argument")
        if options.get("shell") is True:
            raise ValueError("collector process adapters never permit shell execution")
        timeout = options.get("timeout")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise ValueError("command timeout must be positive")
        return subprocess.run(normalized, **options)


process_adapter = ProcessAdapter()


def which(command: str) -> str | None:
    """Resolve one host executable through the shared platform boundary."""
    if not isinstance(command, str) or not command.strip() or any(value in command for value in "\r\n\x00"):
        raise ValueError("executable name must be a non-empty single-line string")
    return shutil.which(command)


class RegistryAdapter:
    """Expose bounded read-only registry primitives behind one mockable boundary."""

    HKEYType = _winreg.HKEYType if _winreg is not None else object
    HKEY_CURRENT_USER = _winreg.HKEY_CURRENT_USER if _winreg is not None else 0
    HKEY_LOCAL_MACHINE = _winreg.HKEY_LOCAL_MACHINE if _winreg is not None else 0

    @staticmethod
    def _module():
        if _winreg is None:
            raise OSError("the Windows registry is unavailable on this platform")
        return _winreg

    def OpenKey(self, key: Any, sub_key: str):
        return self._module().OpenKey(key, sub_key)

    def QueryValueEx(self, key: Any, value_name: str):
        return self._module().QueryValueEx(key, value_name)

    def EnumKey(self, key: Any, index: int) -> str:
        return self._module().EnumKey(key, index)

    def EnumValue(self, key: Any, index: int):
        return self._module().EnumValue(key, index)

    def last_write_time(self, key: Any) -> str | None:
        """Return one key's Win32 last-write timestamp in UTC when available."""
        if _winreg is None:
            return None
        import ctypes
        from ctypes import wintypes

        timestamp = wintypes.FILETIME()
        result = ctypes.windll.advapi32.RegQueryInfoKeyW(
            wintypes.HKEY(key.handle), None, None, None, None, None, None,
            None, None, None, None, ctypes.byref(timestamp),
        )
        if result != 0:
            return None
        ticks = (timestamp.dwHighDateTime << 32) | timestamp.dwLowDateTime
        return (datetime(1601, 1, 1, tzinfo=timezone.utc)
                + timedelta(microseconds=ticks // 10)).isoformat()


registry_adapter = RegistryAdapter()


class NetworkAdapter:
    """Centralize local host resolution and explicitly authorized socket creation."""

    AF_INET = _socket.AF_INET
    SOCK_RAW = _socket.SOCK_RAW
    IPPROTO_IP = _socket.IPPROTO_IP
    IP_HDRINCL = _socket.IP_HDRINCL
    SIO_RCVALL = getattr(_socket, "SIO_RCVALL", 0)
    RCVALL_ON = getattr(_socket, "RCVALL_ON", 1)
    RCVALL_OFF = getattr(_socket, "RCVALL_OFF", 0)
    gaierror = _socket.gaierror

    @staticmethod
    def gethostname() -> str:
        return _socket.gethostname()

    @staticmethod
    def gethostbyname(hostname: str) -> str:
        return _socket.gethostbyname(hostname)

    @staticmethod
    def getaddrinfo(host: str, port: int | str | None):
        return _socket.getaddrinfo(host, port)

    @staticmethod
    def inet_ntoa(packed_ip: bytes) -> str:
        return _socket.inet_ntoa(packed_ip)

    @staticmethod
    def socket(family: int = -1, type: int = -1, proto: int = -1, fileno: int | None = None):
        if fileno is None:
            return _socket.socket(family, type, proto)
        return _socket.socket(family, type, proto, fileno=fileno)


network_adapter = NetworkAdapter()


class WindowsApiAdapter:
    """Load explicitly named Win32 libraries behind a replaceable test seam."""

    @staticmethod
    def load_library(name: str):
        if not isinstance(name, str) or not name or any(character in name for character in "/\\\x00"):
            raise ValueError("Win32 library name must be a simple non-empty name")
        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            raise OSError("Win32 libraries are unavailable on this platform")
        return loader(name, use_last_error=True)


windows_api_adapter = WindowsApiAdapter()


class FilesystemAdapter:
    """Centralize host discovery and evidence staging while leaving publication to ArtifactWriter."""

    @staticmethod
    def home() -> Path:
        return Path.home()

    @staticmethod
    def environment_path(name: str, fallback: Path) -> Path:
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ValueError("environment path name must be an alphanumeric variable name")
        return Path(os.environ.get(name, fallback))

    @staticmethod
    def system_drive_root() -> Path:
        return Path(os.environ.get("SystemDrive", "C:") + "\\")

    @staticmethod
    def children(path: Path):
        return path.iterdir()

    @staticmethod
    def glob(path: Path, pattern: str):
        return path.glob(pattern)

    @staticmethod
    def recursive(path: Path, pattern: str = "*"):
        return path.rglob(pattern)

    @staticmethod
    def walk(path: Path, **options: Any):
        return os.walk(path, **options)

    @staticmethod
    def scan(path: Path):
        return os.scandir(path)

    @staticmethod
    def copy_file(source: Path, destination: Path) -> None:
        shutil.copy2(source, destination)

    @staticmethod
    def disk_usage(path: Path):
        return shutil.disk_usage(path)


filesystem_adapter = FilesystemAdapter()
