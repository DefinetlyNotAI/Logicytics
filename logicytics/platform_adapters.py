"""Injectable, mockable boundaries for host-specific operations used by collectors."""

from __future__ import annotations

import subprocess
import socket as _socket
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
