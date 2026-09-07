"""Injectable, mockable boundaries for host-specific operations used by collectors."""

from __future__ import annotations

import ctypes
import importlib
import os
import shutil
import signal
import socket as _socket
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_ctypes_collector = importlib.import_module("logicytics.global.ctypes_collector")
query_registry_key_info = _ctypes_collector.query_registry_key_info
filetime = _ctypes_collector.filetime

try:
    import winreg as _winreg
except ImportError:  # pragma: no cover - exercised by non-Windows package imports
    _winreg = None


class ProcessAdapter:
    """Run one explicit, shell-free host command through a central policy seam."""

    maximum_capture_bytes = 64 * 1024 * 1024
    create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)

    @staticmethod
    def _command(
            command: Sequence[str | os.PathLike[str]],
    ) -> tuple[str, ...]:
        """Validate and normalize an explicit shell-free command sequence."""
        normalized = tuple(
            argument if isinstance(argument, str) else argument.__fspath__()
            for argument in command
        )

        if not normalized:
            raise ValueError("command must contain at least one argument")

        return normalized

    @staticmethod
    def run(
            command: Sequence[str | os.PathLike[str]],
            **options: Any,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        """Delegate to the guarded stdlib runner while retaining its familiar result contract."""
        normalized = ProcessAdapter._command(command)
        capture_directory = options.pop("capture_directory", None)

        if options.get("shell"):
            raise ValueError("collector process adapters never permit shell execution")

        timeout = options.get("timeout")

        if timeout is not None and (
                not isinstance(timeout, (int, float))
                or timeout <= 0
        ):
            raise ValueError("command timeout must be positive")

        capture_output = options.pop("capture_output", False)

        if not capture_output:
            return subprocess.run(normalized, **options)

        if "stdout" in options or "stderr" in options:
            raise ValueError(
                "capture_output cannot be combined with explicit streams"
            )

        wants_text = bool(
            options.pop("text", False)
            or options.get("encoding") is not None
        )
        encoding = options.pop("encoding", None) or "utf-8"
        errors = options.pop("errors", None) or "replace"

        if capture_directory is not None:
            capture_directory = Path(capture_directory)

            if not capture_directory.is_dir():
                raise ValueError(
                    "capture_directory must be an existing directory"
                )

        with (
            tempfile.TemporaryFile(dir=capture_directory) as stdout,
            tempfile.TemporaryFile(dir=capture_directory) as stderr,
        ):
            completed = subprocess.run(
                normalized,
                stdout=stdout,
                stderr=stderr,
                **options,
            )

            stdout_size = stdout.tell()
            stderr_size = stderr.tell()

            if (
                    stdout_size > ProcessAdapter.maximum_capture_bytes
                    or stderr_size > ProcessAdapter.maximum_capture_bytes
            ):
                raise ValueError(
                    f"command output exceeds the "
                    f"{ProcessAdapter.maximum_capture_bytes}-byte capture limit"
                )

            stdout.seek(0)
            stderr.seek(0)

            stdout_bytes = stdout.read()
            stderr_bytes = stderr.read()

        if wants_text:
            return subprocess.CompletedProcess[str](
                normalized,
                completed.returncode,
                stdout_bytes.decode(encoding, errors),
                stderr_bytes.decode(encoding, errors),
            )

        return subprocess.CompletedProcess[bytes](
            normalized,
            completed.returncode,
            stdout_bytes,
            stderr_bytes,
        )

    def popen(
            self,
            command: Sequence[str | os.PathLike[str]],
            **options: Any,
    ) -> subprocess.Popen[Any]:
        """Start one explicit long-lived process without invoking a command shell."""
        normalized = self._command(command)

        if options.get("shell"):
            raise ValueError("process adapters never permit shell execution")

        return subprocess.Popen(normalized, **options)

    @staticmethod
    def memory_bytes(process_id: int) -> int | None:
        """Return one process's resident memory through the host-specific boundary."""
        if (
                not isinstance(process_id, int)
                or isinstance(process_id, bool)
                or process_id <= 0
        ):
            raise ValueError("process_id must be a positive integer")

        if os.name == "nt":
            return windows_api_adapter.process_working_set(process_id)

        status_path = Path(f"/proc/{process_id}/status")

        try:
            for line in status_path.read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None

        return None

    @staticmethod
    def terminate_process_group(process_id: int) -> None:
        """Request termination of one non-Windows process group."""
        if (
                not isinstance(process_id, int)
                or isinstance(process_id, bool)
                or process_id <= 0
        ):
            raise ValueError("process_id must be a positive integer")

        os.killpg(process_id, signal.SIGTERM)


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
        """Return the optional Windows registry module or raise on unsupported hosts."""
        if _winreg is None:
            raise OSError("the Windows registry is unavailable on this platform")
        return _winreg

    def OpenKey(self, key: Any, sub_key: str):
        """Open one registry key through the platform boundary."""
        return self._module().OpenKey(key, sub_key)

    def QueryValueEx(self, key: Any, value_name: str):
        """Read one registry value through the platform boundary."""
        return self._module().QueryValueEx(key, value_name)

    def EnumKey(self, key: Any, index: int) -> str:
        """Return one child registry key by index."""
        return self._module().EnumKey(key, index)

    def EnumValue(self, key: Any, index: int):
        """Return one registry value tuple by index."""
        return self._module().EnumValue(key, index)

    @staticmethod
    def last_write_time(key: Any) -> str | None:
        """Return one key's Win32 last-write timestamp in UTC when available."""
        if _winreg is None:
            return None

        timestamp = filetime()
        result = query_registry_key_info(
            key.handle,
            timestamp,
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
        """Return the local host name."""
        return _socket.gethostname()

    @staticmethod
    def gethostbyname(hostname: str) -> str:
        """Resolve one host name to an IPv4 address."""
        return _socket.gethostbyname(hostname)

    @staticmethod
    def getaddrinfo(host: str, port: int | str | None):
        """Resolve socket addresses for a host and port."""
        return _socket.getaddrinfo(host, port)

    @staticmethod
    def inet_ntoa(packed_ip: bytes) -> str:
        """Convert packed IPv4 bytes into dotted notation."""
        return _socket.inet_ntoa(packed_ip)

    @staticmethod
    def socket(family: int = -1, type: int = -1, proto: int = -1, fileno: int | None = None):
        """Create a socket through the centrally mockable network boundary."""
        if fileno is None:
            return _socket.socket(family, type, proto)
        return _socket.socket(family, type, proto, fileno=fileno)


network_adapter = NetworkAdapter()


class WindowsApiAdapter:
    """Load explicitly named Win32 libraries behind a replaceable test seam."""

    @staticmethod
    def load_library(name: str):
        """Load one validated Win32 DLL name when running on Windows."""
        if (
                not isinstance(name, str)
                or not name
                or any(character in name for character in "/\\\x00")
        ):
            raise ValueError(
                "Win32 library name must be a simple non-empty name"
            )

        if sys.platform != "win32":
            raise OSError("Win32 libraries are unavailable on this platform")

        return ctypes.WinDLL(name, use_last_error=True)

    def is_administrator(self) -> bool | None:
        """Return administrator status, or None when the Win32 API is unavailable."""
        try:
            return bool(self.load_library("shell32").IsUserAnAdmin())
        except OSError:
            return None

    def process_working_set(self, process_id: int) -> int | None:
        """Return a Windows process working set without exposing raw Win32 handles."""

        class ProcessMemoryCounters(ctypes.Structure):
            """Minimal PROCESS_MEMORY_COUNTERS layout used by GetProcessMemoryInfo."""
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            kernel = self.load_library("kernel32")
            psapi = self.load_library("psapi")
        except OSError:
            return None
        handle = kernel.OpenProcess(0x0410, False, process_id)
        if not handle:
            return None
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.WorkingSetSize)
        finally:
            kernel.CloseHandle(handle)

    def process_descendants(self, parent_process_id: int) -> tuple[int, ...]:
        """Snapshot descendants of one Windows process through Toolhelp APIs."""

        class ProcessEntry(ctypes.Structure):
            """Minimal PROCESSENTRY32W layout used for descendant enumeration."""
            _fields_ = [
                ("dwSize", ctypes.c_ulong),
                ("cntUsage", ctypes.c_ulong),
                ("th32ProcessID", ctypes.c_ulong),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", ctypes.c_ulong),
                ("cntThreads", ctypes.c_ulong),
                ("th32ParentProcessID", ctypes.c_ulong),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.c_ulong),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        try:
            kernel = self.load_library("kernel32")
        except OSError:
            return ()
        snapshot = kernel.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot in (0, -1, ctypes.c_void_p(-1).value):
            return ()
        try:
            entry = ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            relationships: dict[int, list[int]] = {}
            valid = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while valid:
                relationships.setdefault(int(entry.th32ParentProcessID), []).append(
                    int(entry.th32ProcessID)
                )
                valid = kernel.Process32NextW(snapshot, ctypes.byref(entry))
            descendants: list[int] = []
            pending = [parent_process_id]
            while pending:
                children = relationships.get(pending.pop(), [])
                descendants.extend(children)
                pending.extend(children)
            return tuple(reversed(descendants))
        finally:
            kernel.CloseHandle(snapshot)

    def terminate_process(self, process_id: int, *, wait_milliseconds: int = 2000) -> bool:
        """Terminate one known Windows process ID and close its handle deterministically."""
        try:
            kernel = self.load_library("kernel32")
        except OSError:
            return False
        handle = kernel.OpenProcess(0x0001 | 0x100000, False, process_id)
        if not handle:
            return False
        try:
            terminated = bool(kernel.TerminateProcess(handle, 1))
            if terminated:
                kernel.WaitForSingleObject(handle, wait_milliseconds)
            return terminated
        finally:
            kernel.CloseHandle(handle)


windows_api_adapter = WindowsApiAdapter()


class FilesystemAdapter:
    """Centralize host discovery and evidence staging while leaving publication to ArtifactWriter."""

    @staticmethod
    def home() -> Path:
        """Return the current user's home directory."""
        return Path.home()

    @staticmethod
    def environment_path(name: str, fallback: Path) -> Path:
        """Read a validated environment variable as a path with a fallback."""
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ValueError("environment path name must be an alphanumeric variable name")
        return Path(os.environ.get(name, fallback))

    @staticmethod
    def system_drive_root() -> Path:
        """Return the current Windows system-drive root path."""
        return Path(os.environ.get("SystemDrive", "C:") + "\\")

    @staticmethod
    def children(path: Path):
        """Iterate immediate children of a directory."""
        return path.iterdir()

    @staticmethod
    def glob(path: Path, pattern: str):
        """Match direct children using a pathlib glob pattern."""
        return path.glob(pattern)

    @staticmethod
    def recursive(path: Path, pattern: str = "*"):
        """Match descendants using a recursive pathlib pattern."""
        return path.rglob(pattern)

    @staticmethod
    def walk(path: Path, **options: Any):
        """Walk a filesystem tree through the shared adapter."""
        return os.walk(path, **options)

    @staticmethod
    def scan(path: Path):
        """Open a directory iterator through the shared adapter."""
        return os.scandir(path)

    @staticmethod
    def copy_file(source: Path, destination: Path) -> None:
        """Copy file contents and metadata to a destination path."""
        shutil.copy2(source, destination)

    @staticmethod
    def disk_usage(path: Path):
        """Return filesystem usage statistics for a path."""
        return shutil.disk_usage(path)


filesystem_adapter = FilesystemAdapter()
