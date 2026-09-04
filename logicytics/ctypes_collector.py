"""Centralized, typed Windows ctypes bindings used by Logicytics."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Final, TypeAlias

if sys.platform != "win32":
    raise RuntimeError("ctypes_collector is only supported on Windows")

DWORD: TypeAlias = wintypes.DWORD
HANDLE: TypeAlias = wintypes.HANDLE
BOOL: TypeAlias = wintypes.BOOL
ULONG: TypeAlias = ctypes.c_ulong
SIZE_T: TypeAlias = ctypes.c_size_t
ULARGE_INTEGER: TypeAlias = ctypes.c_ulonglong

LPDWORD: TypeAlias = ctypes.POINTER(DWORD)
LPULARGE_INTEGER: TypeAlias = ctypes.POINTER(ULARGE_INTEGER)

PROCESS_QUERY_LIMITED_INFORMATION: Final = 0x1000
STILL_ACTIVE: Final = 259


class ProcessMemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS structure."""

    _fields_ = [
        ("cb", DWORD),
        ("PageFaultCount", DWORD),
        ("PeakWorkingSetSize", SIZE_T),
        ("WorkingSetSize", SIZE_T),
        ("QuotaPeakPagedPoolUsage", SIZE_T),
        ("QuotaPagedPoolUsage", SIZE_T),
        ("QuotaPeakNonPagedPoolUsage", SIZE_T),
        ("QuotaNonPagedPoolUsage", SIZE_T),
        ("PagefileUsage", SIZE_T),
        ("PeakPagefileUsage", SIZE_T),
    ]


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_OpenProcess = kernel32.OpenProcess
_OpenProcess.argtypes = [
    DWORD,
    BOOL,
    DWORD,
]
_OpenProcess.restype = HANDLE

_GetExitCodeProcess = kernel32.GetExitCodeProcess
_GetExitCodeProcess.argtypes = [
    HANDLE,
    LPDWORD,
]
_GetExitCodeProcess.restype = BOOL

_CloseHandle = kernel32.CloseHandle
_CloseHandle.argtypes = [
    HANDLE,
]
_CloseHandle.restype = BOOL

_GetVolumeInformationW = kernel32.GetVolumeInformationW
_GetVolumeInformationW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    DWORD,
    LPDWORD,
    LPDWORD,
    LPDWORD,
    wintypes.LPWSTR,
    DWORD,
]
_GetVolumeInformationW.restype = BOOL

_GetDiskFreeSpaceExW = kernel32.GetDiskFreeSpaceExW
_GetDiskFreeSpaceExW.argtypes = [
    wintypes.LPCWSTR,
    LPULARGE_INTEGER,
    LPULARGE_INTEGER,
    LPULARGE_INTEGER,
]
_GetDiskFreeSpaceExW.restype = BOOL


def open_process(
        process_id: int,
        *,
        access: int = PROCESS_QUERY_LIMITED_INFORMATION,
        inherit_handle: bool = False,
) -> HANDLE | None:
    """Open a Windows process and return its handle, or None on failure."""

    handle = _OpenProcess(
        access,
        inherit_handle,
        process_id,
    )

    if not handle:
        return None

    return handle


def get_exit_code_process(handle: HANDLE) -> int | None:
    """Return a process exit code, or None if querying it failed."""

    exit_code = DWORD()

    if not _GetExitCodeProcess(
            handle,
            ctypes.byref(exit_code),
    ):
        return None

    return int(exit_code.value)


def close_handle(handle: HANDLE) -> bool:
    """Close a Windows kernel handle."""

    return bool(_CloseHandle(handle))


def get_volume_information(
        root: str,
        label: ctypes.Array[ctypes.c_wchar],
        serial: DWORD,
        maximum_component_length: DWORD,
        flags: DWORD,
        filesystem: ctypes.Array[ctypes.c_wchar],
) -> bool:
    """Populate volume metadata for a filesystem root."""

    return bool(
        _GetVolumeInformationW(
            root,
            label,
            len(label),
            ctypes.byref(serial),
            ctypes.byref(maximum_component_length),
            ctypes.byref(flags),
            filesystem,
            len(filesystem),
        )
    )


def get_disk_free_space(
        root: str,
        available: ULARGE_INTEGER,
        total: ULARGE_INTEGER,
        free: ULARGE_INTEGER,
) -> bool:
    """Populate disk space counters for a filesystem root."""

    return bool(
        _GetDiskFreeSpaceExW(
            root,
            ctypes.byref(available),
            ctypes.byref(total),
            ctypes.byref(free),
        )
    )


def create_unicode_buffer(size: int) -> ctypes.Array[ctypes.c_wchar]:
    """Create a mutable Windows Unicode buffer."""

    return ctypes.create_unicode_buffer(size)


def dword(value: int = 0) -> DWORD:
    """Create a Windows DWORD value."""

    return DWORD(value)


def ulong(value: int = 0) -> ULONG:
    """Create an unsigned long ctypes value."""

    return ULONG(value)


def ularge_integer(value: int = 0) -> ULARGE_INTEGER:
    """Create an unsigned 64-bit Windows integer."""

    return ULARGE_INTEGER(value)


__all__ = [
    "BOOL",
    "DWORD",
    "HANDLE",
    "LPDWORD",
    "LPULARGE_INTEGER",
    "PROCESS_QUERY_LIMITED_INFORMATION",
    "ProcessMemoryCounters",
    "SIZE_T",
    "STILL_ACTIVE",
    "ULARGE_INTEGER",
    "ULONG",
    "close_handle",
    "create_unicode_buffer",
    "dword",
    "get_disk_free_space",
    "get_exit_code_process",
    "get_volume_information",
    "open_process",
    "ularge_integer",
    "ulong",
]
