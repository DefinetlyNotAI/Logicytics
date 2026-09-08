"""Centralized, typed Windows ctypes bindings used by Logicytics."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Final, TypeAlias

if sys.platform != "win32":
    raise RuntimeError("ctypes_collector is only supported on Windows")

# ctypes aliases

DWORD: TypeAlias = wintypes.DWORD
HANDLE: TypeAlias = wintypes.HANDLE
BOOL: TypeAlias = wintypes.BOOL
ULONG: TypeAlias = ctypes.c_ulong
SIZE_T: TypeAlias = ctypes.c_size_t
ULARGE_INTEGER: TypeAlias = ctypes.c_ulonglong
FILETIME: TypeAlias = wintypes.FILETIME
LPVOID: TypeAlias = wintypes.LPVOID
WORD: TypeAlias = wintypes.WORD

LPDWORD = ctypes.POINTER(DWORD)
LPULARGE_INTEGER = ctypes.POINTER(ULARGE_INTEGER)

# Windows constants

PROCESS_QUERY_LIMITED_INFORMATION: Final = 0x1000
STILL_ACTIVE: Final = 259


# Windows structures


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


class MemoryBasicInformation(ctypes.Structure):
    """Windows MEMORY_BASIC_INFORMATION structure."""

    _fields_ = [
        ("BaseAddress", LPVOID),
        ("AllocationBase", LPVOID),
        ("AllocationProtect", DWORD),
        ("PartitionId", WORD),
        ("RegionSize", SIZE_T),
        ("State", DWORD),
        ("Protect", DWORD),
        ("Type", DWORD),
    ]


class MemoryStatus(ctypes.Structure):
    """Windows MEMORYSTATUSEX structure."""

    _fields_ = [
        ("dwLength", ULONG),
        ("dwMemoryLoad", ULONG),
        ("ullTotalPhys", ULARGE_INTEGER),
        ("ullAvailPhys", ULARGE_INTEGER),
        ("ullTotalPageFile", ULARGE_INTEGER),
        ("ullAvailPageFile", ULARGE_INTEGER),
        ("ullTotalVirtual", ULARGE_INTEGER),
        ("ullAvailVirtual", ULARGE_INTEGER),
        ("ullAvailExtendedVirtual", ULARGE_INTEGER),
    ]


# DLL handles

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

# kernel32 bindings

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

_GetCurrentProcess = kernel32.GetCurrentProcess
_GetCurrentProcess.argtypes = []
_GetCurrentProcess.restype = HANDLE

_GetLogicalDrives = kernel32.GetLogicalDrives
_GetLogicalDrives.argtypes = []
_GetLogicalDrives.restype = DWORD

_GetDriveTypeW = kernel32.GetDriveTypeW
_GetDriveTypeW.argtypes = [
    wintypes.LPCWSTR,
]
_GetDriveTypeW.restype = wintypes.UINT

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

_GlobalMemoryStatusEx = kernel32.GlobalMemoryStatusEx
_GlobalMemoryStatusEx.argtypes = [
    ctypes.POINTER(MemoryStatus),
]
_GlobalMemoryStatusEx.restype = BOOL

_VirtualQuery = kernel32.VirtualQuery
_VirtualQuery.argtypes = [
    LPVOID,
    ctypes.POINTER(MemoryBasicInformation),
    SIZE_T,
]
_VirtualQuery.restype = SIZE_T

# advapi32 bindings

_RegQueryInfoKeyW = advapi32.RegQueryInfoKeyW
_RegQueryInfoKeyW.argtypes = [
    wintypes.HKEY,
    wintypes.LPWSTR,
    LPDWORD,
    LPDWORD,
    LPDWORD,
    LPDWORD,
    LPDWORD,
    LPDWORD,
    LPDWORD,
    LPDWORD,
    LPDWORD,
    ctypes.POINTER(FILETIME),
]
_RegQueryInfoKeyW.restype = wintypes.LONG

# psapi bindings

_GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
_GetProcessMemoryInfo.argtypes = [
    HANDLE,
    ctypes.POINTER(ProcessMemoryCounters),
    DWORD,
]
_GetProcessMemoryInfo.restype = BOOL

_GetMappedFileNameW = psapi.GetMappedFileNameW
_GetMappedFileNameW.argtypes = [
    HANDLE,
    LPVOID,
    wintypes.LPWSTR,
    DWORD,
]
_GetMappedFileNameW.restype = DWORD


# Process wrappers


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


def get_current_process() -> HANDLE:
    """Return the pseudo-handle for the current process."""

    return _GetCurrentProcess()


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


def get_process_memory_info(
    process: HANDLE,
    counters: ProcessMemoryCounters,
) -> bool:
    """Populate memory counters for a process."""

    counters.cb = ctypes.sizeof(counters)

    return bool(
        _GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            ctypes.sizeof(counters),
        )
    )


# Virtual-memory wrappers


def virtual_query(
    address: int,
    memory: MemoryBasicInformation,
) -> int:
    """Query the virtual-memory region containing an address."""

    return int(
        _VirtualQuery(
            ctypes.c_void_p(address),
            ctypes.byref(memory),
            ctypes.sizeof(memory),
        )
    )


def get_mapped_file_name(
    process: HANDLE,
    address: int,
    buffer_size: int = 32_768,
) -> str | None:
    """Return the mapped filename associated with a process address."""

    buffer = ctypes.create_unicode_buffer(buffer_size)

    length = _GetMappedFileNameW(
        process,
        ctypes.c_void_p(address),
        buffer,
        buffer_size,
    )

    if not length:
        return None

    return buffer.value[: int(length)]


def pointer_value(pointer: LPVOID) -> int | None:
    """Return the integer value represented by a Windows pointer."""

    return ctypes.cast(
        pointer,
        ctypes.c_void_p,
    ).value


# Drive and volume wrappers


def get_logical_drives() -> int:
    """Return the bitmask identifying available Windows logical drives."""

    mask = _GetLogicalDrives()

    if mask == 0:
        raise ctypes.WinError(ctypes.get_last_error())

    return int(mask)


def get_drive_type(root: str) -> int:
    """Return the Windows drive type code for a filesystem root."""

    return int(_GetDriveTypeW(root))


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


# Memory wrappers


def global_memory_status() -> MemoryStatus:
    """Return current Windows physical and virtual memory statistics."""

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)

    if not _GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())

    return status


# Registry wrappers


def query_registry_key_info(
    key_handle: int,
    timestamp: FILETIME,
) -> int:
    """Query registry key metadata and populate its last-write timestamp."""

    return int(
        _RegQueryInfoKeyW(
            wintypes.HKEY(key_handle),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ctypes.byref(timestamp),
        )
    )


# ctypes factories


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


def filetime() -> FILETIME:
    """Create an empty Windows FILETIME structure."""

    return FILETIME()


__all__ = [
    # Types
    "BOOL",
    "DWORD",
    "FILETIME",
    "HANDLE",
    "LPDWORD",
    "LPULARGE_INTEGER",
    "LPVOID",
    "MemoryBasicInformation",
    "MemoryStatus",
    "ProcessMemoryCounters",
    "SIZE_T",
    "ULARGE_INTEGER",
    "ULONG",
    "WORD",
    # Constants
    "PROCESS_QUERY_LIMITED_INFORMATION",
    "STILL_ACTIVE",
    # Process API
    "close_handle",
    "get_current_process",
    "get_exit_code_process",
    "get_process_memory_info",
    "open_process",
    # Virtual memory API
    "get_mapped_file_name",
    "pointer_value",
    "virtual_query",
    # Drive and volume API
    "get_disk_free_space",
    "get_drive_type",
    "get_logical_drives",
    "get_volume_information",
    # Memory API
    "global_memory_status",
    # Registry API
    "query_registry_key_info",
    # ctypes factories
    "create_unicode_buffer",
    "dword",
    "filetime",
    "ularge_integer",
    "ulong",
]
