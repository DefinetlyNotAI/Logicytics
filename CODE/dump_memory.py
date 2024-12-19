import ctypes  # TODO - Remove this dependency, change to pywin32 or find a alternative
import datetime
import os
import platform

import psutil

from logicytics import Log, DEBUG

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})
    # Constants
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    MEM_COMMIT = 0x1000
    PAGE_READWRITE = 0x04


# Function to save RAM content snapshot to a file
@log.function
def dump_ram_content():
    """
    Capture the current state of the system's RAM and write it to a file.

    This function gathers memory statistics, system-specific details, and writes
    the information to a file named 'Ram_Snapshot.txt'.
    """
    try:
        # Generate a timestamp for the file
        dump_file = "Ram_Snapshot.txt"

        # Gather memory statistics using psutil
        memory_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()

        # Get system-specific details
        system_info = (
            "System Information:\n"
            "===================================\n"
            f"OS: {platform.system()} {platform.release()}\n"
            f"Architecture: {platform.architecture()[0]}\n"
            f"Processor: {platform.processor()}\n"
            f"Machine: {platform.machine()}\n\n"
        )

        # Prepare content to dump
        dump_content = (
            f"RAM Snapshot - {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n"
            "===================================\n"
            f"{system_info}"
            f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB\n"
            f"Available Memory: {memory_info.available / (1024 ** 3):.2f} GB\n"
            f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB\n"
            f"Memory Usage: {memory_info.percent}%\n\n"
            f"Swap Total: {swap_info.total / (1024 ** 3):.2f} GB\n"
            f"Swap Used: {swap_info.used / (1024 ** 3):.2f} GB\n"
            f"Swap Free: {swap_info.free / (1024 ** 3):.2f} GB\n"
            f"Swap Usage: {swap_info.percent}%\n"
        )

        # Write the content to the file
        with open(dump_file, "w", encoding="utf-8") as file:
            file.write(dump_content)

        log.info(f"RAM snapshot saved to: {dump_file}")

    except Exception as e:
        log.error(f"Error capturing RAM snapshot: {e}")


# Define structures for SystemInfo
class SystemInfo(ctypes.Structure):
    # noinspection PyUnresolvedReferences
    """
        A ctypes Structure to hold system information.

        Attributes:
            wProcessorArchitecture (ctypes.c_ushort): Processor architecture.
            wReserved (ctypes.c_ushort): Reserved.
            dwPageSize (ctypes.c_ulong): Page size.
            lpMinimumApplicationAddress (ctypes.c_void_p): Minimum application address.
            lpMaximumApplicationAddress (ctypes.c_void_p): Maximum application address.
            dwActiveProcessorMask (ctypes.POINTER(ctypes.c_ulong)): Active processor mask.
            dwNumberOfProcessors (ctypes.c_ulong): Number of processors.
            dwProcessorType (ctypes.c_ulong): Processor type.
            dwAllocationGranularity (ctypes.c_ulong): Allocation granularity.
            wProcessorLevel (ctypes.c_ushort): Processor level.
            wProcessorRevision (ctypes.c_ushort): Processor revision.
    """
    _fields_ = [
        ("wProcessorArchitecture", ctypes.c_ushort),
        ("wReserved", ctypes.c_ushort),
        ("dwPageSize", ctypes.c_ulong),
        ("lpMinimumApplicationAddress", ctypes.c_void_p),
        ("lpMaximumApplicationAddress", ctypes.c_void_p),
        ("dwActiveProcessorMask", ctypes.POINTER(ctypes.c_ulong)),
        ("dwNumberOfProcessors", ctypes.c_ulong),
        ("dwProcessorType", ctypes.c_ulong),
        ("dwAllocationGranularity", ctypes.c_ulong),
        ("wProcessorLevel", ctypes.c_ushort),
        ("wProcessorRevision", ctypes.c_ushort),
    ]


# Define BasicMemInfo
class BasicMemInfo(ctypes.Structure):
    # noinspection PyUnresolvedReferences
    """
        A ctypes Structure to hold basic memory information.

        Attributes:
            BaseAddress (ctypes.c_void_p): Base address.
            AllocationBase (ctypes.c_void_p): Allocation base.
            AllocationProtect (ctypes.c_ulong): Allocation protection.
            RegionSize (ctypes.c_size_t): Region size.
            State (ctypes.c_ulong): State.
            Protect (ctypes.c_ulong): Protection.
            Type (ctypes.c_ulong): Type.
    """
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", ctypes.c_ulong),
        ("RegionSize", ctypes.c_size_t),
        ("State", ctypes.c_ulong),
        ("Protect", ctypes.c_ulong),
        ("Type", ctypes.c_ulong),
    ]


def get_system_info() -> SystemInfo:
    """
    Retrieve and return system information using the `GetSystemInfo` function from the Windows API.

    Returns:
        SystemInfo: A `SystemInfo` structure containing details about the system's architecture,
                    processor, memory, and other attributes.
    """
    system_info = SystemInfo()
    ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(system_info))
    return system_info


@log.function
def read_memory():
    """
    Read the memory of the current process and write the content to a file.

    This function opens the current process with the necessary permissions,
    retrieves system information, and iterates through memory pages to read
    """
    # Open current process with permissions
    process = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, os.getpid())
    if not process:
        log.error("Unable to open process for reading.")
        return

    # Get system info
    log.info("Getting SystemRAM Info...")
    system_info = get_system_info()
    min_address = system_info.lpMinimumApplicationAddress
    max_address = system_info.lpMaximumApplicationAddress
    try:
        with open("SystemRam_Info.txt", "w") as sys_file:
            sys_file.write("System Information:\n")
            sys_file.write("===================================\n")
            sys_file.write(f"Minimum Address: {min_address}\n")
            sys_file.write(f"Maximum Address: {max_address}\n")
            sys_file.write(f"Allocation Granularity: {system_info.dwAllocationGranularity}\n")
            sys_file.write(f"Processor Architecture: {system_info.wProcessorArchitecture}\n")
            sys_file.write(f"Number of Processors: {system_info.dwNumberOfProcessors}\n")
            sys_file.write(f"Processor Type: {system_info.dwProcessorType}\n")
            sys_file.write(f"Processor Level: {system_info.wProcessorLevel}\n")
            sys_file.write(f"Processor Revision: {system_info.wProcessorRevision}\n")
            sys_file.write(f"Page Size: {system_info.dwPageSize}\n")
            sys_file.write(f"Reserved: {system_info.wReserved}\n")
            sys_file.write("===================================\n")
            sys_file.write(f"Raw SystemInfo: {system_info}\n")
            sys_file.write("===================================\n")
    except Exception as e:
        log.error(f"Error getting RAM info: {e}")
    log.debug(f"Memory Range: {min_address:#x} - {max_address:#x}")

    # Iterate through memory pages
    memory_info = BasicMemInfo()
    address = min_address
    with open("Ram_Dump.txt", "w") as dump_file:
        while address < max_address:
            result = ctypes.windll.kernel32.VirtualQueryEx(
                process, ctypes.c_void_p(address), ctypes.byref(memory_info), ctypes.sizeof(memory_info)
            )
            if not result:
                break

            # FIXME - mem issues
            try:
                # Check if the memory is committed and readable
                buffer = ctypes.create_string_buffer(memory_info.RegionSize)
                bytes_read = ctypes.c_size_t()
                ctypes.windll.kernel32.ReadProcessMemory(
                    process,
                    ctypes.c_void_p(memory_info.BaseAddress),
                    buffer,
                    memory_info.RegionSize,
                    ctypes.byref(bytes_read),
                )
                dump_file.write(str(buffer.raw[: bytes_read.value]))
                address += memory_info.RegionSize
            except MemoryError as ME:
                log.error(f"Memory is not enough and is not available to be read (RAM): {ME}")
                os.remove("Ram_Dump.txt")

    # Close the process handle
    ctypes.windll.kernel32.CloseHandle(process)
    log.info("Memory dump complete. Saved to 'ram_dump.txt'.")
    log.warning("Encoding is in HEX")


if __name__ == "__main__":
    log.info("Starting memory dump process...")
    dump_ram_content()
    read_memory()
