import os
import platform
import struct
from datetime import datetime

import psutil

from logicytics import log, config

# Constants from config with validation
LIMIT_FILE_SIZE = config.getint("DumpMemory Settings", "file_size_limit")  # MiB
SAFETY_MARGIN = config.getfloat("DumpMemory Settings", "file_size_safety")  # MiB
DUMP_DIR = config.get("DumpMemory Settings", "dump_directory", fallback="memory_dumps")

if SAFETY_MARGIN < 1:
    log.critical("Invalid Safety Margin Inputted - Cannot proceed with dump memory")
    exit(1)


def capture_ram_snapshot():
    """
    Captures and logs the current system memory statistics to a file.
    
    Retrieves detailed information about RAM and swap memory usage using psutil.
    Writes memory statistics in gigabytes to 'Ram_Snapshot.txt', including:
    - Total RAM
    - Used RAM
    - Available RAM
    - Total Swap memory
    - Used Swap memory
    - Free Swap memory
    - Percentage of RAM used
    
    Logs the process and handles potential file writing errors.
    
    Raises:
        IOError: If unable to write to the output file
        Exception: For any unexpected errors during memory snapshot capture
    """

    def memory_helper(mem_var, flavor_text: str, use_free_rather_than_available: bool = False):
        file.write(f"Total {flavor_text}: {mem_var.total / (1024 ** 3):.2f} GB\n")
        file.write(f"Used {flavor_text}: {mem_var.used / (1024 ** 3):.2f} GB\n")
        if use_free_rather_than_available:
            file.write(f"Available {flavor_text}: {mem_var.free / (1024 ** 3):.2f} GB\n")
        else:
            file.write(f"Available {flavor_text}: {mem_var.available / (1024 ** 3):.2f} GB\n")
        file.write(f"{flavor_text} Percent Usage: {mem_var.percent:.2f}%\n")

    log.info("Capturing RAM Snapshot...")
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        with open(os.path.join(DUMP_DIR, "Ram_Snapshot.txt"), "w", encoding="utf-8") as file:
            memory_helper(memory, "RAM")
            memory_helper(swap, "Swap Memory", use_free_rather_than_available=True)
    except Exception as e:
        log.error(f"Failed to capture RAM snapshot: {e}")
    log.info("RAM Snapshot saved to Ram_Snapshot.txt")


def gather_system_info():
    """
    Gathers detailed system information and saves it to a file.
    """
    log.info("Gathering system information...")
    try:
        sys_info = {
            'Architecture': platform.architecture(),
            'System': platform.system(),
            'Machine': platform.machine(),
            'Processor': platform.processor(),
            'Page Size (bytes)': struct.calcsize("P"),
            'CPU Count': psutil.cpu_count(),
            'CPU Frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unavailable',
            'Boot Time': datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception as e:
        log.error(f"Error gathering system information: {e}")
        sys_info = {'Error': 'Failed to gather system information'}
    try:
        with open(os.path.join(DUMP_DIR, "SystemRam_Info.txt"), "w", encoding="utf-8") as file:
            for key, value in sys_info.items():
                file.write(f"{key}: {value}\n")
    except Exception as e:
        log.error(f"Error writing system info to file: {e}")
    log.info("System Information saved to SystemRam_Info.txt")


# Memory Dump
def memory_dump():
    """
    Performs a memory dump of the current process and saves it to a file.
    """
    log.info("Creating basic memory dump scan...")
    pid = os.getpid()

    try:
        process = psutil.Process(pid)
        dump_path = os.path.join(DUMP_DIR, "Ram_Dump.txt")
        with open(dump_path, "wb", encoding="utf-8") as dump_file:
            total_size = 0

            # Disk space safety check
            required_space = LIMIT_FILE_SIZE * 1024 * 1024 * SAFETY_MARGIN
            free_space = psutil.disk_usage(DUMP_DIR).free
            if free_space < required_space:
                log.error(f"Not enough disk space. Need at least {required_space / (1024 ** 2):.2f} MiB")
                return

            for mem_region in process.memory_maps(grouped=False):
                if 'r' not in mem_region.perms:
                    continue

                try:
                    start, end = (int(addr, 16) for addr in mem_region.addr.split('-')) \
                        if '-' in mem_region.addr else (int(mem_region.addr, 16),
                                                        int(mem_region.addr, 16) + mem_region.rss)
                except Exception as e:
                    log.warning(f"Invalid address format '{mem_region.addr}': {e}")
                    continue

                region_metadata = {
                    '   Start Address': hex(start),
                    '   End Address': hex(end),
                    '   Region Size (bytes)': end - start,
                    '   RSS (bytes)': mem_region.rss,
                    '   Permissions': mem_region.perms,
                    '   Path': mem_region.path,
                    '   Index': mem_region.index,
                }

                try:
                    metadata_str = "Memory Region Metadata:\n" + "\n".join(
                        f"{key}: {value}" for key, value in region_metadata.items()) + "\n\n"
                    metadata_bytes = metadata_str.encode()
                    if (total_size + len(metadata_bytes) > LIMIT_FILE_SIZE * 1024 * 1024) and (LIMIT_FILE_SIZE != 0):
                        dump_file.write(f"Truncated: file exceeded {LIMIT_FILE_SIZE} MiB limit.\n".encode())
                        break
                    dump_file.write(metadata_bytes)
                    total_size += len(metadata_bytes)
                except Exception as e:
                    log.error(f"Error writing memory region metadata: {e}")

    except psutil.Error as e:
        log.error(f"Error accessing process memory: {e}")
    except Exception as e:
        log.error(f"General memory dump error: {e}")

    log.info("Memory scan saved to Ram_Dump.txt")


# Main function to run all tasks
@log.function
def main():
    """
    Executes all memory diagnostics and collection routines.
    """
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
    except Exception as e:
        log.critical(f"Failed to create dump directory '{DUMP_DIR}': {e}")
        return

    log.info("Starting system memory collection tasks...")
    capture_ram_snapshot()
    gather_system_info()
    memory_dump()
    log.info("All tasks completed [dump_memory.py].")


if __name__ == "__main__":
    main()
