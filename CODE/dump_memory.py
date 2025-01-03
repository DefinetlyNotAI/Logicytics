import os
import platform
import struct

import psutil

from logicytics import Log, DEBUG

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})

# TODO v3.3.1
#  psutil.virtual_memory(): used, free, percent, total
#  psutil.swap_memory(): used, free, percent, total

# If the file size exceeds this limit, the file will be truncated with a message
# Put 0 to disable the limit
LIMIT_FILE_SIZE = 20  # Always in MiB


# Capture RAM Snapshot
def capture_ram_snapshot():
    log.info("Capturing RAM Snapshot...")
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    with open("Ram_Snapshot.txt", "w") as file:
        try:
            file.write(f"Total RAM: {memory.total / (1024 ** 3):.2f} GB\n")
            file.write(f"Used RAM: {memory.used / (1024 ** 3):.2f} GB\n")
            file.write(f"Available RAM: {memory.available / (1024 ** 3):.2f} GB\n")
            file.write(f"Total Swap: {swap.total / (1024 ** 3):.2f} GB\n")
            file.write(f"Used Swap: {swap.used / (1024 ** 3):.2f} GB\n")
            file.write(f"Free Swap: {swap.free / (1024 ** 3):.2f} GB\n")
            file.write(f"Percent RAM Used: {memory.percent:.2f}%\n")
        except Exception as e:
            log.error(f"Error writing RAM snapshot: {e}")
            file.write("Error writing RAM snapshot.")
    log.info("RAM Snapshot saved to Ram_Snapshot.txt")


# Gather system information
def gather_system_info():
    log.info("Gathering system information...")
    try:
        sys_info = {
            'Architecture': platform.architecture(),
            'System': platform.system(),
            'Machine': platform.machine(),
            'Processor': platform.processor(),
            'Page Size (bytes)': struct.calcsize("P"),
        }
    except Exception as e:
        log.error(f"Error gathering system information: {e}")
        sys_info = {'Error': 'Failed to gather system information'}
    with open("SystemRam_Info.txt", "w") as file:
        for key, value in sys_info.items():
            file.write(f"{key}: {value}\n")
    log.info("System Information saved to SystemRam_Info.txt")


# Memory Dump (Windows-specific, using psutil)
def memory_dump():
    log.info("Creating basic memory dump scan...")
    pid = os.getpid()
    try:
        process = psutil.Process(pid)
        with open("Ram_Dump.txt", "wb") as dump_file:
            total_size = 0
            for mem_region in process.memory_maps(grouped=False):
                # Check if the memory region is readable ('r' permission)
                if 'r' in mem_region.perms:
                    # Extract start and end addresses from the memory region string
                    if '-' in mem_region.addr:
                        start, end = [int(addr, 16) for addr in mem_region.addr.split('-')]
                    else:
                        start = int(mem_region.addr, 16)
                        end = start + mem_region.rss

                    # Gather memory region metadata
                    region_metadata = {
                        '   Start Address': hex(start),
                        '   End Address': hex(end),
                        '   RSS (bytes)': mem_region.rss,  # Using rss as size
                        '   Permissions': mem_region.perms,
                        '   Path': mem_region.path,  # Path is often available for shared memory regions
                        '   Index': mem_region.index,
                    }

                    # Try getting more detailed memory information
                    try:
                        # Check if the memory region corresponds to a file and add file metadata
                        if mem_region.path:
                            # Try to get device and inode-like info
                            file_path = mem_region.path
                            # file_device = win32api.GetFileAttributes(file_path) if os.path.exists(file_path) else 'N/A'
                            region_metadata['   File Path'] = file_path
                            # region_metadata['File Device'] = file_device

                    except Exception as e:
                        log.error(f"Error adding extra file information: {str(e)}")

                    # Write the metadata to the dump file
                    try:
                        metadata_str = "Memory Region Metadata:\n" + "\n".join(
                            f"{key}: {value}" for key, value in region_metadata.items()) + "\n\n"
                        metadata_bytes = metadata_str.encode()
                        if total_size + len(metadata_bytes) > LIMIT_FILE_SIZE * 1024 * 1024 and LIMIT_FILE_SIZE != 0:
                            dump_file.write(f"Truncated due to file exceeding {LIMIT_FILE_SIZE}\n"
                                            "Additional memory regions not included.\n".encode())
                            break
                        dump_file.write(metadata_bytes)
                        total_size += len(metadata_bytes)
                    except Exception as e:
                        log.error(f"Error writing memory region metadata: {str(e)}")
    except psutil.Error as e:
        log.error(f"Error opening process memory: {str(e)}")
    except Exception as e:
        log.error(f"Error creating memory scan: {str(e)}")

    log.info("Memory scan saved to Ram_Dump.txt")


# Main function to run all tasks
@log.function
def main():
    log.info("Starting system memory collection tasks...")
    capture_ram_snapshot()
    gather_system_info()
    memory_dump()
    log.info("All tasks completed [dump_memory.py].")


if __name__ == "__main__":
    main()
