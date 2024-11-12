import threading

from __lib_class import *
from _health import backup, update
from _hide_my_tracks import attempt_hide
from _zipper import Zip

# Initialization
FileManagement.mkdir()

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})

"""
This python script is the main entry point for the tool called Logicytics.
The script performs various actions based on command-line flags and configuration settings.

Here's a high-level overview of what the script does:

1. Initializes directories and checks for admin privileges.
2. Parses command-line flags and sets up logging.
3. Performs special actions based on flags, such as debugging, updating, or restoring backups.
4. Creates an execution list of files to run, which can be filtered based on flags.
5. Runs the files in the execution list, either sequentially or in parallel using threading.
6. Zips generated files and attempts to delete event logs.
7. Performs sub-actions, such as shutting down or rebooting the system, or sending a webhook.

The script appears to be designed to be highly configurable and modular, 
with many options and flags that can be used to customize its behavior.
"""


if isinstance(Flag.data(), tuple):
    try:
        # Get flags
        action, sub_action = Flag.data()
    except Exception:
        action = Flag.data()
        action = action[0]
        sub_action = None
else:
    parser = Flag.data()
    parser.print_help()
    input("Press Enter to exit...")
    exit(1)


def special_run(file_path: str):
    sr_current_dir = os.path.dirname(os.path.abspath(__file__))
    sr_script_path = os.path.join(sr_current_dir, file_path)
    sr_process = subprocess.Popen(["cmd.exe", "/c", "start", "python", sr_script_path])
    sr_process.wait()
    exit(0)


# Special actions -> Quit
if action == "debug":
    log.info("Opening debug menu...")
    special_run("_debug.py")

messages = Check.sys_internal_zip()
if messages:
    # If there are messages, log them with debug
    log.debug(messages)

if action == "dev":
    log.info("Opening developer menu...")
    special_run("_dev.py")

if action == "extra":
    log.info("Opening extra tools menu...")
    special_run("_extra.py")

if action == "update":
    log.info("Updating...")
    message, log_type = update()
    log.string(message, log_type)
    log.info("Update complete!")
    input("Press Enter to exit...")
    exit(0)

if action == "restore":
    log.warning(
        "Sorry, this feature is yet to be implemented. You can manually Restore your backups, We will open "
        "the location for you"
    )
    FileManagement.open_file("../ACCESS/BACKUP/")
    input("Press Enter to exit...")
    exit(1)

if action == "backup":
    log.info("Backing up...")
    backup(".", "Default_Backup")
    log.debug("Backup complete -> CODE dir")
    backup("../MODS", "Mods_Backup")
    log.debug("Backup complete -> MODS dir")
    log.info("Backup complete!")
    input("Press Enter to exit...")
    exit(0)

if action == "unzip_extra":
    log.warning(
        "The contents of this directory can be flagged as malicious and enter quarantine, please use with "
        "caution"
    )
    log.info("Unzipping...")
    FileManagement.unzip(Path("..\\EXTRA\\EXTRA.zip"))
    log.info("Unzip complete!")
    input("Press Enter to exit...")
    exit(0)

log.info("Starting Logicytics...")

# Check for privileges and errors
if not Check.admin():
    if DEBUG == "DEBUG":
        log.warning("Running in debug mode, continuing without admin privileges - This may cause issues")
    else:
        log.critical("Please run this script with admin privileges - To ignore this message, run with DEBUG in config")
        input("Press Enter to exit...")
        exit(1)

if Check.uac():
    log.warning("UAC is enabled, this may cause issues")
    log.warning("Please disable UAC if possible")

# Create execution list [default]
execution_list = [
    "cmd_commands.py",
    "log_miner.py",
    "media_backup.py",
    "registry.py",
    "ssh_miner.py",
    "sys_internal.py",
    "tasklist.py",
    "tree.bat",
    "wmic.py",
    "browser_miner.ps1",
    "netadapter.ps1",
    "property_scraper.ps1",
    "window_feature_miner.ps1",
    "wifi_stealer.py",
]

if action == "minimal":
    execution_list = [
        "cmd_commands.py",
        "registry.py",
        "tasklist.py",
        "tree.bat",
        "wmic.py",
        "netadapter.ps1",
        "property_scraper.ps1",
        "window_feature_miner.ps1",
    ]

if action == "nopy":
    execution_list = [
        "browser_miner.ps1",
        "netadapter.ps1",
        "property_scraper.ps1",
        "window_feature_miner.ps1",
        "tree.bat"
    ]

if action == "modded":
    # Add all files in MODS to execution list
    execution_list = Get.list_of_files(Path("../MODS"), execution_list)

if action == "depth":
    log.warning("This flag will use clunky and huge scripts, and so may take a long time, but reap great rewards.")
    execution_list.append("sensitive_data_miner.py")
    execution_list.append("dir_list.py")
    log.warning("This flag will use threading!")
    action = "threaded"

log.debug(f"The following will be executed: {execution_list}")

# Check weather to use threading or not, as well as execute code
if action == "threaded":
    def threaded_execution(execution_list_thread, index_thread):
        try:
            thread_log = Execute.file(execution_list_thread, index_thread)
            if thread_log[0]:
                log.info(thread_log[0])
            log.info(thread_log[1])
        except UnicodeDecodeError as err:
            log.error(f"Error in thread: {err}")
        except Exception as err:
            log.error(f"Error in thread: {err}")

    threads = []
    for index, file in enumerate(execution_list):
        thread = threading.Thread(
            target=Execute.file,
            args=(
                execution_list,
                index,
            ),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
else:
    try:
        for file in range(len(execution_list)):  # Loop through List
            log.info(Execute.script(execution_list[file]))
            log.info(f"{execution_list[file]} executed")
    except UnicodeDecodeError as e:
        log.error(f"Error in thread: {e}")
    except Exception as e:
        log.error(f"Error in thread: {e}")

# Zip generated files
if action == "modded":
    zip_loc_mod, hash_loc = Zip().and_hash("..\\MODS", "MODS", action)
    log.info(zip_loc_mod)
    zip_values = Zip().and_hash("..\\MODS", "MODS", action)
    if isinstance(zip_values, str):
        log.error(zip_values)
    else:
        zip_loc_mod, hash_loc = zip_values
        log.info(zip_loc_mod)
        log.debug(hash_loc)

zip_values = Zip().and_hash(".", "CODE", action)
if isinstance(zip_values, str):
    # If error, log it
    log.error(zip_values)
else:
    zip_loc, hash_loc = zip_values
    log.info(zip_loc)
    log.debug(hash_loc)

# Attempt event log deletion
attempt_hide()

# Finish with sub actions
log.info("Completed successfully")
if sub_action == "shutdown":
    log.info("Shutting down in 3 seconds...")
    subprocess.call("shutdown /s /t 3", shell=False)
if sub_action == "reboot":
    log.info("Rebooting in 3 seconds...")
    subprocess.call("shutdown /r /t 3", shell=False)
# if sub_action == "webhook":
    # Implement this in future
    # log.warning("This feature is not implemented yet! Sorry")

log.info("Exiting...")
input("Press Enter to exit...")
log.newline()
