import threading
from _debug import debug
from _extra import menu, unzip
from _health import backup, update
from _hide_my_tracks import attempt_hide
from _zipper import Zip
from __lib_class import *

log = Log(debug=DEBUG)
log_funcs = {
    "INFO": log.info,
    "WARNING": log.warning,
    "ERROR": log.error,
    "CRITICAL": log.critical,
    None: log.debug,
}


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

# Initialization
Actions.mkdir()
check_status = Check()

if isinstance(Actions().flags(), tuple):
    try:
        # Get flags
        action, sub_action = Actions().flags()
    except Exception:
        action = Actions().flags()
        action = action[0]
        sub_action = None
else:
    parser = Actions().flags()
    parser.print_help()
    input("Press Enter to exit...")
    exit(1)

# Special actions -> Quit
if action == "debug":
    debug()
    input("Press Enter to exit...")
    exit(0)

check_status.sys_internal_zip()

if action == "dev":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "_dev.py")
    process = subprocess.Popen(['cmd.exe', '/c', 'start', 'python', script_path])
    process.wait()
    exit(0)

if action == "extra":
    log.info("Opening extra tools menu...")
    menu()
    input("Press Enter to exit...")
    exit(0)

if action == "update":
    log.info("Updating...")
    log.info(update())
    log.info("Update complete!")
    input("Press Enter to exit...")
    exit(0)

if action == "restore":
    log.warning(
        "Sorry, this feature is yet to be implemented. You can manually Restore your backups, We will open "
        "the location for you"
    )
    Actions().open_file("../ACCESS/BACKUP/")
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
    unzip("..\\EXTRA\\EXTRA.zip")
    log.info("Unzip complete!")
    input("Press Enter to exit...")
    exit(0)


log.info("Starting Logicytics...")


# Check for privileges and errors
if not check_status.admin():
    log.critical("Please run this script with admin privileges", "_L", "P", "BA")
    if not DEBUG:
        input("Press Enter to exit...")
        exit(1)
    else:
        log.warning("Running in debug mode, continuing without admin privileges")

if check_status.uac():
    log.warning("UAC is enabled, this may cause issues")
    log.warning("Please disable UAC if possible")

# Create execution list
execution_list = [
    "driverquery+sysinfo.py",
    "log_miner.py",
    "media_backup.py",
    "online_ip_scraper.py",
    "registry.py",
    "sensitive_data_miner.py",
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
        "driverquery+sysinfo.py",
        "registry.py",
        "tasklist.py",
        "tree.bat",
        "wmic.py",
        "netadapter.ps1",
        "property_scraper.ps1",
        "window_feature_miner.ps1",
    ]

if action == "exe":
    log.warning(
        "EXE is not fully implemented yet - For now its only SysInternal and WMIC wrappers"
    )
    execution_list = ["sys_internal.py", "wmic.py"]

if action == "modded":
    # Add all files in MODS to execution list
    execution_list = Execute.get_files("../MODS", execution_list)


log.debug(execution_list)

# Check weather to use threading or not
if action == "threaded":
    log.warning("Threading does not support sensitive data miner yet, ignoring")
    execution_list.remove("sensitive_data_miner.py")
    threads = []
    for index, file in enumerate(execution_list):
        thread = threading.Thread(
            target=Execute().file,
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
    for file in range(len(execution_list)):  # Loop through List
        Execute().execute_script(execution_list[file])
        log.info(f"{execution_list[file]} executed")

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
if sub_action == "webhook":
    # Implement this in future
    log.warning("This feature is not fully implemented yet! Sorry")

log.info("Exiting...")
input("Press Enter to exit...")
# Special feature that allows to create a `-` line only
log.debug("*-*")
