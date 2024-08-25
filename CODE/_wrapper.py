import ctypes
import threading
import os
from __lib_actions import *
from __lib_log import Log
from _zipper import zip_and_hash
from _hide_my_tracks import attempt_hide
from _dev import dev_checks, open_file
from _health import backup, update
from _extra import unzip, menu
from _debug import debug


class Checks:
    def __init__(self):
        """
        Initializes an instance of the class.

        Sets the Actions attribute to an instance of the Actions class.
        """
        self.Actions = Actions()

    @staticmethod
    def admin() -> bool:
        """
        Checks if the current user has administrative privileges.

        Returns:
            bool: True if the user is an admin, False otherwise.
        """
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except AttributeError:
            return False

    def uac(self) -> bool:
        """
        Checks if User Account Control (UAC) is enabled on the system.

        This function runs a PowerShell command to retrieve the value of the EnableLUA registry key,
        which indicates whether UAC is enabled. It then returns True if UAC is enabled, False otherwise.

        Returns:
            bool: True if UAC is enabled, False otherwise.
        """
        value = self.Actions.run_command(
            r"powershell (Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System).EnableLUA"
        )
        return int(value.strip("\n")) == 1


class Do:
    @staticmethod
    def get_files_with_extensions(directory: str, List: list) -> list:
        """
        Retrieves a list of files in the specified directory that have the specified extensions.

        Parameters:
            directory (str): The path of the directory to search.
            List (list): The list to append the filenames to.

        Returns:
            list: The list of filenames with the specified extensions.
        """
        for filename in os.listdir(directory):
            if filename.endswith(('.py', '.exe', '.ps1', '.bat')) and not filename.startswith('_'):
                List.append(filename)
        return List

    def execute_file(self, Index: int) -> None:
        """
        Executes a file from the execution list at the specified index.

        Parameters:
            Index (int): The index of the file to be executed in the execution list.

        Returns:
            None
        """
        self.execute(execution_list[Index])
        log.info(f"{execution_list[Index]} executed")

    @staticmethod
    def execute(script: str) -> None:
        """
        Executes a script file and handles its output based on the file extension.

        Parameters:
            script (str): The path of the script file to be executed.

        Returns:
            None
        """
        if script.endswith(".ps1"):
            try:
                unblock_command = (
                    f'powershell.exe -Command "Unblock-File -Path {script}"'
                )
                subprocess.run(unblock_command, shell=True, check=True)
                log.info("PS1 Script unblocked.")
            except Exception as err:
                log.critical(f"Failed to unblock script: {err}", "_W", "G", "E")

        if script.endswith(".py"):
            result = subprocess.Popen(
                ["python", script], stdout=subprocess.PIPE
            ).communicate()[0]
            print(result.decode())
        else:
            result = subprocess.Popen(
                ["powershell.exe", ".\\" + script], stdout=subprocess.PIPE
            ).communicate()[0]
            lines = result.decode().splitlines()
            ID = next(
                (line.split(":")[0].strip() for line in lines if ":" in line), None
            )
            if ID == "INFO":
                log.info("\n".join(lines))
            if ID == "WARNING":
                log.warning("\n".join(lines))
            if ID == "ERROR":
                log.error("\n".join(lines))
            if ID == "CRITICAL":
                if script[0] == "_":
                    fcode = '_' + script[1]
                else:
                    fcode = script[0]
                log.critical("\n".join(lines), fcode, "U", "X")
            else:
                log.debug("\n".join(lines))


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
os.makedirs("../ACCESS/LOGS/", exist_ok=True)
os.makedirs("../ACCESS/LOGS/DEBUG", exist_ok=True)
os.makedirs("../ACCESS/BACKUP/", exist_ok=True)
os.makedirs("../ACCESS/DATA/Hashes", exist_ok=True)
os.makedirs("../ACCESS/DATA/Zip", exist_ok=True)
check_status = Checks()

try:
    action, sub_action = Actions().flags()
except Exception:
    action = Actions().flags()
    action = action[0]
    sub_action = None

# Special actions -> Quit
if action == "debug":
    debug()
    exit(0)
log = Log(debug=DEBUG)
if action == "dev":
    dev_checks()
    exit(0)
if action == "extra":
    log.info("Opening extra tools menu...")
    menu()
    exit(0)
if action == "update":
    log.info("Updating...")
    update()
    log.info("Update complete!")
    exit(0)
if action == "restore":
    log.warning("Sorry, this feature is yet to be implemented. You can manually Restore your backups, We will open "
                "the location for you")
    open_file("../ACCESS/BACKUP/")
    exit(1)
if action == "backup":
    log.info("Backing up...")
    backup(".", "Default_Backup")
    log.debug("Backup complete -> CODE dir")
    backup(".", "Mods_Backup")
    log.debug("Backup complete -> MODS dir")
    log.info("Backup complete!")
    exit(0)
if action == "unzip-extra":
    log.warning("The contents of this directory can be flagged as malicious and enter quarantine, please use with "
                "caution")
    log.info("Unzipping...")
    unzip("..\\EXTRA\\EXTRA.zip")
    log.info("Unzip complete!")
    exit(0)

log.info("Starting Logicytics...")

# Checks for privileges and errors
if not check_status.admin():
    log.critical("Please run this script with admin privileges", "_W", "P", "BA")
    exit(1)
if check_status.uac():
    log.warning("UAC is enabled, this may cause issues")
    log.warning("Please disable UAC if possible")

# Create execution list
execution_list = [
    "driverquery.py",
    "log_miner.py",
    "media_backup.py",
    "online_ip_scraper.py",
    "registry.py",
    "sensitive_data_miner.py",
    "ssh_miner.py",
    "sys_internal.py",
    "sysinfo.py",
    "tasklist.py",
    "tree.bat",
    "wmic.py",
    "browser_miner.ps1",
    "netadapter.ps1",
    "property_scraper.ps1",
    "window_feature_miner.ps1",
]
if action == "minimal":
    execution_list = [
        "driverquery.py",
        "registry.py",
        "sysinfo.py",
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
    execution_list = Do().get_files_with_extensions('../MODS', execution_list)

log.debug(execution_list)

# Check weather to use threading or not
if action == "threaded":
    threads = []
    for index, file in enumerate(execution_list):
        thread = threading.Thread(target=Do().execute_file, args=(index,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
else:
    for file in range(len(execution_list)):  # Loop through List
        Do().execute(execution_list[file])
        log.info(f"{execution_list[file]} executed")

# Zip generated files
if action == "modded":
    zip_loc, hash_loc = zip_and_hash('../MODS', 'MODS', action)
    log.info(zip_loc)
    log.debug(hash_loc)
zip_loc, hash_loc = zip_and_hash('../CODE', 'CODE', action)
log.info(zip_loc)
log.debug(hash_loc)

# Attempt event log deletion
attempt_hide()

# Finish with sub actions
log.info("Completed successfully")
if sub_action == "shutdown":
    log.info("Shutting down...")
    os.system("shutdown /s /t 0")
if sub_action == "reboot":
    log.info("Rebooting...")
    os.system("shutdown /r /t 0")
if sub_action == "webhook":
    log.info("Sending webhook...")
    if WEBHOOK is None or WEBHOOK == "":
        log.critical("WEBHOOK URL not set and the request action was webhook", "_W", "P", "BA")
        exit(1)
    # TODO Implement

log.info("Exiting...")
exit(0)
