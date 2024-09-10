import ctypes
import os.path
import threading
import zipfile

from __lib_actions import *
from __lib_log import Log
from _debug import debug
from _dev import dev_checks, open_file
from _extra import unzip, menu
from _health import backup, update
from _hide_my_tracks import attempt_hide
from _zipper import zip_and_hash


class Check:
    def __init__(self):
        """
        Initializes an instance of the class.

        Sets the Actions attribute to an instance of the Actions class.
        """
        self.Actions = Actions()

    @staticmethod
    def admin() -> bool:
        """
        Check if the current user has administrative privileges.

        Returns:
            bool: True if the user is an admin, False otherwise.
        """
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except AttributeError:
            return False

    def uac(self) -> bool:
        """
        Check if User Account Control (UAC) is enabled on the system.

        This function runs a PowerShell command to retrieve the value of the EnableLUA registry key,
        which indicates whether UAC is enabled. It then returns True if UAC is enabled, False otherwise.

        Returns:
            bool: True if UAC is enabled, False otherwise.
        """
        value = self.Actions.run_command(
            r"powershell (Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System).EnableLUA"
        )
        return int(value.strip("\n")) == 1

    @staticmethod
    def sys_internal_zip():
        # TODO Test me
        try:
            ignore_file = os.path.exists("SysInternal_Suite/.ignore")
            zip_file = os.path.exists("SysInternal_Suite/SysInternal_Suite.zip")

            if zip_file and not ignore_file:
                log.info("Extracting SysInternal_Suite zip")
                with zipfile.ZipFile("SysInternal_Suite/SysInternal_Suite.zip") as zip_ref:
                    zip_ref.extractall("SysInternal_Suite")

                # Remove the original zip file
                os.remove("SysInternal_Suite/SysInternal_Suite.zip")

            elif ignore_file:
                log.info("Found .ignore file, skipping SysInternal_Suite zip extraction")

        except Exception as err:
            log.critical(f"Failed to unzip SysInternal_Suite: {err}", "_L", "G", "CS")
            exit(f"Failed to unzip SysInternal_Suite: {err}")


class Execute:
    @staticmethod
    def get_files(directory: str, file_list: list) -> list:
        """
        Retrieves a list of files in the specified directory that have the specified extensions.
        Parameters:
            directory (str): The path of the directory to search.
            file_list (list): The list to append the filenames to.
        Returns:
            list: The list of filenames with the specified extensions.
        """
        for filename in os.listdir(directory):
            if (
                    filename.endswith((".py", ".exe", ".ps1", ".bat"))
                    and not filename.startswith("_")
                    and filename != "Logicytics.py"
            ):
                file_list.append(filename)
        return file_list

    def file(self, Index: int) -> None:
        """
        Executes a file from the execution list at the specified index.
        Parameters:
            Index (int): The index of the file to be executed in the execution list.
        Returns:
            None
        """
        self.execute_script(execution_list[Index])
        log.info(f"{execution_list[Index]} executed")

    def execute_script(self, script: str) -> None:
        """
        Executes a script file and handles its output based on the file extension.
        Parameters:
            script (str): The path of the script file to be executed.
        Returns:
            None
        """
        if script.endswith(".ps1"):
            self.__run_ps1_script(script)
            self.__run_other_script(script)
        elif script.endswith(".py"):
            self.__run_python_script(script)
        else:
            self.__run_other_script(script)

    @staticmethod
    def __run_ps1_script(script: str) -> None:
        """
        Unblocks and runs a PowerShell (.ps1) script.
        Parameters:
            script (str): The path of the PowerShell script.
        Returns:
            None
        """
        try:
            unblock_command = f'powershell.exe -Command "Unblock-File -Path {script}"'
            subprocess.run(unblock_command, shell=True, check=True)
            log.info("PS1 Script unblocked.")
        except Exception as err:
            log.critical(f"Failed to unblock script: {err}", "_L", "G", "E")

    @staticmethod
    def __run_python_script(script: str) -> None:
        """
        Runs a Python (.py) script.
        Parameters:
            script (str): The path of the Python script.
        Returns:
            None
        """
        result = subprocess.Popen(
            ["python", script], stdout=subprocess.PIPE
        ).communicate()[0]
        print(result.decode())

    @staticmethod
    def __run_other_script(script: str) -> None:
        """
        Runs a script with other extensions and logs output based on its content.
        Parameters:
            script (str): The path of the script.
        Returns:
            None
        """
        result = subprocess.Popen(
            ["powershell.exe", ".\\" + script], stdout=subprocess.PIPE
        ).communicate()[0]
        lines = result.decode().splitlines()
        ID = next((line.split(":")[0].strip() for line in lines if ":" in line), None)

        log_funcs = {
            "INFO": log.info,
            "WARNING": log.warning,
            "ERROR": log.error,
            "CRITICAL": log.critical,
            None: log.debug,
        }

        log_func = log_funcs.get(ID, log.debug)
        log_func("\n".join(lines))


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
Actions().mkdir()
check_status = Check()
check_status.sys_internal_zip()

try:
    # Get flags
    action, sub_action = Actions().flags()
except Exception:
    action = Actions().flags()
    action = action[0]
    sub_action = None

# Special actions -> Quit
if action == "debug":
    debug()
    input("Press Enter to exit...")
    exit(0)

log = Log(debug=DEBUG)

if action == "dev":
    dev_checks()
    input("Press Enter to exit...")
    exit(0)

if action == "extra":
    log.info("Opening extra tools menu...")
    menu()
    input("Press Enter to exit...")
    exit(0)

if action == "update":
    log.info("Updating...")
    update()
    log.info("Update complete!")
    input("Press Enter to exit...")
    exit(0)

if action == "restore":
    log.warning(
        "Sorry, this feature is yet to be implemented. You can manually Restore your backups, We will open "
        "the location for you"
    )
    open_file("../ACCESS/BACKUP/")
    input("Press Enter to exit...")
    exit(1)

if action == "backup":
    log.info("Backing up...")
    backup(".", "Default_Backup")
    log.debug("Backup complete -> CODE dir")
    backup(".", "Mods_Backup")
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
    input("Press Enter to exit...")
    exit(1)

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
    execution_list.remove("sensitive_data_miner.py")
    threads = []
    for index, file in enumerate(execution_list):
        thread = threading.Thread(target=Execute().file, args=(index,))
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
    zip_loc_mod, hash_loc = zip_and_hash("..\\MODS", "MODS", action)
    log.info(zip_loc_mod)
    log.debug(hash_loc)

zip_loc, hash_loc = zip_and_hash("..\\CODE", "CODE", action)
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
    log.warning("This feature is not fully implemented yet")
    """
    log.info("Sending webhook...")
    if WEBHOOK is None or WEBHOOK == "":
        log.critical("WEBHOOK URL not set and the request action was webhook", "_L", "P", "BA")
        input("Press Enter to exit...")
        exit(1)
    """

log.info("Exiting...")
input("Press Enter to exit...")
