from __future__ import annotations

import threading
from typing import Any

from logicytics import *


# Initialization
FileManagement.mkdir()
log = Log({"log_level": DEBUG})


class Health:
    @staticmethod
    @log.function
    def backup(directory: str, name: str):
        """
        Creates a backup of a specified directory by zipping its contents and moving it to a designated backup location.

        Args:
            directory (str): The path to the directory to be backed up.
            name (str): The name of the backup file.

        Returns:
            None
        """
        # Check if backup exists, delete it if so
        if os.path.exists(f"../ACCESS/BACKUP/{name}.zip"):
            os.remove(f"../ACCESS/BACKUP/{name}.zip")

        # Zip the directory and move it to the backup location
        with zipfile.ZipFile(f"{name}.zip", "w") as zip_file:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(str(file_path), start=os.getcwd())
                    zip_file.write(str(file_path), arcname=relative_path)

        shutil.move(f"{name}.zip", "../ACCESS/BACKUP")

    @staticmethod
    @log.function
    def update() -> tuple[str, str]:
        """
        Updates the repository by pulling the latest changes from the remote repository.

        This function navigates to the parent directory, pulls the latest changes using Git,
        and then returns to the current working directory.

        Returns:
            str: The output from the git pull command.
        """
        # Check if git command is available
        if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
            return "Git is not installed or not available in the PATH.", "error"

        # Check if the project is a git repository
        if not os.path.exists(os.path.join(os.getcwd(), "../.git")):
            return "This project is not a git repository. The update flag uses git.", "error"

        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        os.chdir(parent_dir)
        output = subprocess.run(["git", "pull"], capture_output=True).stdout.decode()
        os.chdir(current_dir)
        return output, "info"


def get_flags():
    """
    Retrieves the command-line flags and sub-actions.

    This function checks if the flags are provided as a tuple. If so, it attempts to unpack
    the tuple into ACTION and SUB_ACTION. If an exception occurs, it sets SUB_ACTION to None.
    If the flags are not a tuple, it prints the help message and exits the program.

    """
    if isinstance(Flag.data(), tuple):
        global ACTION, SUB_ACTION
        try:
            # Get flags
            ACTION, SUB_ACTION = Flag.data()
        except Exception:
            ACTIONS = Flag.data()
            ACTION = ACTIONS[0]
            SUB_ACTION = None
    else:
        parser = Flag.data()
        parser.print_help()
        input("Press Enter to exit...")
        exit(1)


@log.function
def special_execute(file_path: str):
    """
    Executes a Python script in a new command prompt window.

    Args:
        file_path (str): The relative path to the Python script to be executed.
    """
    sr_current_dir = os.path.dirname(os.path.abspath(__file__))
    sr_script_path = os.path.join(sr_current_dir, file_path)
    sr_process = subprocess.Popen(["cmd.exe", "/c", "start", "python", sr_script_path])
    sr_process.wait()
    exit(0)


@log.function
def handle_special_actions():
    """
    Handles special actions based on the provided action flag.

    This function checks the value of the `action` variable and performs
    corresponding special actions such as opening debug, developer, or extra
    tools menus, updating the repository, restoring backups, creating backups,
    or unzipping extra files.
    """
    # Special actions -> Quit
    if ACTION == "debug":
        log.info("Opening debug menu...")
        special_execute("_debug.py")

    messages = Check.sys_internal_zip()
    if messages:
        # If there are messages, log them with debug
        log.debug(messages)

    if ACTION == "dev":
        log.info("Opening developer menu...")
        special_execute("_dev.py")

    if ACTION == "extra":
        log.info("Opening extra tools menu...")
        special_execute("_extra.py")

    if ACTION == "update":
        log.info("Updating...")
        message, log_type = Health.update()
        log.string(message, log_type)
        if log_type == "info":
            log.info("Update complete!")
        else:
            log.error("Update failed!")
        input("Press Enter to exit...")
        exit(0)

    if ACTION == "restore":
        log.warning(
            "Sorry, this feature is yet to be implemented. You can manually Restore your backups, We will open "
            "the location for you"
        )
        FileManagement.open_file("../ACCESS/BACKUP/")
        input("Press Enter to exit...")
        exit(1)

    if ACTION == "backup":
        log.info("Backing up...")
        Health.backup(".", "Default_Backup")
        log.debug("Backup complete -> CODE dir")
        Health.backup("../MODS", "Mods_Backup")
        log.debug("Backup complete -> MODS dir")
        log.info("Backup complete!")
        input("Press Enter to exit...")
        exit(0)

    if ACTION == "unzip_extra":
        log.warning(
            "The contents of this directory can be flagged as malicious and enter quarantine, please use with "
            "caution"
        )
        log.info("Unzipping...")
        FileManagement.unzip(Path("..\\EXTRA\\EXTRA.zip"))
        log.info("Unzip complete!")
        input("Press Enter to exit...")
        exit(0)


@log.function
def check_privileges():
    """
    Checks if the script is running with administrative privileges and handles UAC (User Account Control) settings.

    This function verifies if the script has admin privileges. If not, it either logs a warning (in debug mode) or
    prompts the user to run the script with admin privileges and exits. It also checks if UAC is enabled and logs
    warnings accordingly.
    """
    if not Check.admin():
        if DEBUG == "DEBUG":
            log.warning("Running in debug mode, continuing without admin privileges - This may cause issues")
        else:
            log.critical(
                "Please run this script with admin privileges - To ignore this message, run with DEBUG in config")
            input("Press Enter to exit...")
            exit(1)

    if Check.uac():
        log.warning("UAC is enabled, this may cause issues - Please disable UAC if possible")


@log.function
def generate_execution_list() -> list | list[str] | list[str | Any]:
    """
    Creates an execution list based on the provided action.

    Returns:
        list: The execution list of scripts to be executed.
    """
    execution_list = Get.list_of_files(".")
    execution_list.remove("sensitive_data_miner.py")
    execution_list.remove("dir_list.py")
    execution_list.remove("tree.ps1")

    if ACTION == "minimal":
        execution_list = [
            "cmd_commands.py",
            "registry.py",
            "tasklist.py",
            "wmic.py",
            "netadapter.ps1",
            "property_scraper.ps1",
            "window_feature_miner.ps1",
            "event_log.py",
        ]

    if ACTION == "nopy":
        execution_list = [
            "browser_miner.ps1",
            "netadapter.ps1",
            "property_scraper.ps1",
            "window_feature_miner.ps1",
            "tree.ps1"
        ]

    if ACTION == "modded":
        # Add all files in MODS to execution list
        execution_list = Get.list_of_files("../MODS", execution_list)

    if ACTION == "depth":
        log.warning("This flag will use clunky and huge scripts, and so may take a long time, but reap great rewards.")
        execution_list.append("sensitive_data_miner.py")
        execution_list.append("dir_list.py")
        execution_list.append("tree.ps1")
        log.warning("This flag will use threading!")

    log.debug(f"The following will be executed: {execution_list}")
    return execution_list


@log.function
def execute_scripts():
    """Executes the scripts in the execution list based on the action."""
    # Check weather to use threading or not, as well as execute code
    if ACTION == "threaded" or ACTION == "depth":
        def threaded_execution(execution_list_thread, index_thread):
            log.debug(f"Thread {index_thread} started")
            try:
                log.parse_execution(Execute.script(execution_list_thread[index_thread]))
                log.info(f"{execution_list_thread[index_thread]} executed")
            except UnicodeDecodeError as err:
                log.error(f"Error in thread: {err}")
            except Exception as err:
                log.error(f"Error in thread: {err}")
            log.debug(f"Thread {index_thread} finished")

        log.debug("Using threading")
        threads = []
        execution_list = generate_execution_list(ACTION)
        for index, file in enumerate(execution_list):
            thread = threading.Thread(
                target=threaded_execution,
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
            execution_list = generate_execution_list(ACTION)
            for file in range(len(execution_list)):  # Loop through List
                log.parse_execution(Execute.script(execution_list[file]))
                log.info(f"{execution_list[file]} executed")
        except UnicodeDecodeError as e:
            log.error(f"Error in code: {e}")
        except Exception as e:
            log.error(f"Error in code: {e}")


@log.function
def zip_generated_files():
    """Zips generated files based on the action."""

    def zip_and_log(directory, name):
        zip_values = FileManagement.Zip.and_hash(directory, name, ACTION)
        if isinstance(zip_values, str):
            log.error(zip_values)
        else:
            zip_loc, hash_loc = zip_values
            log.info(zip_loc)
            log.debug(hash_loc)

    if ACTION == "modded":
        zip_and_log("..\\MODS", "MODS")
    zip_and_log(".", "CODE")


@log.function
def handle_sub_action():
    """
    Handles sub-actions based on the provided sub_action flag.

    This function checks the value of the `sub_action` variable and performs
    corresponding sub-actions such as shutting down or rebooting the system.
    """
    if SUB_ACTION == "shutdown":
        subprocess.call("shutdown /s /t 3", shell=False)
    elif SUB_ACTION == "reboot":
        subprocess.call("shutdown /r /t 3", shell=False)
    # elif sub_action == "webhook":
        # Implement this in future
        # log.warning("This feature is not implemented yet! Sorry")


if __name__ == "__main__":
    # Get flags and configs
    get_flags()
    # Check for special actions
    handle_special_actions()
    # Check for privileges and errors
    check_privileges()
    # Execute scripts
    log.info("Starting Logicytics...")
    execute_scripts()
    # Zip generated files
    zip_generated_files()
    # Finish with sub actions
    log.info("Completed successfully!")
    # Finish with sub actions
    log.newline()
    handle_sub_action()
    # Finish
    input("Press Enter to exit...")
