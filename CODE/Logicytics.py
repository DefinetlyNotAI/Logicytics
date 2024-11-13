from __future__ import annotations

import threading
from typing import Any

from __lib_class import *
from _health import backup, update
from _hide_my_tracks import attempt_hide
from _zipper import Zip


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


def get_flags():
    if isinstance(Flag.data(), tuple):
        try:
            # Get flags
            actions, sub_actions = Flag.data()
        except Exception:
            actions = Flag.data()
            actions = actions[0]
            sub_actions = None
    else:
        parser = Flag.data()
        parser.print_help()
        input("Press Enter to exit...")
        exit(1)
    return actions, sub_actions


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


def handle_special_actions():
    # Special actions -> Quit
    if action == "debug":
        log.info("Opening debug menu...")
        special_execute("_debug.py")

    messages = Check.sys_internal_zip()
    if messages:
        # If there are messages, log them with debug
        log.debug(messages)

    if action == "dev":
        log.info("Opening developer menu...")
        special_execute("_dev.py")

    if action == "extra":
        log.info("Opening extra tools menu...")
        special_execute("_extra.py")

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


def check_privileges():
    if not Check.admin():
        if DEBUG == "DEBUG":
            log.warning("Running in debug mode, continuing without admin privileges - This may cause issues")
        else:
            log.critical(
                "Please run this script with admin privileges - To ignore this message, run with DEBUG in config")
            input("Press Enter to exit...")
            exit(1)

    if Check.uac():
        log.warning("UAC is enabled, this may cause issues")
        log.warning("Please disable UAC if possible")


def generate_execution_list(actions: str) -> list | list[str] | list[str | Any]:
    """
    Creates an execution list based on the provided action.

    Args:
        actions (str): The action to determine the execution list.

    Returns:
        list: The execution list of scripts to be executed.
    """
    execution_list = [
        "cmd_commands.py",
        "log_miner.py",
        "media_backup.py",
        "registry.py",
        "ssh_miner.py",
        "sys_internal.py",
        "tasklist.py",
        "tree.ps1",
        "wmic.py",
        "browser_miner.ps1",
        "netadapter.ps1",
        "property_scraper.ps1",
        "window_feature_miner.ps1",
        "wifi_stealer.py",
    ]

    if actions == "minimal":
        execution_list = [
            "cmd_commands.py",
            "registry.py",
            "tasklist.py",
            "tree.ps1",
            "wmic.py",
            "netadapter.ps1",
            "property_scraper.ps1",
            "window_feature_miner.ps1",
        ]

    if actions == "nopy":
        execution_list = [
            "browser_miner.ps1",
            "netadapter.ps1",
            "property_scraper.ps1",
            "window_feature_miner.ps1",
            "tree.ps1"
        ]

    if actions == "modded":
        # Add all files in MODS to execution list
        execution_list = Get.list_of_files(Path("../MODS"), execution_list)

    if actions == "depth":
        log.warning("This flag will use clunky and huge scripts, and so may take a long time, but reap great rewards.")
        execution_list.append("sensitive_data_miner.py")
        execution_list.append("dir_list.py")
        log.warning("This flag will use threading!")

    log.debug(f"The following will be executed: {execution_list}")
    return execution_list


def execute_scripts():
    """Executes the scripts in the execution list based on the action."""
    # Check weather to use threading or not, as well as execute code
    if action == "threaded" or action == "depth":
        def threaded_execution(execution_list_thread, index_thread):
            log.debug(f"Thread {index_thread} started")
            try:
                log.execute_log_parse(Execute.script(execution_list_thread[index_thread]))
                log.info(f"{execution_list_thread[index_thread]} executed")
            except UnicodeDecodeError as err:
                log.error(f"Error in thread: {err}")
            except Exception as err:
                log.error(f"Error in thread: {err}")
            log.debug(f"Thread {index_thread} finished")

        log.debug("Using threading")
        threads = []
        execution_list = generate_execution_list(action)
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
            execution_list = generate_execution_list(action)
            for file in range(len(execution_list)):  # Loop through List
                log.execute_log_parse(Execute.script(execution_list[file]))
                log.info(f"{execution_list[file]} executed")
        except UnicodeDecodeError as e:
            log.error(f"Error in code: {e}")
        except Exception as e:
            log.error(f"Error in code: {e}")


def zip_generated_files():
    """Zips generated files based on the action."""
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


def handle_sub_action():
    if sub_action == "shutdown":
        subprocess.call("shutdown /s /t 3", shell=False)
    elif sub_action == "reboot":
        subprocess.call("shutdown /r /t 3", shell=False)
    # elif sub_action == "webhook":
    # Implement this in future
    # log.warning("This feature is not implemented yet! Sorry")


# Initialization
FileManagement.mkdir()

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})
    # Get flags and configs
    action, sub_action = get_flags()
    # Check for special actions
    handle_special_actions()
    # Check for privileges and errors
    check_privileges()
    # Execute scripts
    log.info("Starting Logicytics...")
    execute_scripts()
    # Zip generated files
    zip_generated_files()
    # Attempt event log deletion
    attempt_hide()
    # Finish with sub actions
    log.info("Completed successfully!")
    # Finish with sub actions
    log.newline()
    handle_sub_action()
    # Finish
    input("Press Enter to exit...")
