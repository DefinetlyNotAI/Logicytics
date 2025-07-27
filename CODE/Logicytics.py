from __future__ import annotations

import gc
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from prettytable import PrettyTable

from logicytics import (
    Log,
    execute,
    check,
    get,
    file_management,
    flag,
    DEBUG,
    DELETE_LOGS,
    config,
)

# Initialization
log = Log({"log_level": DEBUG, "delete_log": DELETE_LOGS})
ACTION, SUB_ACTION = None, None
MAX_WORKERS = config.getint(
    "Settings", "max_workers", fallback=min(32, (os.cpu_count() or 1) + 4)
)
log.debug(f"MAX_WORKERS: {MAX_WORKERS}")


class ExecuteScript:
    def __init__(self):
        self.execution_list = self.__generate_execution_list()

    @staticmethod
    def __safe_remove(file_name: str, file_list: list[str] | set[str]) -> list[str]:
        file_set = set(file_list)
        if file_name in file_set:
            file_set.remove(file_name)
        else:
            log.critical(
                f"The file {file_name} should exist in this directory - But was not found!"
            )
        return list(file_set)

    @staticmethod
    def __safe_append(file_name: str, file_list: list[str] | set[str]) -> list[str]:
        file_set = set(file_list)
        if os.path.exists(file_name):
            file_set.add(file_name)
        else:
            log.critical(f"Missing required file: {file_name}")
        return list(file_set)

    def __generate_execution_list(self) -> list[str]:
        """
        Generate an execution list of scripts based on the specified action.

        This function dynamically creates a list of scripts to be executed by filtering and selecting
        scripts based on the global ACTION variable. It supports different execution modes:
        - 'minimal': A predefined set of lightweight scripts
        - 'nopy': PowerShell and script-based scripts without Python
        - 'modded': Includes scripts from the MODS directory
        - 'depth': Comprehensive script execution with data mining and logging scripts
        - 'vulnscan_ai': Vulnerability scanning script only

        Returns:
            list[str]: A list of script file paths to be executed, filtered and modified based on the current action.

        Raises:
            ValueError: Implicitly if a script file cannot be removed from the initial list.

        Notes:
            - Removes sensitive or unnecessary scripts from the initial file list
            - Logs the final execution list for debugging purposes
            - Warns users about potential long execution times for certain actions
        """
        execution_list = get.list_of_files(
            ".",
            only_extensions=(".py", ".exe", ".ps1", ".bat"),
            exclude_files=["Logicytics.py"],
            exclude_dirs=["logicytics", "SysInternal_Suite"],
        )
        files_to_remove = {
            "sensitive_data_miner.py",
            "dir_list.py",
            "tree.ps1",
            "vulnscan.py",
            "event_log.py",
        }
        execution_list = [
            file for file in execution_list if file not in files_to_remove
        ]

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

        elif ACTION == "nopy":
            execution_list = [
                "browser_miner.ps1",
                "netadapter.ps1",
                "property_scraper.ps1",
                "window_feature_miner.ps1",
                "tree.ps1",
            ]

        elif ACTION == "modded":
            # Add all files in MODS to execution list
            execution_list = get.list_of_files(
                "../MODS",
                only_extensions=(".py", ".exe", ".ps1", ".bat"),
                append_file_list=execution_list,
                exclude_files=["Logicytics.py"],
                exclude_dirs=["logicytics", "SysInternal_Suite"],
            )

        elif ACTION == "depth":
            log.warning(
                "This flag will use clunky and huge scripts, and so may take a long time, but reap great rewards."
            )
            files_to_append = {
                "sensitive_data_miner.py",
                "dir_list.py",
                "tree.ps1",
                "event_log.py",
            }
            for file in files_to_append:
                execution_list = self.__safe_append(file, execution_list)
            log.warning("This flag will use threading!")

        elif ACTION == "vulnscan_ai":
            # Only vulnscan detector
            if os.path.exists("vulnscan.py"):
                execution_list = ["vulnscan.py"]
            else:
                log.critical("Vulnscan is missing...")
                exit(1)

        if len(execution_list) == 0:
            log.critical(
                "Nothing is in the execution list.. This is due to faulty code or corrupted Logicytics files!"
            )
            exit(1)

        log.debug(f"Execution list length: {len(execution_list)}")
        log.debug(f"The following will be executed: {execution_list}")
        return execution_list

    @staticmethod
    def __script_handler(script: str) -> tuple[str, Exception | None]:
        """
        Executes a single script and logs the result, capturing any exceptions that occur during execution.

        Parameters:
            script (str): The path to the script to be executed
        """
        log.debug(f"Executing {script}")
        try:
            log.execution(execute.script(script))
            log.info(f"{script} executed successfully")
            return script, None
        except Exception as err:
            log.error(f"Error executing {script}: {err}")
            return script, err

    def handler(self):
        """Executes the scripts in the execution list based on the action."""
        log.info("Starting Logicytics...")

        if ACTION == "threaded" or ACTION == "depth":
            self.__threaded()
        elif ACTION == "performance_check":
            self.__performance()
        else:
            self.__default()

    def __threaded(self):
        """Executes scripts in parallel using threading."""
        log.debug("Using threading")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.__script_handler, script): script
                for script in self.execution_list
            }

            for future in as_completed(futures):
                script = futures[future]
                try:
                    result, error = future.result()
                    if error:
                        log.error(f"Failed to execute {script}: {error}")
                    else:
                        log.debug(f"Completed {script}")
                except Exception as e:
                    log.error(f"Thread crashed while executing {script}: {e}")

    def __default(self):
        """Executes scripts sequentially."""
        try:
            for script in self.execution_list:
                result, error = self.__script_handler(script)
                if error:
                    log.error(f"Failed to execute {script}")
                else:
                    log.debug(f"Completed {script}")
        except UnicodeDecodeError as e:
            log.error(f"Error in script execution (Unicode): {e}")
        except Exception as e:
            log.error(f"Error in script execution: {e}")

    def __performance(self):
        """Checks performance of each script."""
        if DEBUG.lower() != "debug":
            log.warning("Advised to turn on DEBUG logging!!")

        execution_times = []

        for file in range(len(self.execution_list)):
            gc.collect()
            start_time = datetime.now()
            log.execution(execute.script(self.execution_list[file]))
            end_time = datetime.now()
            elapsed_time = end_time - start_time
            execution_times.append((self.execution_list[file], elapsed_time))
            log.info(f"{self.execution_list[file]} executed in {elapsed_time}")

        table = PrettyTable()
        table.field_names = ["Script", "Execution Time"]
        for script, elapsed_time in execution_times:
            table.add_row([script, elapsed_time])

        try:
            with open(
                f"../ACCESS/LOGS/PERFORMANCE/Performance_Summary_"
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
                "w",
            ) as f:
                f.write(table.get_string())
                f.write("\nNote: This test only measures execution time.\n")
            log.info(
                "Performance check complete! Performance log found in ACCESS/LOGS/PERFORMANCE"
            )
        except Exception as e:
            log.error(f"Error writing performance log: {e}")


class SpecialAction:
    @staticmethod
    def update() -> tuple[str, str]:
        """
        Updates the repository by pulling the latest changes from the remote repository.

        This function navigates to the parent directory, pulls the latest changes using Git,
        and then returns to the current working directory.

        Returns:
            str: The output from the git pull command.
        """
        # Check if git command is available
        try:
            if (
                subprocess.run(["git", "--version"], capture_output=True).returncode
                != 0
            ):
                return "Git is not installed or not available in the PATH.", "error"
        except FileNotFoundError:
            return "Git is not installed or not available in the PATH.", "error"

        # Check if the project is a git repository
        try:
            if not os.path.exists(os.path.join(os.getcwd(), "../.git")):
                return (
                    "This project is not a git repository. The update flag uses git.",
                    "error",
                )
        except Exception as e:
            return f"Error checking for git repository: {e}", "error"

        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        os.chdir(parent_dir)
        output = subprocess.run(["git", "pull"], capture_output=True).stdout.decode()
        os.chdir(current_dir)
        return output, "info"

    @staticmethod
    def execute_new_window(file_path: str):
        """
        Execute a Python script in a new command prompt window.

        This function launches the specified Python script in a separate command prompt window, waits for its completion, and then exits the current process.

        Parameters:
            file_path (str): The relative path to the Python script to be executed,
                             which will be resolved relative to the current script's directory.

        Side Effects:
            - Opens a new command prompt window
            - Runs the specified Python script
            - Terminates the current process after script execution

        Raises:
            FileNotFoundError: If the specified script path does not exist
            subprocess.SubprocessError: If there are issues launching the subprocess
        """
        sr_current_dir = os.path.dirname(os.path.abspath(__file__))
        sr_script_path = os.path.join(sr_current_dir, file_path)
        sr_process = subprocess.Popen(
            ["cmd.exe", "/c", "start", sys.executable, sr_script_path]
        )
        sr_process.wait()
        exit(0)


def get_flags():
    """
    Retrieves action and sub-action flags from the Flag module and sets global variables.

    This function extracts the current action and sub-action from the Flag module, setting global
    ACTION and SUB_ACTION variables. It logs the retrieved values for debugging and tracing purposes.

    No parameters.

    Side effects:
        - Sets global variables ACTION and SUB_ACTION
        - Logs debug information about current action and sub-action
    """
    global ACTION, SUB_ACTION
    # Get flags_list
    ACTION, SUB_ACTION = flag.data()
    log.debug(f"Action: {ACTION}")
    log.debug(f"Sub-Action: {SUB_ACTION}")


def handle_special_actions():
    """
    Handles special actions based on the current action flag.

    This function performs specific actions depending on the global `ACTION` variable:
    - For "debug": Opens the debug menu by executing '_debug.py'
    - For "dev": Opens the developer menu by executing '_dev.py'
    - For "update": Updates the repository using Health.update() method
    - For "restore": Displays a warning and opens the backup location
    - For "backup": Creates backups of the CODE and MODS directories

    Side Effects:
        - Logs informational, debug, warning, or error messages
        - May execute external Python scripts
        - May open file locations
        - May terminate the program after completing special actions

    Raises:
        SystemExit: Exits the program after completing certain special actions
    """
    # Special actions -> Quit
    if ACTION == "debug":
        log.info("Opening debug menu...")
        SpecialAction.execute_new_window("_debug.py")

    messages = check.sys_internal_zip()
    if messages:
        # If there are messages, log them with debug
        log.debug(messages)

    if ACTION == "dev":
        log.info("Opening developer menu...")
        SpecialAction.execute_new_window("_dev.py")

    if ACTION == "update":
        log.info("Updating...")
        message, log_type = SpecialAction.update()
        log.string(message, log_type)
        if log_type == "info":
            log.info("Update complete!")
        else:
            log.error("Update failed!")
        input("Press Enter to exit...")
        exit(0)

    if ACTION == "usage":
        flag.Match.generate_summary_and_graph()
        input("Press Enter to exit...")
        exit(1)


def check_privileges():
    """
    Checks if the script is running with administrative privileges and handles UAC (User Account Control) settings.

    This function verifies if the script has admin privileges. If not, it either logs a warning (in debug mode) or
    prompts the user to run the script with admin privileges and exits. It also checks if UAC is enabled and logs
    warnings accordingly.

    Raises:
        SystemExit: If the script is not running with admin privileges and not in debug mode.

    Notes:
        - Requires the `Check` module with `admin()` and `uac()` methods
        - Depends on global `DEBUG` configuration variable
        - Logs warnings or critical messages based on privilege and UAC status
    """
    if not check.admin():
        if DEBUG == "DEBUG":
            log.warning(
                "Running in debug mode, continuing without admin privileges - This may cause issues"
            )
        else:
            log.critical(
                "Please run this script with admin privileges - To ignore this message, run with DEBUG in config"
            )
            input("Press Enter to exit...")
            exit(1)

    if check.uac():
        log.warning(
            "UAC is enabled, this may cause issues - Please disable UAC if possible"
        )


class ZIP:
    @classmethod
    def files(cls):
        """Zips generated files based on the action."""
        if ACTION == "modded":
            cls.__and_log("..\\MODS", "MODS")
        cls.__and_log(".", "CODE")

    @staticmethod
    def __and_log(directory: str, name: str):
        log.debug(
            f"Zipping directory '{directory}' with name '{name}' under action '{ACTION}'"
        )
        zip_values = file_management.Zip.and_hash(
            directory,
            name,
            ACTION
            if ACTION is not None
            else f"ERROR_NO_ACTION_SPECIFIED_{datetime.now().isoformat()}",
        )
        if isinstance(zip_values, str):
            log.error(zip_values)
        else:
            zip_loc, hash_loc = zip_values
            log.info(zip_loc)
            log.debug(hash_loc)


def handle_sub_action():
    """
    Handles sub-actions based on the provided sub_action flag.

    This function checks the value of the `sub_action` variable and performs
    corresponding sub-actions such as shutting down or rebooting the system.
    """
    log.info("Completed successfully!")
    log.newline()
    # Handle sub actions for all actions except performance check
    if ACTION != "performance_check":
        if SUB_ACTION == "shutdown":
            subprocess.call("shutdown /s /t 3", shell=False)
        elif SUB_ACTION == "reboot":
            subprocess.call("shutdown /r /t 3", shell=False)


@log.function
def Logicytics():
    """
    Orchestrates the complete Logicytics workflow, managing script execution, system actions, and user interactions.

    This function serves as the primary entry point for the Logicytics utility, coordinating a series of system-level operations:
    - Retrieves command-line configuration flags
    - Processes special actions
    - Verifies system privileges
    - Executes targeted scripts
    - Compresses generated output files
    - Handles final system sub-actions
    - Provides a graceful exit mechanism

    Performs actions sequentially without returning a value, designed to be the main execution flow of the Logicytics utility.
    """
    # Get flags_list and configs
    get_flags()
    # Check for special actions
    handle_special_actions()
    # Check for privileges and errors
    check_privileges()
    # Execute scripts
    ExecuteScript().handler()
    # Zip generated files
    ZIP.files()
    # Finish with sub actions
    handle_sub_action()
    # Finish
    input("Press Enter to exit...")


if __name__ == "__main__":
    try:
        Logicytics()
    except KeyboardInterrupt:
        log.warning(
            "Force shutdown detected! Some temporary files might be left behind."
        )
        log.warning("Next time, let the program finish naturally for complete cleanup.")
        # Emergency cleanup - zip generated files
        ZIP.files()
        exit(0)
else:
    log.error("This script cannot be imported!")
    exit(1)
