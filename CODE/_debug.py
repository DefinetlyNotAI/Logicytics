from __future__ import annotations
import platform
import os.path
import requests
import psutil
import sys
from __lib_class import *

if __name__ == "__main__":
    log_debug = Log(debug=DEBUG, filename="../ACCESS/LOGS/DEBUG/DEBUG.LOG")
    log_debug_funcs = {
        "INFO": log_debug.info,
        "WARNING": log_debug.warning,
        "ERROR": log_debug.error,
        "CRITICAL": log_debug.critical,
        None: log_debug.debug,
    }


class HealthCheck:
    def get_online_config(
        self,
    ) -> bool | tuple[tuple[str, str, str], tuple[str, str, str]]:
        """
        Retrieves configuration data from a remote repository and compares it with the local configuration.

        Returns:
            bool: False if a connection error occurs, otherwise a tuple containing version check and file check results.
            tuple[tuple[str, str, str], tuple[str, str, str]]: A tuple containing version check and file check results.
        """
        try:
            url = "https://raw.githubusercontent.com/DefinetlyNotAI/Logicytics/main/CODE/config.json"
            config = json.loads(requests.get(url).text)
        except requests.exceptions.ConnectionError:
            log_debug.warning("No connection found")
            return False
        version_check = self.__compare_versions(VERSION, config["VERSION"])
        file_check = self.__check_files(CURRENT_FILES, config["CURRENT_FILES"])

        return version_check, file_check

    @staticmethod
    def __compare_versions(
        local_version: str, remote_version: str
    ) -> tuple[str, str, str]:
        """
        Compares the local version with the remote version and returns a tuple containing a comparison result message,
        a version information message, and a severity level.

        Args:
            local_version (str): The version number of the local system.
            remote_version (str): The version number of the remote repository.

        Returns:
            tuple[str, str, str]: A tuple containing a comparison result message, a version information message, and a severity level.
        """
        if local_version == remote_version:
            return "Version is up to date.", f"Your Version: {local_version}", "INFO"
        elif local_version > remote_version:
            return (
                "Version is ahead of the repository.",
                f"Your Version: {local_version}, Repository Version: {remote_version}",
                "WARNING",
            )
        else:
            return (
                "Version is behind the repository.",
                f"Your Version: {local_version}, Repository Version: {remote_version}",
                "ERROR",
            )

    @staticmethod
    def __check_files(local_files: list, remote_files: list) -> tuple[str, str, str]:
        """
        Check if all the files in the local_files list are present in the remote_files list.

        Args:
            local_files (list): A list of files in the local repository.
            remote_files (list): A list of files in the remote repository.

        Returns:
            tuple[str, str, str]: A tuple containing the result message, a message detailing the files present or missing,
                                 and the log level.
        """
        missing_files = set(remote_files) - set(local_files)
        if not missing_files:
            return (
                "All files are present.",
                f"Your files: {local_files} contain all the files in the repository.",
                "INFO",
            )
        else:
            return (
                "You have missing files.",
                f"You are missing the following files: {missing_files}",
                "ERROR",
            )


class DebugCheck:
    @staticmethod
    def SysInternal_Binaries(path: str) -> tuple[str, str]:
        """
        Checks the contents of the given path and determines the status of the SysInternal Binaries.

        Args:
            path (str): The path to the directory containing the SysInternal Binaries.

        Returns:
            tuple[str, str]: A tuple containing a status message and a severity level.
                The status message indicates the result of the check.
                The severity level is either "INFO", "WARNING", or "ERROR".

        Raises:
            FileNotFoundError: If the given path does not exist.
            Exception: If an unexpected error occurs during the check.
        """
        try:
            contents = os.listdir(path)
            log_debug.debug(contents)
            if any(file.endswith(".ignore") for file in contents):
                return "A `.sys.ignore` file was found - Ignoring", "WARNING"
            if any(file.endswith(".zip") for file in contents) and not any(
                file.endswith(".exe") for file in contents
            ):
                return "Only zip files - Missing EXE's due to no `ignore` file", "ERROR"
            elif any(file.endswith(".zip") for file in contents) and any(
                file.endswith(".exe") for file in contents
            ):
                return "Both zip and exe files - All good", "INFO"
            else:
                return (
                    "SysInternal Binaries Not Found: Missing Files - Corruption detected",
                    "ERROR",
                )
        except FileNotFoundError:
            return (
                "SysInternal Binaries Not Found: Missing Directory- Corruption detected",
                "ERROR",
            )
        except Exception as e:
            return f"An Unexpected error occurred: {e}", "ERROR"

    @staticmethod
    def execution_policy() -> bool:
        """
        Checks the current PowerShell execution policy.

        Returns:
            bool: True if the execution policy is unrestricted, False otherwise.
        """
        result = subprocess.run(
            ["powershell", "-Command", "Get-ExecutionPolicy"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().lower() == "unrestricted"

    @staticmethod
    def cpu_info() -> tuple[str, str, str]:
        """
        Retrieves information about the CPU.

        Returns:
            tuple[str, str, str]: A tuple containing the CPU architecture, vendor ID, and model.
        """
        return (
            "CPU Architecture: " + platform.machine(),
            "CPU Vendor Id: " + platform.system(),
            "CPU Model: " + f"{platform.release()} {platform.version()}",
        )


def debug():
    """
    Performs a series of system checks and logs the results.

    This function performs the following checks:
    1. Clears the debug log file.
    2. Checks the integrity of files by comparing local and remote configurations.
    3. Checks the status of SysInternal Binaries.
    4. Checks for admin privileges.
    5. Checks if User Account Control (UAC) is enabled.
    6. Logs the execution paths.
    7. Checks if the script is running in a virtual environment.
    8. Checks the PowerShell execution policy.
    9. Logs the Python version being used.
    10. Logs the repository path.
    11. Logs CPU information.
    12. Logs the debug configuration.

    Returns:
        None
    """
    # Clear Debug Log
    if os.path.exists("../ACCESS/LOGS/DEBUG/DEBUG.LOG"):
        os.remove("../ACCESS/LOGS/DEBUG/DEBUG.LOG")

    # Check File integrity (Online)
    if HealthCheck().get_online_config():
        version_tuple, file_tuple = HealthCheck().get_online_config()
        log_debug_funcs.get(version_tuple[2], log_debug.debug)(
            "\n".join(version_tuple[0]).replace("\n", "")
        )
        log_debug_funcs.get(file_tuple[2], log_debug.debug)(
            "\n".join(file_tuple[0]).replace("\n", "")
        )
    message, type = DebugCheck.SysInternal_Binaries("SysInternal_Suite")
    log_debug_funcs.get(type, log_debug.debug)("\n".join(message).replace("\n", ""))

    # Check Admin
    if Check().admin():
        log_debug.info("Admin privileges found")
    else:
        log_debug.warning("Admin privileges not found")

    # Check UAC
    if Check().uac():
        log_debug.info("UAC enabled")
    else:
        log_debug.warning("UAC disabled")

    # Check Execution Path
    log_debug.info(f"Execution path: {psutil.__file__}")
    log_debug.info(f"Global execution path: {sys.executable}")
    log_debug.info(f"Local execution path: {sys.prefix}")

    # Check if running in a virtual environment
    if sys.prefix != sys.base_prefix:
        log_debug.info("Running in a virtual environment")
    else:
        log_debug.warning("Not running in a virtual environment")

    # Check Execution Policy
    if DebugCheck.execution_policy():
        log_debug.info("Execution policy is unrestricted")
    else:
        log_debug.warning("Execution policy is not unrestricted")

    # Get Python Version
    log_debug.info(
        f"Python Version Used: {sys.version.split()[0]} - Recommended Version is: ~"
    )

    # Get Repo Path
    log_debug.info(os.path.abspath(__file__).removesuffix("\\CODE\\_debug.py"))

    # Get CPU Info
    architecture, vID, cpuModel = DebugCheck.cpu_info()
    log_debug.info(architecture)
    log_debug.info(vID)
    log_debug.info(cpuModel)

    # Get config data
    log_debug.info("Debug: " + DEBUG)
