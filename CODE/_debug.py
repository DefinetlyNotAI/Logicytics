from __future__ import annotations

import configparser
import os.path
import platform
import subprocess
import sys
from datetime import datetime

import psutil
import requests

from logicytics import Log, DEBUG, VERSION, Check

if __name__ == "__main__":
    log_debug = Log({"log_level": DEBUG, "filename": "../ACCESS/LOGS/DEBUG/DEBUG.log", "truncate_message": False})


class HealthCheck:
    @log_debug.function
    def get_online_config(self) -> bool | tuple[str, str, str]:
        """
        Retrieves configuration data from a remote repository and compares it with the local configuration.

        Returns:
            bool: False if a connection error occurs, otherwise a tuple containing version check and file check results.
            tuple[tuple[str, str, str], tuple[str, str, str]]: A tuple containing version check and file check results.
        """
        try:
            url = "https://raw.githubusercontent.com/DefinetlyNotAI/Logicytics/main/CODE/config.ini"
            config = configparser.ConfigParser()
            config.read_string(requests.get(url, timeout=15).text)
        except requests.exceptions.ConnectionError:
            log_debug.warning("No connection found")
            return False
        version_check = self.__compare_versions(VERSION, config["System Settings"]["version"])

        return version_check

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


class DebugCheck:
    @staticmethod
    @log_debug.function
    def sys_internal_binaries(path: str) -> tuple[str, str]:
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
            log_debug.debug(str(contents))
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
    @log_debug.function
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
    @log_debug.function
    def cpu_info() -> tuple[str, str, str]:
        """
        Retrieves information about the CPU.

        Returns:
            tuple[str, str, str]: A tuple containing the CPU architecture, vendor ID, and model.
        """
        return (
            "CPU Architecture: " + platform.machine(),
            "CPU Vendor ID: " + platform.system(),
            "CPU Model: " + f"{platform.release()} {platform.version()}",
        )


@log_debug.function
def debug():
    """
    Performs a series of system checks and logs the results.
    """
    # Clear Debug Log
    log_path = "../ACCESS/LOGS/DEBUG/DEBUG.LOG"
    if os.path.exists(log_path):
        os.remove(log_path)

    # Check File integrity (Online)
    online_config = HealthCheck().get_online_config()
    if online_config:
        version_tuple = online_config
        log_debug.string(version_tuple[0], version_tuple[2])
        log_debug.raw(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] > DATA:     | {version_tuple[1] + ' ' * (153 - len(version_tuple[1])) + '|'}")

    # Check SysInternal Binaries
    message, type = DebugCheck.sys_internal_binaries("SysInternal_Suite")
    log_debug.string(message, type)

    # Check Admin
    log_debug.info("Admin privileges found" if Check.admin() else "Admin privileges not found")

    # Check UAC
    log_debug.info("UAC enabled" if Check.uac() else "UAC disabled")

    # Log Execution Paths
    log_debug.info(f"Execution path: {psutil.__file__}")
    log_debug.info(f"Global execution path: {sys.executable}")
    log_debug.info(f"Local execution path: {sys.prefix}")

    # Check if running in a virtual environment
    log_debug.info("Running in a virtual environment" if sys.prefix != sys.base_prefix else "Not running in a virtual environment")

    # Check Execution Policy
    log_debug.info("Execution policy is unrestricted" if DebugCheck.execution_policy() else "Execution policy is not unrestricted")

    # Get Python Version
    try:
        major, minor = map(int, sys.version.split()[0].split(".")[:2])
        if major == 3 and minor == 11:
            log_debug.info(f"Python Version Used: {sys.version.split()[0]} - Perfect")
        elif major == 3:
            log_debug.warning(f"Python Version Used: {sys.version.split()[0]} - Recommended Version is: 3.11.X")
        else:
            log_debug.error(f"Python Version Used: {sys.version.split()[0]} - Incompatible Version")
    except Exception as e:
        log_debug.error(f"Failed to get Python Version: {e}")

    # Get Repo Path
    log_debug.info(os.path.abspath(__file__).removesuffix("\\CODE\\_debug.py"))

    # Get CPU Info
    architecture, vID, cpuModel = DebugCheck.cpu_info()
    log_debug.info(architecture)
    log_debug.info(vID)
    log_debug.info(cpuModel)

    # Get config data
    log_debug.info(f"Debug: {DEBUG}")


debug()
input("Press Enter to exit...")
exit(0)
