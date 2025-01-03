from __future__ import annotations

import configparser
import os
import platform
import sys

import psutil
import requests

from logicytics import Log, DEBUG, VERSION, Check

if __name__ == "__main__":
    log_debug = Log(
        {"log_level": DEBUG,
         "filename": "../ACCESS/LOGS/DEBUG/DEBUG.log",
         "truncate_message": False}
    )


class HealthCheck:
    @staticmethod
    def __version_tuple(version: str) -> tuple[int, int, int | str, str]:
        """
        Parses a version string into a tuple.

        Args:
            version (str): The version string to parse.

        Returns:
            tuple[int, int, int, str]: A tuple containing the major, minor, and patch versions,
                                       and a string indicating whether it is a snapshot or release version.
        """
        try:
            if version.startswith("snapshot-"):
                parts = version.split('-')[1].split('.')
                major, minor = map(int, parts[:2])
                patch = parts[2] if len(parts) > 2 else "0"
                return major, minor, patch, "snapshot"
            else:
                return tuple(map(int, version.split('.'))) + ("release",)
        except Exception as err:
            log_debug.error(f"Failed to parse version: {err}")
            return 0, 0, 0, "error"

    @staticmethod
    def files(directory: str, required_files: list[str]):
        """
        Checks if all required files are present in the directory and its subdirectories.

        Args:
            directory (str): Path to the directory to check.
            required_files (list[str]): List of required file names with relative paths.
        """
        try:
            log_debug.debug(f"Checking directory: {directory}")
            if not os.path.exists(directory):
                log_debug.error(f"Directory {directory} does not exist.")

            # Gather all files with relative paths
            actual_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    relative_path = os.path.relpath(os.path.join(root, file), start=directory)
                    actual_files.append(
                        relative_path.replace("\\", "/").replace('"', ''))  # Normalize paths for comparison

            log_debug.debug(f"Actual files found: {actual_files}")

            # Track missing and extra files
            missing_files = []
            extra_files = []

            # Normalize required files
            normalized_required_files = [required_file.strip().replace("\\", "/").replace('"', '') for required_file in
                                         required_files]

            # Check for missing files
            for required_file in normalized_required_files:
                if required_file not in actual_files:
                    missing_files.append(required_file)

            log_debug.debug(f"Missing files: {missing_files}")

            # Check for extra files
            for actual_file in actual_files:
                if actual_file not in normalized_required_files:
                    extra_files.append(actual_file)

            log_debug.debug(f"Extra files: {extra_files}")

            if missing_files:
                log_debug.error(f"Missing files: {', '.join(missing_files)}")
            if extra_files:
                log_debug.warning(f"Extra files found: {', '.join(extra_files)}")
            log_debug.info("All required files are present.")

        except Exception as e:
            log_debug.error(f"Unexpected error during file check: {e}")

    @classmethod
    def versions(cls, local_version: str, remote_version: str):
        """
        Compares local and remote versions.

        Args:
            local_version (str): Local version.
            remote_version (str): Remote version.
        """

        local_version_tuple = cls.__version_tuple(local_version)
        remote_version_tuple = cls.__version_tuple(remote_version)

        if "error" in local_version_tuple or "error" in remote_version_tuple:
            log_debug.error("Version parsing error.")
            return

        try:
            if "snapshot" in local_version_tuple or "snapshot" in remote_version_tuple:
                log_debug.warning("Snapshot versions are unstable.")

            if local_version_tuple == remote_version_tuple:
                log_debug.info(f"Version is up to date. Your Version: {local_version}")
            elif local_version_tuple > remote_version_tuple:
                log_debug.warning("Version is ahead of the repository. "
                                  f"Your Version: {local_version}, "
                                  f"Repository Version: {remote_version}")
            else:
                log_debug.error("Version is behind the repository. "
                                f"Your Version: {local_version}, Repository Version: {remote_version}")
        except Exception as e:
            log_debug.error(f"Version comparison error: {e}")


class DebugCheck:
    @staticmethod
    def sys_internal_binaries(path: str):
        """
        Checks the SysInternal Binaries in the given directory.

        Args:
            path (str): Directory path.
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError("Directory does not exist")

            contents = os.listdir(path)
            log_debug.debug(str(contents))

            has_zip = any(file.endswith(".zip") for file in contents)
            has_exe = any(file.endswith(".exe") for file in contents)

            if any(file.endswith(".ignore") for file in contents):
                log_debug.warning("A `.sys.ignore` file was found - Ignoring")
            elif has_zip and not has_exe:
                log_debug.error("Only zip files - Missing EXEs due to no `ignore` file")
            elif has_zip and has_exe:
                log_debug.info("Both zip and exe files - All good")
            else:
                log_debug.error("SysInternal Binaries Not Found: Missing Files - Corruption detected")
        except Exception as e:
            log_debug.error(f"Unexpected error: {e}")

    @staticmethod
    def cpu_info() -> tuple[str, str, str]:
        """
        Retrieves CPU details.

        Returns:
            tuple[str, str, str]: Architecture, vendor ID, and model.
        """
        return (
            f"CPU Architecture: {platform.machine()}",
            f"CPU Vendor ID: {platform.system()}",
            f"CPU Model: {platform.release()} {platform.version()}",
        )


def python_version():
    """
    Checks the current Python version against recommended version ranges and logs the result.
    
    This function determines the compatibility of the current Python runtime by comparing its version
    against predefined minimum and maximum version thresholds. It provides informative logging about
    the Python version status.
    
    Parameters:
        None
    
    Logs:
        - Info: When Python version is within the recommended range (3.11.x to 3.12.x)
        - Warning: When Python version is below the minimum recommended version (< 3.11)
        - Error: When Python version is above the maximum supported version (>= 3.13) or parsing fails
    
    Raises:
        No explicit exceptions are raised; errors are logged internally
    
    Example:
        Typical log outputs might include:
        - "Python Version: 3.11.5 - Perfect"
        - "Python Version: 3.10.2 - Recommended: 3.11.x"
        - "Python Version: 3.13.0 - Incompatible"
    """
    version = sys.version.split()[0]
    MIN_VERSION = (3, 11)
    MAX_VERSION = (3, 13)
    try:
        major, minor = map(int, version.split(".")[:2])
        if MIN_VERSION <= (major, minor) < MAX_VERSION:
            log_debug.info(f"Python Version: {version} - Perfect")
        elif (major, minor) < MIN_VERSION:
            log_debug.warning(f"Python Version: {version} - Recommended: 3.11.x")
        else:
            log_debug.error(f"Python Version: {version} - Incompatible")
    except Exception as e:
        log_debug.error(f"Failed to parse Python Version: {e}")


def get_online_config() -> dict | None:
    """
        Retrieves configuration data from a remote repository.

        Returns:
            dict: Parsed configuration data if successful.
            None: If there was an error fetching the configuration.
        """
    try:
        url = "https://raw.githubusercontent.com/DefinetlyNotAI/Logicytics/main/CODE/config.ini"
        config = configparser.ConfigParser()
        config.read_string(requests.get(url, timeout=15).text)
        return config
    except requests.exceptions.RequestException as e:
        log_debug.error(f"Connection error: {e}")
        return None


@log_debug.function
def debug():
    """
    Executes a comprehensive system debug routine, performing various checks and logging system information.
    
    This function performs the following tasks:
    - Clears the existing debug log file
    - Retrieves and validates online configuration
    - Checks system version compatibility
    - Verifies required file integrity
    - Checks SysInternal binaries
    - Logs system privileges and environment details
    - Checks Python version compatibility
    - Retrieves and logs CPU information
    
    Logs are written to the debug log file, capturing system state, configuration, and potential issues.
    
    Notes:
    - Requires admin privileges for full system checks
    - Logs information about execution environment
    - Checks system and Python version compatibility
    - Provides insights into system configuration and potential security settings
    """
    # Clear Debug Log
    log_path = "../ACCESS/LOGS/DEBUG/DEBUG.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    # Online Configuration Check
    config = get_online_config()
    if config:
        HealthCheck.versions(VERSION, config["System Settings"]["version"])

        # File Integrity Check
        required_files = config["System Settings"].get("files", "").split(",")
        HealthCheck.files(".", required_files)

    # SysInternal Binaries Check
    DebugCheck.sys_internal_binaries("SysInternal_Suite")

    # System Checks
    log_debug.info("Admin privileges found" if Check.admin() else "Admin privileges not found")
    log_debug.info("UAC enabled" if Check.uac() else "UAC disabled")
    log_debug.info(f"Execution path: {psutil.__file__}")
    log_debug.info(f"Global execution path: {sys.executable}")
    log_debug.info(f"Local execution path: {sys.prefix}")
    log_debug.info(
        "Running in a virtual environment" if sys.prefix != sys.base_prefix else "Not running in a virtual environment")
    log_debug.info(
        "Execution policy is unrestricted" if Check.execution_policy() else "Execution policy is not unrestricted")

    # Python Version Check
    python_version()

    # CPU Info
    for info in DebugCheck.cpu_info():
        log_debug.info(info)

    # Final Debug Status
    log_debug.info(f"Debug: {DEBUG}")


if __name__ == "__main__":
    debug()
    input("Press Enter to exit...")
