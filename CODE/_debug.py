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
    def check_files(directory: str, required_files: list[str]):
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

    @staticmethod
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

    @staticmethod
    def compare_versions(local_version: str, remote_version: str):
        """
        Compares local and remote versions.

        Args:
            local_version (str): Local version.
            remote_version (str): Remote version.
        """
        if local_version == remote_version:
            log_debug.info(f"Version is up to date. Your Version: {local_version}")
        elif local_version > remote_version:
            log_debug.warning("Version is ahead of the repository. "
                              f"Your Version: {local_version}, "
                              f"Repository Version: {remote_version}")
        else:
            log_debug.error("Version is behind the repository."
                            f"Your Version: {local_version}, Repository Version: {remote_version}"
                            )


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
    version = sys.version.split()[0]
    try:
        major, minor = map(int, version.split(".")[:2])
        if (major, minor) == (3, 11):
            log_debug.info(f"Python Version: {version} - Perfect")
        elif major == 3:
            log_debug.warning(f"Python Version: {version} - Recommended: 3.11.x")
        else:
            log_debug.error(f"Python Version: {version} - Incompatible")
    except Exception as e:
        log_debug.error(f"Failed to parse Python Version: {e}")


def debug():
    """
    Executes system checks and logs results.
    """
    # Clear Debug Log
    log_path = "../ACCESS/LOGS/DEBUG/DEBUG.LOG"
    if os.path.exists(log_path):
        os.remove(log_path)

    # Online Configuration Check
    config = HealthCheck.get_online_config()
    if config:
        HealthCheck.compare_versions(VERSION, config["System Settings"]["version"])

        # File Integrity Check
        required_files = config["System Settings"].get("files", "").split(",")
        HealthCheck.check_files(".", required_files)

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
