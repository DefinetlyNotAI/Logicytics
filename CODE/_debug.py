from __future__ import annotations

import configparser
import os
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
    @staticmethod
    def check_files(directory: str, required_files: list[str]) -> tuple[str, str]:
        """
        Checks if all required files are present in the directory and its subdirectories.

        Args:
            directory (str): Path to the directory to check.
            required_files (list[str]): List of required file names with relative paths.

        Returns:
            tuple[str, str]: Status message and severity level.
        """
        try:
            log_debug.debug(f"Checking directory: {directory}")
            if not os.path.exists(directory):
                log_debug.error(f"Directory {directory} does not exist.")
                return f"Directory {directory} does not exist.", "ERROR"

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
                return f"Missing files: {', '.join(missing_files)}", "ERROR"
            if extra_files:
                return f"Extra files found: {', '.join(extra_files)}", "WARNING"
            return "All required files are present.", "INFO"

        except Exception as e:
            log_debug.error(f"Unexpected error during file check: {e}")
            return f"Unexpected error during file check: {e}", "ERROR"

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
            log_debug.warning(f"Connection error: {e}")
            return None

    @staticmethod
    def compare_versions(local_version: str, remote_version: str) -> tuple[str, str, str]:
        """
        Compares local and remote versions.

        Args:
            local_version (str): Local version.
            remote_version (str): Remote version.

        Returns:
            tuple[str, str, str]: Comparison result, version details, and severity level.
        """
        if local_version == remote_version:
            return "Version is up to date.", f"Your Version: {local_version}", "INFO"
        if local_version > remote_version:
            return (
                "Version is ahead of the repository.",
                f"Your Version: {local_version}, Repository Version: {remote_version}",
                "WARNING",
            )
        return (
            "Version is behind the repository.",
            f"Your Version: {local_version}, Repository Version: {remote_version}",
            "ERROR",
        )


class DebugCheck:
    @staticmethod
    def sys_internal_binaries(path: str) -> tuple[str, str]:
        """
        Checks the SysInternal Binaries in the given directory.

        Args:
            path (str): Directory path.

        Returns:
            tuple[str, str]: Status message and severity level.
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError("Directory does not exist")

            contents = os.listdir(path)
            log_debug.debug(str(contents))

            has_zip = any(file.endswith(".zip") for file in contents)
            has_exe = any(file.endswith(".exe") for file in contents)

            if any(file.endswith(".ignore") for file in contents):
                return "A `.sys.ignore` file was found - Ignoring", "WARNING"
            if has_zip and not has_exe:
                return "Only zip files - Missing EXEs due to no `ignore` file", "ERROR"
            if has_zip and has_exe:
                return "Both zip and exe files - All good", "INFO"

            return "SysInternal Binaries Not Found: Missing Files - Corruption detected", "ERROR"
        except Exception as e:
            return f"Unexpected error: {e}", "ERROR"

    @staticmethod
    def execution_policy() -> bool:
        """
        Checks if the execution policy is unrestricted.

        Returns:
            bool: True if unrestricted, False otherwise.
        """
        try:
            result = subprocess.run(
                ["powershell", "-Command", "Get-ExecutionPolicy"],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip().lower() == "unrestricted"
        except Exception as e:
            log_debug.error(f"Failed to check execution policy: {e}")
            return False

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
        version_check = HealthCheck.compare_versions(VERSION, config["System Settings"]["version"])
        log_debug.string(version_check[0], version_check[2])
        log_debug.raw(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] > DATA: {version_check[1]}")

        # File Integrity Check
        required_files = config["System Settings"].get("files", "").split(",")
        message, severity = HealthCheck.check_files(".", required_files)
        log_debug.string(message, severity)

    # SysInternal Binaries Check
    message, severity = DebugCheck.sys_internal_binaries("SysInternal_Suite")
    log_debug.string(message, severity)

    # System Checks
    log_debug.info("Admin privileges found" if Check.admin() else "Admin privileges not found")
    log_debug.info("UAC enabled" if Check.uac() else "UAC disabled")
    log_debug.info(f"Execution path: {psutil.__file__}")
    log_debug.info(f"Global execution path: {sys.executable}")
    log_debug.info(f"Local execution path: {sys.prefix}")
    log_debug.info(
        "Running in a virtual environment" if sys.prefix != sys.base_prefix else "Not running in a virtual environment")

    # Execution Policy Check
    log_debug.info(
        "Execution policy is unrestricted" if DebugCheck.execution_policy() else "Execution policy is not unrestricted")

    # Python Version Check
    python_version = sys.version.split()[0]
    try:
        major, minor = map(int, python_version.split(".")[:2])
        if (major, minor) == (3, 11):
            log_debug.info(f"Python Version: {python_version} - Perfect")
        elif major == 3:
            log_debug.warning(f"Python Version: {python_version} - Recommended: 3.11.x")
        else:
            log_debug.error(f"Python Version: {python_version} - Incompatible")
    except Exception as e:
        log_debug.error(f"Failed to parse Python Version: {e}")

    # CPU Info
    for info in DebugCheck.cpu_info():
        log_debug.info(info)

    # Final Debug Status
    log_debug.info(f"Debug: {DEBUG}")


if __name__ == "__main__":
    debug()
    input("Press Enter to exit...")
