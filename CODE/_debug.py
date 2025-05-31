from __future__ import annotations

import configparser
import os
import platform
import sys
import time

import psutil
import requests

from logicytics import Log, DEBUG, VERSION, check, config

log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACCESS\\LOGS\\DEBUG\\DEBUG.log")
log = Log({"log_level": DEBUG, "filename": log_path, "truncate_message": False, "delete_log": True})
url = config.get("System Settings", "config_url")


class VersionManager:
    @staticmethod
    def parse_version(version: str) -> tuple[int, int, int | str, str]:
        """
        Parses a version string into a tuple (major, minor, patch, type).
        """
        try:
            if version.startswith("snapshot-"):
                parts = version.split('-')[1].split('.')
                major, minor = map(int, parts[:2])
                patch = parts[2] if len(parts) > 2 else "0"
                return major, minor, patch, "snapshot"
            else:
                return tuple(map(int, version.split('.'))) + ("release",)
        except Exception as e:
            log.error(f"Failed to parse version: {e}")
            return 0, 0, 0, "error"


class FileManager:
    @staticmethod
    def check_required_files(directory: str, required_files: list[str]):
        """
        Checks if all required files are present in the directory and its subdirectories.
        """
        try:
            log.debug(f"Checking directory: {directory}")
            if not os.path.exists(directory):
                log.error(f"Directory {directory} does not exist.")
                return

            actual_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    relative_path = os.path.relpath(os.path.join(root, file), start=directory)
                    actual_files.append(relative_path.replace("\\", "/").replace('"', ''))  # Normalize paths

            log.debug(f"Actual files found: {actual_files}")

            # Strip quotes and normalize paths for comparison
            normalized_required_files = [
                required_file.strip().replace("\\", "/").replace('"', '')  # Remove quotes and normalize paths
                for required_file in required_files
            ]

            # Compare files
            missing_files, extra_files = FileManager.compare_files(actual_files, normalized_required_files)

            if missing_files:
                log.error(f"Missing files: {', '.join(missing_files)}")
            if extra_files:
                log.warning(f"Extra files found: {', '.join(extra_files)}")
            if not missing_files and not extra_files:
                log.info("All required files are present.")
        except Exception as e:
            log.error(f"Unexpected error during file check: {e}")

    @staticmethod
    def compare_files(actual_files: list[str], required_files: list[str]) -> tuple[list[str], list[str]]:
        """
        Compares actual and required files, returning missing and extra files.
        """
        missing_files = [file for file in required_files if file not in actual_files]
        extra_files = [file for file in actual_files if file not in required_files]
        return missing_files, extra_files


class SysInternalManager:
    @staticmethod
    def check_binaries(path: str):
        """
        Checks the SysInternal Binaries in the given directory.
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError("Directory does not exist")

            contents = os.listdir(path)
            log.debug(str(contents))

            has_zip = any(file.endswith(".zip") for file in contents)
            has_exe = any(file.endswith(".exe") for file in contents)

            if any(file.endswith(".ignore") for file in contents):
                log.warning("A `.sys.ignore` file was found - Ignoring")
            elif has_zip and not has_exe:
                log.error("Only zip files - Missing EXEs due to no `ignore` file")
            elif has_zip and has_exe:
                log.info("Both zip and exe files - All good")
            else:
                log.error("SysInternal Binaries Not Found: Missing Files - Corruption detected")
        except Exception as e:
            log.error(f"Unexpected error: {e}")


class SystemInfoManager:
    @staticmethod
    def cpu_info() -> tuple[str, str, str]:
        """
        Retrieves CPU details.
        """
        return (
            f"CPU Architecture: {platform.machine()}",
            f"CPU Vendor ID: {platform.system()}",
            f"CPU Model: {platform.release()} {platform.version()}"
        )

    @staticmethod
    def python_version():
        """
        Checks the current Python version against recommended version ranges and logs the result.
        """
        version = sys.version.split()[0]
        MIN_VERSION = (3, 11)
        MAX_VERSION = (3, 13)
        try:
            major, minor = map(int, version.split(".")[:2])
            if MIN_VERSION <= (major, minor) < MAX_VERSION:
                log.info(f"Python Version: {version} - Perfect")
            elif (major, minor) < MIN_VERSION:
                log.warning(f"Python Version: {version} - Recommended: 3.11.x")
            else:
                log.error(f"Python Version: {version} - Incompatible")
        except Exception as e:
            log.error(f"Failed to parse Python Version: {e}")


class ConfigManager:
    @staticmethod
    def get_online_config() -> dict | None:
        """
        Retrieves configuration data from a remote repository.
        """
        try:
            _config = configparser.ConfigParser()
            _config.read_string(requests.get(url, timeout=15).text)
            return _config
        except requests.exceptions.RequestException as e:
            log.error(f"Connection error: {e}")
            return None


class HealthCheck:
    @staticmethod
    def check_versions(local_version: str, remote_version: str):
        """
        Compares local and remote versions.
        """
        local_version_tuple = VersionManager.parse_version(local_version)
        remote_version_tuple = VersionManager.parse_version(remote_version)

        if "error" in local_version_tuple or "error" in remote_version_tuple:
            log.error("Version parsing error.")
            return

        try:
            if "snapshot" in local_version_tuple or "snapshot" in remote_version_tuple:
                log.warning("Snapshot versions are unstable.")

            if local_version_tuple == remote_version_tuple:
                log.info(f"Version is up to date. Your Version: {local_version}")
            elif local_version_tuple > remote_version_tuple:
                log.warning("Version is ahead of the repository. "
                            f"Your Version: {local_version}, "
                            f"Repository Version: {remote_version}")
            else:
                log.error("Version is behind the repository. "
                          f"Your Version: {local_version}, Repository Version: {remote_version}")
        except Exception as e:
            log.error(f"Version comparison error: {e}")


@log.function
def debug():
    """
    Executes a comprehensive system debug routine, performing various checks and logging system information.
    """
    # Online Configuration Check
    _config = ConfigManager.get_online_config()
    if _config:
        HealthCheck.check_versions(VERSION, _config["System Settings"]["version"])

        # File Integrity Check
        required_files = _config["System Settings"].get("files", "").split(",")
        FileManager.check_required_files(".", required_files)

    # SysInternal Binaries Check
    SysInternalManager.check_binaries("SysInternal_Suite")

    # System Checks
    log.info("Admin privileges found" if check.admin() else "Admin privileges not found")
    log.info("UAC enabled" if check.uac() else "UAC disabled")
    log.info(f"Execution path: {psutil.__file__}")
    log.info(f"Global execution path: {sys.executable}")
    log.info(f"Local execution path: {sys.prefix}")
    log.info(
        "Running in a virtual environment" if sys.prefix != sys.base_prefix else "Not running in a virtual environment")
    log.info(
        "Execution policy is unrestricted" if check.execution_policy() else "Execution policy is restricted")

    # Python Version Check
    SystemInfoManager.python_version()

    # CPU Info
    for info in SystemInfoManager.cpu_info():
        log.info(info)

    # Final Debug Status
    log.info(f"Log Level: {DEBUG}")


if __name__ == "__main__":
    try:
        debug()
    except Exception as err:
        log.error(f"Failed to execute debug routine: {err}")
    time.sleep(0.5)
    input("Press Enter to exit...")
