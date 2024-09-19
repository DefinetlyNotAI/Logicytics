from __future__ import annotations
open("CUSTOM.LOG.MECHANISM", "w").close()
import platform
import os.path
import requests
import psutil
import sys
from __lib_class import *
log_debug = Log(debug=DEBUG, filename="../ACCESS/LOGS/DEBUG/DEBUG.LOG")
log_debug_funcs = {
    "INFO": log_debug.info,
    "WARNING": log_debug.warning,
    "ERROR": log_debug.error,
    "CRITICAL": log_debug.critical,
    None: log_debug.debug,
}


class HealthCheck:
    def get_config_data(self) -> bool | tuple[tuple[str, str, str], tuple[str, str, str]]:
        try:
            url = "https://raw.githubusercontent.com/DefinetlyNotAI/Logicytics/main/CODE/config.json"
            config = json.loads(requests.get(url).text)
        except requests.exceptions.ConnectionError:
            log_debug.warning("No connection found")
            return False
        version_check = self.__compare_versions(VERSION, config['VERSION'])
        file_check = self.__check_files(CURRENT_FILES, config['CURRENT_FILES'])

        return version_check, file_check

    @staticmethod
    def __compare_versions(local_version, remote_version) -> tuple[str, str, str]:
        if local_version == remote_version:
            return "Version is up to date.", f"Your Version: {local_version}", "INFO"
        elif local_version > remote_version:
            return "Version is ahead of the repository.", f"Your Version: {local_version}, Repository Version: {remote_version}", "WARNING"
        else:
            return "Version is behind the repository.", f"Your Version: {local_version}, Repository Version: {remote_version}", "ERROR"

    @staticmethod
    def __check_files(local_files, remote_files) -> tuple[str, str, str]:
        missing_files = set(remote_files) - set(local_files)
        if not missing_files:
            return "All files are present.", f"Your files: {local_files} contain all the files in the repository.", "INFO"
        else:
            return "You have missing files.", f"You are missing the following files: {missing_files}", "ERROR"


class DebugCheck:
    @staticmethod
    def SysInternal_Binaries(path):
        try:
            contents = os.listdir(path)
            log_debug.debug(contents)
            if any(file.endswith('.ignore') for file in contents):
                return "A `.sys.ignore` file was found - Ignoring", "WARNING"
            if any(file.endswith('.zip') for file in contents) and not any(file.endswith('.exe') for file in contents):
                return "Only zip files - Missing EXE's due to no `ignore` file", "ERROR"
            elif any(file.endswith('.zip') for file in contents) and any(file.endswith('.exe') for file in contents):
                return "Both zip and exe files - All good", "INFO"
            else:
                return "SysInternal Binaries Not Found: Missing Files - Corruption detected", "ERROR"
        except FileNotFoundError:
            return "SysInternal Binaries Not Found: Missing Directory- Corruption detected", "ERROR"
        except Exception as e:
            return f"An Unexpected error occurred: {e}", "ERROR"

    @staticmethod
    def execution_policy():
        result = subprocess.run(['powershell', '-Command', 'Get-ExecutionPolicy'], capture_output=True, text=True)
        return result.stdout.strip().lower() == 'unrestricted'

    @staticmethod
    def cpu_info():
        return 'CPU Architecture: ' + platform.machine(), 'CPU Vendor Id: ' + platform.system(), 'CPU Model: ' + f"{platform.release()} {platform.version()}"


def debug():
    # Clear Debug Log
    if os.path.exists("../ACCESS/LOGS/DEBUG/DEBUG.LOG"):
        os.remove("../ACCESS/LOGS/DEBUG/DEBUG.LOG")

    # Check File integrity (Online)
    if HealthCheck().get_config_data():
        version_tuple, file_tuple = HealthCheck().get_config_data()
        log_debug_funcs.get(version_tuple[2], log_debug.debug)("\n".join(version_tuple[0]).replace('\n', ''))
        log_debug_funcs.get(file_tuple[2], log_debug.debug)("\n".join(file_tuple[0]).replace('\n', ''))
    message, type = DebugCheck.SysInternal_Binaries("SysInternal_Suite")
    log_debug_funcs.get(type, log_debug.debug)("\n".join(message).replace('\n', ''))

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
    log_debug.info(f"Python Version Used: {sys.version.split()[0]} - Recommended Version is: ~")

    # Get Repo Path
    log_debug.info(os.path.abspath(__file__).removesuffix("\\CODE\\_debug.py"))

    # Get CPU Info
    architecture, vID, cpuModel = DebugCheck.cpu_info()
    log_debug.info(architecture)
    log_debug.info(vID)
    log_debug.info(cpuModel)

    # Get config data
    log_debug.info("Debug: " + DEBUG)
