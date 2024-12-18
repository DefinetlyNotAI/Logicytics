from __future__ import annotations

import ctypes
import os.path
import subprocess
import zipfile

from logicytics.Execute import Execute


class Check:
    @staticmethod
    def admin() -> bool:
        """
        Check if the current user has administrative privileges.

        Returns:
            bool: True if the user is an admin, False otherwise.
        """
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except AttributeError:
            return False

    @staticmethod
    def execution_policy() -> bool:
        result = subprocess.run(
            ["powershell", "-Command", "Get-ExecutionPolicy"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().lower() == "unrestricted"

    @staticmethod
    def uac() -> bool:
        """
        Check if User Account Control (UAC) is enabled on the system.

        This function runs a PowerShell command to retrieve the value of the EnableLUA registry key,
        which indicates whether UAC is enabled. It then returns True if UAC is enabled, False otherwise.

        Returns:
            bool: True if UAC is enabled, False otherwise.
        """
        value = Execute.command(
            r"powershell (Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System).EnableLUA"
        )
        return int(value.strip("\n")) == 1

    @staticmethod
    def sys_internal_zip() -> str:
        """
        Extracts the SysInternal_Suite zip file if it exists and is not ignored.

        This function checks if the SysInternal_Suite zip file exists and if it is not ignored.
         If the zip file exists and is not ignored,
         it extracts its contents to the SysInternal_Suite directory.
         If the zip file is ignored, it prints a message indicating that it is skipping the extraction.

        Raises:
            Exception: If there is an error during the extraction process. The error message is printed to the console and the program exits.
        """
        try:
            ignore_file = os.path.exists("../SysInternal_Suite/.sys.ignore")
            zip_file = os.path.exists("../SysInternal_Suite/SysInternal_Suite.zip")

            if zip_file and not ignore_file:
                with zipfile.ZipFile(
                        "../SysInternal_Suite/SysInternal_Suite.zip"
                ) as zip_ref:
                    zip_ref.extractall("SysInternal_Suite")
                    return "SysInternal_Suite zip extracted"

            elif ignore_file:
                return "Found .sys.ignore file, skipping SysInternal_Suite zip extraction"

        except Exception as err:
            exit(f"Failed to unzip SysInternal_Suite: {err}")
