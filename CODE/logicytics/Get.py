from __future__ import annotations

import configparser
import os.path
from pathlib import Path

from logicytics.Logger import *


class Get:
    @staticmethod
    def list_of_files(directory: str, append_file_list: list = None,
                      extensions: tuple = (".py", ".exe", ".ps1", ".bat")) -> list:
        """
        Retrieves a list of files in the specified directory that have the specified extensions.

        Files starting with an underscore (_) and the file Logicytics.py are excluded from the list.

        Parameters:
            directory (str): The path of the directory to search.
            append_file_list (list): The list to append the filenames to.
            extensions (tuple): The extensions of the files to search for.
        Returns:
            list: The list of filenames with the specified extensions.
        """
        if not append_file_list:
            append_file_list = []
        for filename in os.listdir(Path(directory)):
            if (
                    filename.endswith(extensions)
                    and not filename.startswith("_")
                    and filename != "Logicytics.py"
            ):
                append_file_list.append(filename)
        return append_file_list

    @staticmethod
    def list_of_code_files(directory: str) -> list:
        """
        Retrieves a list of files with specific extensions within a specified directory and its subdirectories.

        Args:
            directory (str): The path to the directory to search for files.

        Returns:
            list: A list of file paths with the following extensions: .py, .exe, .ps1, .bat, .vbs.
        """
        file = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith((".py", ".exe", ".ps1", ".bat", ".vbs")):
                    files_path = os.path.join(root, filename)
                    file.append(files_path.removeprefix(".\\"))
        return file

    @staticmethod
    def config_data() -> tuple[str, str, list[str], bool]:
        """
        Retrieves configuration data from the 'config.ini' file.

        This method attempts to read the 'config.ini' file located in the current directory.
        If the file is not found, it attempts to read the 'config.ini' file from the parent directory.
        If neither file is found, the program exits with an error message.

        Returns:
            tuple[str, str, list[str], bool]: A tuple containing the log level, version, and a list of files.
        """
        def get_config_data(config_file_name: str) -> tuple[str, str, list[str], bool]:
            """
            Reads configuration data from the specified 'config.ini' file.

            Args:
                config_file_name (str): The name of the configuration file to read.

            Returns:
                tuple[str, str, list[str], bool]: A tuple containing the log level, version, and a list of files.
            """
            config = configparser.ConfigParser()
            config.read(config_file_name)

            log_using_debug = config.getboolean("Settings", "log_using_debug")
            delete_old_logs = config.getboolean("Settings", "delete_old_logs")
            version = config.get("System Settings", "version")
            files = config.get("System Settings", "files").split(", ")

            log_using_debug = "DEBUG" if log_using_debug else "INFO"

            return log_using_debug, version, files, delete_old_logs

        try:
            return get_config_data("config.ini")
        except FileNotFoundError:
            try:
                return get_config_data("../config.ini")
            except FileNotFoundError:
                print("The config.ini file is not found.")
                exit(1)
