from __future__ import annotations

import configparser
import os.path
from pathlib import Path


class Get:
    @staticmethod
    def list_of_files(directory: str, extensions: tuple | bool = True, append_file_list: list[str] = None,
                      exclude_files: list[str] = None) -> list:
        """
        Retrieves a list of files in the specified directory based on given extensions and exclusion criteria.
                      
        Supports two modes of file retrieval:
            1. When `extensions` is `True`, retrieves all files recursively from the directory.
            2. When `extensions` is a tuple, retrieves files matching specific extensions while applying exclusion rules.
                      
        Parameters:
            directory (str): Path of the directory to search for files.
            extensions (tuple | bool, optional): File extensions to filter or True to retrieve all files. Defaults to True.
            append_file_list (list, optional): Existing list to append found filenames to. Creates a new list if not provided. Defaults to None.
            exclude_files (list, optional): List of filenames to exclude from results. Defaults to None.
                      
        Returns:
            list: A list of filenames matching the specified criteria.
                      
        Exclusion rules:
            - Ignores files starting with an underscore (_)
            - Excludes "Logicytics.py"
            - Skips files specified in `exclude_files`
        """
        append_file_list = [] if not append_file_list else append_file_list

        if isinstance(extensions, bool) and extensions:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.relpath(os.path.join(root, filename), directory)
                    append_file_list.append(file_path)
            return append_file_list

        for filename in os.listdir(Path(directory)):
            if (
                    filename.endswith(extensions)
                    and not filename.startswith("_")
                    and filename != "Logicytics.py"
                    and (exclude_files is None or filename not in exclude_files)
            ):
                append_file_list.append(filename)
        return append_file_list

    @staticmethod
    def config_data() -> tuple[str, str, list[str], bool]:
        """
        Retrieves configuration data from the 'config.ini' file.
        
        This method attempts to read the 'config.ini' file from multiple potential locations:
        1. Current directory
        2. Parent directory
        3. Grandparent directory
        
        If the configuration file is not found in any of these locations, the program exits with an error message.
        
        Returns:
            tuple[str, str, list[str], bool]: A tuple containing:
                - Log level (str): Either "DEBUG" or "INFO"
                - Version (str): System version from configuration
                - Files (list[str]): List of files specified in configuration
                - Delete old logs (bool): Flag indicating whether to delete old log files
        
        Raises:
            SystemExit: If the 'config.ini' file cannot be found in any of the attempted locations
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
        except Exception:
            try:
                return get_config_data("../config.ini")
            except Exception:
                try:
                    return get_config_data("../../config.ini")
                except Exception:
                    print("The config.ini file is not found.")
                    exit(1)
