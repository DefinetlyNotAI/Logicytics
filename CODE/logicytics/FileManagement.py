from __future__ import annotations
from __future__ import annotations

import hashlib
import os.path
import shutil
import subprocess
import zipfile
from datetime import datetime


class FileManagement:
    @staticmethod
    def open_file(file: str, use_full_path: bool = False) -> str | None:
        """
        Opens a specified file using its default application in a cross-platform manner.
        Args:
            file (str): The path to the file to be opened.
            use_full_path (bool): Whether to use the full path of the file or not.
        Returns:
            None
        """
        if not file == "":
            if use_full_path:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(current_dir, file)
            else:
                file_path = os.path.realpath(file)
            try:
                subprocess.run(["start", file_path], shell=False)
            except Exception as e:
                return f"Error opening file: {e}"
        return None

    @staticmethod
    def mkdir():
        """
        Creates the necessary directories for storing logs, backups, and data.
        
        This method ensures the existence of specific directory structures used by the application, including:
        - Log directories for general, debug, and performance logs
        - Backup directory
        - Data directories for storing hashes and zip files
        
        The method uses `os.makedirs()` with `exist_ok=True` to create directories without raising an error if they already exist.
        
        Returns:
            None: No return value. Directories are created as a side effect.
        """
        os.makedirs("../ACCESS/LOGS/", exist_ok=True)
        os.makedirs("../ACCESS/LOGS/DEBUG", exist_ok=True)
        os.makedirs("../ACCESS/LOGS/PERFORMANCE", exist_ok=True)
        os.makedirs("../ACCESS/BACKUP/", exist_ok=True)
        os.makedirs("../ACCESS/DATA/Hashes", exist_ok=True)
        os.makedirs("../ACCESS/DATA/Zip", exist_ok=True)

    class Zip:
        """
        A class to handle zipping files, generating SHA256 hashes, and moving files.

        Methods:
            __get_files_to_zip(path: str) -> list:
                Returns a list of files to be zipped, excluding certain file types and names.

            __create_zip_file(path: str, files: list, filename: str):
                Creates a zip file from the given list of files.

            __remove_files(path: str, files: list):
                Removes the specified files from the given path.

            __generate_sha256_hash(filename: str) -> str:
                Generates a SHA256 hash for the specified zip file.

            __write_hash_to_file(filename: str, sha256_hash: str):
                Writes the SHA256 hash to a file.

            __move_files(filename: str):
                Moves the zip file and its hash file to designated directories.

            and_hash(cls, path: str, name: str, flag: str) -> tuple | str:
                Zips files, generates a SHA256 hash, and moves the files.
        """

        @staticmethod
        def __get_files_to_zip(path: str) -> list:
            """
            Returns a list of files and directories to be zipped, excluding certain file types and names.

            Args:
                path (str): The directory path to search for files.

            Returns:
                list: A list of file and directory names to be zipped.
            """
            excluded_extensions = (".py", ".exe", ".bat", ".ps1", ".pkl", ".pth")
            excluded_prefixes = ("config.ini", "SysInternal_Suite",
                                 "__pycache__", "logicytics", "VulnScan")

            return [
                f for f in os.listdir(path)
                if not f.endswith(excluded_extensions) and not f.startswith(excluded_prefixes)
            ]

        @staticmethod
        def __create_zip_file(path: str, files: list, filename: str):
            """
            Creates a zip file from the given list of files.

            Args:
                path (str): The directory path containing the files.
                files (list): A list of file names to be zipped.
                filename (str): The name of the output zip file.

            Returns:
                None
            """

            def ignore_files(files_func):
                for root, _, file_func in os.walk(os.path.join(path, files_func)):
                    for f in file_func:
                        zip_file.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), path))

            with zipfile.ZipFile(f"{filename}.zip", "w") as zip_file:
                for file in files:
                    if os.path.isdir(os.path.join(path, file)):
                        ignore_files(file)
                    else:
                        zip_file.write(os.path.join(path, file))

        @staticmethod
        def __remove_files(path: str, files: list) -> str | None:
            """
            Removes the specified files from the given path.

            Args:
                path (str): The directory path containing the files.
                files (list): A list of file names to be removed.

            Returns:
                None or str: Returns an error message if an exception occurs.
            """
            for file in files:
                try:
                    shutil.rmtree(os.path.join(path, file))
                except OSError:
                    os.remove(os.path.join(path, file))
                except Exception as e:
                    return f"Error: {e}"
            return None

        @staticmethod
        def __generate_sha256_hash(filename: str) -> str:
            """
            Generates a SHA256 hash for the specified zip file.

            Args:
                filename (str): The name of the zip file.

            Returns:
                str: The SHA256 hash of the zip file.
            """
            with open(f"{filename}.zip", "rb") as zip_file:
                zip_data = zip_file.read()
            return hashlib.sha256(zip_data).hexdigest()

        @staticmethod
        def __write_hash_to_file(filename: str, sha256_hash: str):
            """
            Writes the SHA256 hash to a file.

            Args:
                filename (str): The name of the hash file.
                sha256_hash (str): The SHA256 hash to be written.

            Returns:
                None
            """
            with open(f"{filename}.hash", "w") as hash_file:
                hash_file.write(sha256_hash)

        @staticmethod
        def __move_files(filename: str):
            """
            Moves the zip file and its hash file to designated directories.

            Args:
                filename (str): The name of the files to be moved.

            Returns:
                None
            """
            shutil.move(f"{filename}.zip", "../ACCESS/DATA/Zip")
            shutil.move(f"{filename}.hash", "../ACCESS/DATA/Hashes")

        @classmethod
        def and_hash(cls, path: str, name: str, flag: str) -> tuple | str:
            """
            Zips files, generates a SHA256 hash, and moves the files.

            Args:
                path (str): The directory path containing the files.
                name (str): The base name for the output files.
                flag (str): A flag to be included in the output file names.

            Returns:
                tuple or str: A tuple containing success messages or an error message.
            """
            time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"Logicytics_{name}_{flag}_{time}"
            files_to_zip = cls.__get_files_to_zip(path)
            cls.__create_zip_file(path, files_to_zip, filename)
            check = cls.__remove_files(path, files_to_zip)
            if isinstance(check, str):
                return check
            else:
                sha256_hash = cls.__generate_sha256_hash(filename)
                cls.__write_hash_to_file(filename, sha256_hash)
                cls.__move_files(filename)
                return (
                    f"Zip file moved to ../ACCESS/DATA/Zip/{filename}.zip",
                    f"SHA256 Hash file moved to ../ACCESS/DATA/Hashes/{filename}.hash",
                )
