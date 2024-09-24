from __future__ import annotations
from datetime import datetime
import hashlib
import shutil
import os
import zipfile


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

        and_hash(self, path: str, name: str, flag: str) -> tuple | str:
            Zips files, generates a SHA256 hash, and moves the files.
    """

    @staticmethod
    def __get_files_to_zip(path: str) -> list:
        """
        Returns a list of files to be zipped, excluding certain file types and names.

        Args:
            path (str): The directory path to search for files.

        Returns:
            list: A list of file names to be zipped.
        """
        return [
            f
            for f in os.listdir(path)
            if not f.endswith((".py", ".exe", ".bat", ".ps1"))
            and not f.startswith(("config.", "SysInternal_Suite", "__pycache__"))
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
        with zipfile.ZipFile(f"{filename}.zip", "w") as zip_file:
            for file in files:
                zip_file.write(os.path.join(path, file))

    @staticmethod
    def __remove_files(path: str, files: list):
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

    def and_hash(self, path: str, name: str, flag: str) -> tuple | str:
        """
        Zips files, generates a SHA256 hash, and moves the files.

        Args:
            path (str): The directory path containing the files.
            name (str): The base name for the output files.
            flag (str): A flag to be included in the output file names.

        Returns:
            tuple or str: A tuple containing success messages or an error message.
        """
        today = datetime.now()
        filename = f"Logicytics_{name}_{flag}_{today.strftime('%Y-%m-%d_%H-%M-%S')}"
        files_to_zip = self.__get_files_to_zip(path)
        self.__create_zip_file(path, files_to_zip, filename)
        check = self.__remove_files(path, files_to_zip)
        if isinstance(check, str):
            return check
        else:
            sha256_hash = self.__generate_sha256_hash(filename)
            self.__write_hash_to_file(filename, sha256_hash)
            self.__move_files(filename)
            return (
                f"Zip file moved to ../ACCESS/DATA/Zip/{filename}.zip",
                f"SHA256 Hash file moved to ../ACCESS/DATA/Hashes/{filename}.hash",
            )
