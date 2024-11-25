from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os.path
import shutil
import subprocess
import zipfile
from pathlib import Path
from subprocess import CompletedProcess

from __lib_log import *


class Flag:
    @classmethod
    def colorify(cls, text: str, color: str) -> str:
        """
        Adds color to the given text based on the specified color code.

        Args:
            text (str): The text to be colorized.
            color (str): The color code ('y' for yellow, 'r' for red, 'b' for blue).

        Returns:
            str: The colorized text if the color code is valid, otherwise the original text.
        """
        colors = {
            "y": "\033[93m",
            "r": "\033[91m",
            "b": "\033[94m"
        }
        RESET = "\033[0m"
        return f"{colors.get(color, '')}{text}{RESET}" if color in colors else text

    @classmethod
    def __available_arguments(cls) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
        """
        A static method used to parse command-line arguments for the Logicytics application.

        It defines various flags that can be used to customize the behavior of the application,
        including options for running in default or minimal mode, unzipping extra files,
        backing up or restoring data, updating from GitHub, and more.

        The method returns a tuple containing the parsed arguments and the argument parser object.

        Returns:
            tuple[argparse.Namespace, argparse.ArgumentParser]: A tuple containing the parsed arguments and the argument parser object.
        """
        # Define the argument parser
        parser = argparse.ArgumentParser(
            description="Logicytics, The most powerful tool for system data analysis. "
                        "This tool provides a comprehensive suite of features for analyzing system data, "
                        "including various modes for different levels of detail and customization."
        )

        # Define Actions Flags
        parser.add_argument(
            "--default",
            action="store_true",
            help="Runs Logicytics with its default settings and scripts. "
                 f"{cls.colorify('- Recommended for most users -', 'b')}",
        )
        parser.add_argument(
            "--threaded",
            action="store_true",
            help="Runs Logicytics using threads, where it runs in parallel, default settings though"
                 f"{cls.colorify('- Recommended for some users -', 'b')}",
        )
        parser.add_argument(
            "--modded",
            action="store_true",
            help="Runs the normal Logicytics, as well as any File in the MODS directory, "
                 "Used for custom scripts as well as default ones.",
        )
        parser.add_argument(
            "--depth",
            action="store_true",
            help="This flag will run all default script's in threading mode, "
                 "as well as any clunky and huge code, which produces a lot of data "
                 f"{cls.colorify('- Will take a long time -', 'y')}",
        )
        parser.add_argument(
            "--nopy",
            action="store_true",
            help="Run Logicytics using all non-python scripts, "
                 f"These may be {cls.colorify('outdated', 'y')} "
                 "and not the best, use only if the device doesnt have python installed.",
        )
        parser.add_argument(
            "--minimal",
            action="store_true",
            help="Run Logicytics in minimal mode. Just bare essential scraping using only quick scripts",
        )

        # Define Side Flags
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Runs the Debugger, Will check for any issues, "
                 "warning etc, useful for debugging and issue reporting "
                 f"{cls.colorify('- Use to get a special log file to report the bug -', 'b')}.",
        )
        parser.add_argument(
            "--backup",
            action="store_true",
            help="Backup Logicytics files to the ACCESS/BACKUPS directory "
                 f"{cls.colorify('- Use on your own device only -', 'y')}.",
        )
        parser.add_argument(
            "--update",
            action="store_true",
            help="Update Logicytics from GitHub, only if you have git properly installed "
                 "and the project was downloaded via git "
                 f"{cls.colorify('- Use on your own device only -', 'y')}.",
        )
        parser.add_argument(
            "--unzip-extra",
            action="store_true",
            help="Unzip the extra directory zip File "
                 f"{cls.colorify('- Use on your own device only -', 'y')}.",
        )
        parser.add_argument(
            "--extra",
            action="store_true",
            help="Open's the extra directory menu to use more tools. "
                 f"{cls.colorify('- Still experimental -', 'y')} "
                 f"{cls.colorify('- MUST have used --unzip-extra flag -', 'b')}.",
        )
        parser.add_argument(
            "--dev",
            action="store_true",
            help="Run Logicytics developer mod, this is only for people who want to "
                 "register their contributions properly. "
                 f"{cls.colorify('- Use on your own device only -', 'y')}.",
        )

        # Define After-Execution Flags
        parser.add_argument(
            "--reboot",
            action="store_true",
            help="Execute Flag that will reboot the device afterward",
        )
        parser.add_argument(
            "--shutdown",
            action="store_true",
            help="Execute Flag that will shutdown the device afterward",
        )

        # Not yet Implemented
        parser.add_argument(
            "--webhook",
            action="store_true",
            help="Execute Flag that will send zip File via webhook "
                 f"{cls.colorify('- Not yet Implemented -', 'r')}",
        )
        parser.add_argument(
            "--restore",
            action="store_true",
            help="Restore Logicytics files from the ACCESS/BACKUPS directory "
                 f"{cls.colorify('- Use on your own device only -', 'y')} "
                 f"{cls.colorify('- Not yet Implemented -', 'r')}",
        )
        return parser.parse_args(), parser

    @staticmethod
    def __exclusivity_logic(args: argparse.Namespace) -> bool:
        """
        Checks if exclusive flags are used in the provided arguments.

        Args:
            args (argparse.Namespace): The arguments to be checked.

        Returns:
            bool: True if exclusive flags are used, False otherwise.
        """
        special_flag_used = False
        if args.reboot or args.shutdown or args.webhook:
            if not (
                    args.default or args.threaded or args.modded or args.minimal or args.nopy or args.depth
            ):
                print("Invalid combination of flags.")
                exit(1)
            special_flag_used = True
        return special_flag_used

    @staticmethod
    def __used_flags_logic(args: argparse.Namespace) -> tuple[str, ...]:
        """
        Sets flags based on the provided arguments.

        Args:
            args (argparse.Namespace): The arguments to be checked for flags.

        Returns:
            tuple[str, ...]: A tuple of flag names that are set to True.
        """
        flags = {key: getattr(args, key) for key in vars(args)}
        true_keys = []
        for key, value in flags.items():
            if value:
                true_keys.append(key)
                if len(true_keys) == 2:
                    break
        return tuple(true_keys)

    @classmethod
    def data(cls) -> tuple[str, ...] | argparse.ArgumentParser:
        """
        Handles the parsing and validation of command-line flags.

        Returns either a tuple of used flag names or an ArgumentParser instance.
        """
        args, parser = cls.__available_arguments()
        special_flag_used = cls.__exclusivity_logic(args)

        if not special_flag_used:
            used_flags = [flag for flag in vars(args) if getattr(args, flag)]
            if len(used_flags) > 1:
                print("Only one flag is allowed.")
                exit(1)

        if special_flag_used:
            used_flags = cls.__used_flags_logic(args)
            if len(used_flags) > 2:
                print("Invalid combination of flags.")
                exit(1)

        if not tuple(used_flags):
            return parser
        else:
            return tuple(used_flags)


class FileManagement:
    @staticmethod
    def open_file(file: str, use_full_path=False) -> str | None:
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

    @staticmethod
    def mkdir():
        """
        Creates the necessary directories for storing logs, backups, and data.

        Returns:
            None
        """
        os.makedirs("../ACCESS/LOGS/", exist_ok=True)
        os.makedirs("../ACCESS/LOGS/DEBUG", exist_ok=True)
        os.makedirs("../ACCESS/BACKUP/", exist_ok=True)
        os.makedirs("../ACCESS/DATA/Hashes", exist_ok=True)
        os.makedirs("../ACCESS/DATA/Zip", exist_ok=True)

    @staticmethod
    def unzip(zip_path: Path):
        """
        Unzips a given zip file to a new directory with the same name.

        Args:
            zip_path (str): The path to the zip file to be unzipped.

        Returns:
            None
        """
        # Get the base name of the zip file
        base_name = os.path.splitext(os.path.basename(zip_path))[0]

        # Create a new directory with the same name as the zip file
        output_dir = os.path.join(os.path.dirname(zip_path), base_name)
        os.makedirs(output_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(path=str(output_dir))

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
            excluded_extensions = (".py", ".exe", ".bat", ".ps1")
            excluded_prefixes = ("config.", "SysInternal_Suite", "__pycache__")

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
            today = datetime.now()
            filename = f"Logicytics_{name}_{flag}_{today.strftime('%Y-%m-%d_%H-%M-%S')}"
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
            ignore_file = os.path.exists("SysInternal_Suite/.sys.ignore")
            zip_file = os.path.exists("SysInternal_Suite/SysInternal_Suite.zip")

            if zip_file and not ignore_file:
                with zipfile.ZipFile(
                        "SysInternal_Suite/SysInternal_Suite.zip"
                ) as zip_ref:
                    zip_ref.extractall("SysInternal_Suite")
                    return "SysInternal_Suite zip extracted"

            elif ignore_file:
                return "Found .sys.ignore file, skipping SysInternal_Suite zip extraction"

        except Exception as err:
            exit(f"Failed to unzip SysInternal_Suite: {err}")


class Execute:
    @classmethod
    def script(cls, script_path: str) -> list[list[str]] | None:
        """
        Executes a script file and handles its output based on the file extension.
        Parameters:
            script_path (str): The path of the script file to be executed.
        """
        if script_path.endswith(".py"):
            cls.__run_python_script(script_path)
            return None
        else:
            if script_path.endswith(".ps1"):
                cls.__unblock_ps1_script(script_path)
            return cls.__run_other_script(script_path)

    @staticmethod
    def command(command: str) -> str:
        """
        Runs a command in a subprocess and returns the output as a string.

        Parameters:
            command (str): The command to be executed.

        Returns:
            CompletedProcess.stdout: The output of the command as a string.
        """
        process = subprocess.run(command, capture_output=True, text=True)
        return process.stdout

    @staticmethod
    def __unblock_ps1_script(script: str):
        """
        Unblocks and runs a PowerShell (.ps1) script.
        Parameters:
            script (str): The path of the PowerShell script.
        Returns:
            None
        """
        try:
            unblock_command = f'powershell.exe -Command "Unblock-File -Path {script}"'
            subprocess.run(unblock_command, shell=False, check=True)
        except Exception as err:
            exit(f"Failed to unblock script: {err}")

    @staticmethod
    def __run_python_script(script: str):
        """
        Runs a Python (.py) script.
        Parameters:
            script (str): The path of the Python script.
        Returns:
            None
        """
        result = subprocess.Popen(
            ["python", script], stdout=subprocess.PIPE
        ).communicate()[0]
        # LEAVE AS PRINT
        print(result.decode())

    @classmethod
    def __run_other_script(cls, script: str) -> list[list[str]]:
        """
        Runs a script with other extensions and logs output based on its content.
        Parameters:
            script (str): The path of the script.
        Returns:
            None
        """
        result = cls.command(f"powershell.exe -File {script}")
        lines = result.splitlines()
        messages = []
        for line in lines:
            if ":" in line:
                id_part, message_part = line.split(":", 1)
                messages.append([message_part.strip(), id_part.strip()])
        return messages


class Get:
    @staticmethod
    def list_of_files(directory: str, append_file_list: list = None, extensions: tuple = (".py", ".exe", ".ps1", ".bat")) -> list:
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
    def config_data() -> tuple[str, str, list[str]]:
        """
        Reads the configuration from the config.json file.

        Returns:
            A tuple containing the webhook URL, debug mode, version, API key, and a list of current files.
            The types of the returned values are:
                - webhook_url: str
                - debug: bool
                - version: str
                - api_key: str
                - files: list[str]

        Raises:
            FileNotFoundError: If the config.json file is not found.
            SystemExit: If the config.json file has an invalid format.
        """
        try:
            script_dir = Path(__file__).parent.absolute()
            config_path = script_dir / "config.json"
            with open(config_path, "r") as file:
                data = json.load(file)

                debug = data.get("Log Level Debug?", False)
                version = data.get("VERSION", "Unknown")
                files = data.get("CURRENT_FILES", ["Unknown"])

                if not (
                        isinstance(debug, bool)
                        and isinstance(version, str)
                        and isinstance(files, list)
                ):
                    print("Invalid config.json format.")
                    input("Press Enter to exit...")
                    exit(1)
                if debug:
                    debug = "DEBUG"
                else:
                    debug = "INFO"
                return debug, version, files
        except FileNotFoundError:
            print("The config.json File is not found.")
            input("Press Enter to exit...")
            exit(1)


DEBUG, VERSION, CURRENT_FILES = Get.config_data()
if __name__ == "__main__":
    Log().exception(
        "This is a library file and should not be executed directly.", Exception
    )
