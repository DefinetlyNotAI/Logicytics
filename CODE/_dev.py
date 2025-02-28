from __future__ import annotations

import os
import re
import subprocess

import configobj

from logicytics import log, Get, FileManagement, CURRENT_FILES, VERSION


def color_print(text, color="reset", is_input=False) -> None | str:
    colors = {
        "reset": "\033[0m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "cyan": "\033[36m",
    }

    color_code = colors.get(color.lower(), colors["reset"])
    if is_input:
        return input(f"{color_code}{text}{colors['reset']}")
    else:
        print(f"{color_code}{text}{colors['reset']}")


def _update_ini_file(filename: str, new_data: list | str, key: str) -> None:
    """
    Updates an INI file with a new array of current files or version.
    Args:
        filename (str): The path to the INI file to be updated.
        new_data (list | str): The list of current files or the new version to be written to the INI file.
        key (str): The key in the INI file to be updated.
    Returns:
        None
    """
    try:
        config = configobj.ConfigObj(filename, encoding="utf-8", write_empty_values=True)

        if key == "files":
            config["System Settings"][key] = ", ".join(new_data)
        elif key == "version":
            config["System Settings"][key] = new_data
        else:
            color_print(f"[!] Invalid key: {key}", "yellow")
            return

        config.write()
    except FileNotFoundError:
        color_print("[x] INI file not found", "red")
    except configobj.ConfigObjError as e:
        color_print(f"[x] Parsing INI file failed: {e}", "red")
    except Exception as e:
        color_print(f"[x] {e}", "red")


def _prompt_user(question: str, file_to_open: str = None, special: bool = False) -> bool:
    """
    Prompts the user with a yes/no question and optionally opens a file.
    
    Parameters:
        question (str): The question to be presented to the user.
        file_to_open (str, optional): Path to a file that will be opened if the user does not respond affirmatively.
        special (bool, optional): Flag to suppress the default reminder message when the user responds negatively.
    
    Returns:
        bool: True if the user responds with 'yes' or 'Y', False otherwise.
    
    Raises:
        Exception: Logs any unexpected errors during user interaction.
    
    Notes:
        - Uses subprocess to open files on Windows systems
        - Case-insensitive input handling for 'yes' responses
        - Provides optional file opening and reminder messaging
    """
    try:
        answer = color_print(f"[?] {question} (y)es or (n)o:- ", "cyan", is_input=True)
        if not (answer.lower() == "yes" or answer.lower() == "y"):
            if file_to_open:
                subprocess.run(["start", file_to_open], shell=True)
            if not special:
                color_print(
                    "[x] Please ensure you fix the issues/problem and try again with the checklist.", "red"
                )
            return False
        return True
    except Exception as e:
        color_print(f"[x] {e}", "red")


def _perform_checks() -> bool:
    """
    Performs a series of user prompts for various checks.
    
    Returns:
        bool: True if all checks are confirmed by the user, False otherwise.
    """
    checks = [
        ("[-] Have you read the required contributing guidelines?", "..\\CONTRIBUTING.md"),
        ("[-] Have you made files you don't want to be run start with '_'?", "."),
        ("[-] Have you added the file to CODE dir?", "."),
        ("[-] Have you added docstrings and comments?", "..\\CONTRIBUTING.md"),
        ("[-] Is each file containing around 1 main feature?", "..\\CONTRIBUTING.md"),
    ]

    for question, file_to_open in checks:
        if not _prompt_user(question, file_to_open):
            return False
    return True


def _handle_file_operations() -> None:
    """
    Handles file operations and logging for added, removed, and normal files.
    """
    EXCLUDE_FILES = ["logicytics\\User_History.json.gz", "logicytics\\User_History.json"]
    files = Get.list_of_files(".", exclude_files=EXCLUDE_FILES, exclude_dirs=["SysInternal_Suite"])
    added_files, removed_files, normal_files = [], [], []
    clean_files_list = [file.replace('"', '') for file in CURRENT_FILES]

    files_set = set(os.path.abspath(f) for f in files)
    clean_files_set = set(os.path.abspath(f) for f in clean_files_list)

    for file in files_set:
        if file in clean_files_set and file not in EXCLUDE_FILES:
            normal_files.append(file)
        elif file not in clean_files_set and file not in EXCLUDE_FILES:
            added_files.append(file)

    for file in clean_files_set:
        if file not in files_set and file not in EXCLUDE_FILES:
            removed_files.append(file)

    print("\n".join([f"\033[92m+ {file}\033[0m" for file in added_files]))  # Green +
    print("\n".join([f"\033[91m- {file}\033[0m" for file in removed_files]))  # Red -
    print("\n".join([f"* {file}" for file in normal_files]))

    if not _prompt_user("[-] Does the list above include your added files?"):
        color_print("[x] Something went wrong! Please contact support.", "red")
        return

    max_attempts = 10
    attempts = 0
    _update_ini_file("config.ini", files, "files")

    while True:
        version = color_print(f"[?] Enter the new version of the project (Old version is {VERSION}): ", "cyan",
                              is_input=True)
        if attempts >= max_attempts:
            color_print("[x] Maximum attempts reached. Please run the script again.", "red")
            exit()
        if re.match(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$", version):
            _update_ini_file("config.ini", version, "version")
            break
        else:
            color_print("[!] Please enter a valid version number (e.g., 1.2.3)", "yellow")
            attempts += 1
            color_print(f"[!] {max_attempts - attempts} attempts remaining", "yellow")

    color_print("\n[-] Great Job! Please tick the box in the GitHub PR request for completing steps in --dev", "green")


@log.function
def dev_checks() -> None:
    """
    Performs comprehensive developer checks to ensure code quality and project guidelines compliance.
    
    This function guides developers through a series of predefined checks, validates file additions, 
    and updates project configuration. It performs the following key steps:
    - Verify adherence to contributing guidelines
    - Check file naming conventions
    - Validate file placement
    - Confirm docstring and comment coverage
    - Assess feature modularity
    - Categorize and display file changes
    - Update project configuration file
    
    Raises:
        None: Returns None if any check fails or an error occurs during the process.
    
    Side Effects:
        - Creates necessary directories
        - Prompts user for multiple confirmations
        - Prints file change lists with color coding
        - Updates configuration file with current files and version
        - Logs warnings or errors during the process
    """
    FileManagement.mkdir()
    if not _perform_checks():
        return
    _handle_file_operations()


if __name__ == "__main__":
    dev_checks()
    # Wait for the user to press Enter to exit the program
    input("\n[*] Press Enter to exit the program... ")
