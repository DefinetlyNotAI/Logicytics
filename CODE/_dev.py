from __future__ import annotations

import subprocess

import configobj

from logicytics import log, Get, FileManagement, CURRENT_FILES, VERSION


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
        config = configobj.ConfigObj(filename, encoding='utf-8', write_empty_values=True)
        if key == "files":
            config["System Settings"][key] = ", ".join(new_data)
        elif key == "version":
            config["System Settings"][key] = new_data
        else:
            log.error(f"Invalid key: {key}")
            return
        config.write()
    except FileNotFoundError:
        log.error(f"File not found: {filename}")
    except configobj.ConfigObjError as e:
        log.error(f"Error parsing INI file: {filename}, {e}")
    except Exception as e:
        log.error(f"An error occurred: {e}")


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
        answer = input(question + " (Y)es or (N)o:- ")
        if not (answer.lower() == "yes" or answer.lower() == "y"):
            if file_to_open:
                subprocess.run(["start", file_to_open], shell=True)
            if not special:
                print(
                    "Please ensure you fix the issues/problem and try again with the checklist."
                )
            return False
        return True
    except Exception as e:
        log.error(e)


def _perform_checks() -> bool:
    """
    Performs a series of user prompts for various checks.
    
    Returns:
        bool: True if all checks are confirmed by the user, False otherwise.
    """
    checks = [
        ("Have you read the required contributing guidelines?", "../CONTRIBUTING.md"),
        ("Have you made files you don't want to be run start with '_'?", "."),
        ("Have you added the file to CODE dir?", "."),
        ("Have you added docstrings and comments?", "../CONTRIBUTING.md"),
        ("Is each file containing around 1 main feature?", "../CONTRIBUTING.md"),
    ]

    for question, file_to_open in checks:
        if not _prompt_user(question, file_to_open):
            log.warning("Fix the issues and try again with the checklist.")
            return False
    return True


def _handle_file_operations() -> None:
    """
    Handles file operations and logging for added, removed, and normal files.
    """
    EXCLUDE_FILES = ["logicytics\\User_History.json.gz", "logicytics\\User_History.json"]
    files = Get.list_of_files(".", True, exclude_files=EXCLUDE_FILES)
    added_files, removed_files, normal_files = [], [], []
    clean_files_list = [file.replace('"', '') for file in CURRENT_FILES]

    for f in files:
        clean_f = f.replace('"', '')
        if clean_f in clean_files_list and clean_f not in EXCLUDE_FILES:
            normal_files.append(clean_f)
        elif clean_f not in EXCLUDE_FILES:
            added_files.append(clean_f)

    for f in clean_files_list:
        clean_f = f.replace('"', '')
        if clean_f not in files and clean_f not in EXCLUDE_FILES:
            removed_files.append(clean_f)

    print("\n".join([f"\033[92m+ {file}\033[0m" for file in added_files]))  # Green +
    print("\n".join([f"\033[91m- {file}\033[0m" for file in removed_files]))  # Red -
    print("\n".join([f"* {file}" for file in normal_files]))

    if not _prompt_user("Does the list above include your added files?"):
        log.critical("Something went wrong! Please contact support.")
        return

    _update_ini_file("config.ini", files, "files")
    _update_ini_file("config.ini", input(f"Enter the new version of the project (Old version is {VERSION}): "),
                     "version")
    print("\nGreat Job! Please tick the box in the GitHub PR request for completing steps in --dev")


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
    
    Example:
        Typical usage is during project development to ensure consistent practices:
        >>> dev_checks()
        # Interactively guides developer through project checks
    """
    FileManagement.mkdir()
    if not _perform_checks():
        return
    _handle_file_operations()


if __name__ == "__main__":
    dev_checks()
    # Wait for the user to press Enter to exit the program
    input("\nPress Enter to exit the program... ")
