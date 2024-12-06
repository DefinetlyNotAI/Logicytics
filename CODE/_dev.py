from __future__ import annotations

import configparser
import subprocess

from logicytics import Log, DEBUG, Get, FileManagement, CURRENT_FILES, VERSION

if __name__ == "__main__":
    log_dev = Log({"log_level": DEBUG})


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
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(filename)
        if key == "files":
            config["System Settings"][key] = ", ".join(new_data)
        elif key == "version":
            config["System Settings"][key] = new_data
        else:
            log_dev.error(f"Invalid key: {key}")
            return
        with open(filename, "w", encoding="utf-8") as configfile:
            # noinspection PyTypeChecker
            config.write(configfile, space_around_delimiters=False)
    except FileNotFoundError:
        log_dev.error(f"File not found: {filename}")
    except configparser.Error as e:
        log_dev.error(f"Error parsing INI file: {filename}, {e}")
    except Exception as e:
        log_dev.error(f"An error occurred: {e}")


def _prompt_user(question: str, file_to_open: str = None, special: bool = False) -> bool:
    """
        Prompts the user with a question and optionally opens a file if the answer is not 'yes'.
        Args:
            question (str): The question to ask the user.
            file_to_open (str, optional): The file to open if the user doesn't answer 'yes'.
        Returns:
            bool: True if the user's answer is 'yes', otherwise False.
        """
    try:
        answer = input(question + " (yes or no):- ")
        if answer.lower() != "yes":
            if file_to_open:
                subprocess.run(["start", file_to_open], shell=True)
            if not special:
                print(
                    "Please ensure you fix the issues/problem and try again with the checklist."
                )
            return False
        return True
    except Exception as e:
        log_dev.error(e)


def dev_checks() -> None:
    """
        Performs a series of checks to ensure that the developer has followed the required guidelines and best practices.
        Returns:
            bool: True if all checks pass, otherwise False.
        """
    # Create the necessary directories if they do not exist
    FileManagement.mkdir()

    # List of checks to be performed, each check is a tuple containing a question and a file to open if the answer is not 'yes'
    checks = [
        ("Have you read the required contributing guidelines?", "../CONTRIBUTING.md"),
        ("Have you made files you don't want to be run start with '_'?", "."),
        ("Have you added the file to CODE dir?", "."),
        ("Have you added docstrings and comments?", "../CONTRIBUTING.md"),
        ("Is each file containing around 1 main feature?", "../CONTRIBUTING.md"),
    ]

    try:
        # Iterate through each check and prompt the user
        for question, file_to_open in checks:
            if not _prompt_user(question, file_to_open):
                log_dev.warning("Fix the issues and try again with the checklist.")
                return None

        # Get the list of code files in the current directory
        files = Get.list_of_code_files(".")
        added_files = [f for f in files if f not in CURRENT_FILES]
        removed_files = [f for f in CURRENT_FILES if f not in files]
        normal_files = [f for f in files if f in CURRENT_FILES]

        # Print the list of added, removed, and normal files in color
        print("\n".join([f"\033[92m+ {file}\033[0m" for file in added_files]))  # Green +
        print("\n".join([f"\033[91m- {file}\033[0m" for file in removed_files]))  # Red -
        print("\n".join([f"* {file}" for file in normal_files]))

        # Prompt the user to confirm if the list includes their added files
        if not _prompt_user("Does the list above include your added files?"):
            log_dev.critical("Something went wrong! Please contact support.")
            return None

        # Update the JSON file with the current list of files
        _update_ini_file("config.ini", files, "files")

        # Prompt the user to enter the new version of the project and update the JSON file
        _update_ini_file("config.ini", input(f"Enter the new version of the project (Old version is {VERSION}): "),
                         "version")

        # Print a message indicating the completion of the steps
        print("\nGreat Job! Please tick the box in the GitHub PR request for completing steps in --dev")
    except Exception as e:
        # Log any exceptions that occur during the process
        log_dev.exception(str(e))


dev_checks()
# Wait for the user to press Enter to exit the program
input("\nPress Enter to exit the program... ")
