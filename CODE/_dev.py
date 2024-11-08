from __future__ import annotations
from __lib_class import *


class Dev:
    @staticmethod
    def __update_json_file(filename: str, new_data: list | str, key: str) -> None:
        """
        Updates a JSON file with a new array of current files.
        Args:
            filename (str): The path to the JSON file to be updated.
            new_data (list | str): The list of current files to be written to the JSON file.
            key (str): The key in the JSON file to be updated.
        Returns:
            None
        """
        try:
            with open(filename, "r+") as f:
                data = json.load(f)
                data[key] = new_data
                f.seek(0)
                # noinspection PyTypeChecker
                json.dump(data, f, indent=4)
                f.truncate()
        except FileNotFoundError:
            log_dev.error(f"File not found: {filename}")
        except json.JSONDecodeError:
            log_dev.error(f"Error decoding JSON in the file: {filename}")
        except Exception as e:
            log_dev.error(f"An error occurred: {e}")

    @staticmethod
    def __prompt_user(question: str, file_to_open: str = None, special: bool = False) -> bool:
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

    def dev_checks(self) -> str | None:
        """
        Performs a series of checks to ensure that the developer has followed the required guidelines and best practices.
        Returns:
            bool: True if all checks pass, otherwise False.
        """
        Actions.mkdir()
        checks = [
            ("Have you read the required contributing guidelines?", "../CONTRIBUTING.md"),
            ("Have you made files you don't want to be run start with '_'?", "."),
            ("Have you added the file to CODE dir?", "."),
            ("Have you added docstrings and comments?", "../CONTRIBUTING.md"),
            ("Is each file containing no more than 1 feature?", "../CONTRIBUTING.md"),
            ("Have you NOT modified __wrapper__.py without authorization?", "Logicytics.py"),
        ]
        try:
            for question, file_to_open in checks:
                if not self.__prompt_user(question, file_to_open):
                    return "Fix the issues and try again with the checklist."

            remind = self.__prompt_user(
                "If the update is a major or minor upgrade (non-patch update) answer `yes`?", special=True
            )
            if remind:
                remind = not self.__prompt_user("Did You Build the EXE with Advanced Installer?", "../Logicytics.aip")

            files = Actions.check_current_files(".")
            print(files)
            if not self.__prompt_user("Does the list above include your added files?"):
                return "Something went wrong! Please contact support."

            self.__update_json_file("config.json", files, "CURRENT_FILES")
            self.__update_json_file(
                "config.json",
                input(f"Enter the new version of the project (Old version is {VERSION}):"),
                "VERSION",
            )
            print("Great Job! Please tick the box in the GitHub PR request for completing steps in --dev")
            if remind:
                print("Remember to upload the EXE files on the PR!")
            return None
        except Exception as e:
            return str(e)


log_dev = Log({"log_level": DEBUG})
message = Dev().dev_checks()
if message is not None:
    log_dev.error(message)
