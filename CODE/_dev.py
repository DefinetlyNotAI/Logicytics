import json
import os

from __lib_actions import Actions


# Super inefficient - If it works, it works tho ;)


def open_file(file: str) -> None:
    """
    Opens a specified file using its default application.

    Args:
        file (str): The path to the file to be opened.

    Returns:
        None
    """
    os.startfile(os.path.realpath(file))


def update_json_file(filename, new_array: list) -> None:
    """
    Updates a JSON file with a new array of current files.

    Args:
        filename (str): The path to the JSON file to be updated.
        new_array (list): The list of current files to be written to the JSON file.

    Returns:
        None
    """
    with open(filename, "r+") as f:
        data = json.load(f)
        data["CURRENT_FILES"] = new_array
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


def dev_checks() -> None:
    """
    Performs a series of checks to ensure that the developer has followed the required guidelines and best practices.

    This function prompts the developer with a series of questions to ensure that they have followed the required
    contributing guidelines, added files with a specific naming convention, added the file to the CODE directory,
    added docstrings and comments to their code, tested their code, ensured that each file contains only one feature,
    and has included the proper flags in their code.

    Returns:
        None

    Raises:
        None

    Example usage:
        dev_checks()
    """
    # Checks
    answer = input(
        "Have you made sure you read the required contributing guidlines? (yes or no):- "
    )
    if answer != "yes":
        open_file("../CONTRIBUTING.md")
        exit(
            "You did not select yes to the question, please try again after fixing your issue"
        )

    answer = input(
        "Have you made files you dont want to be run, start with '_'? (yes or no):- "
    )
    if answer != "yes":
        open_file(".")
        exit(
            "You did not select yes to the question, please try again after fixing your issue"
        )

    answer = input("Have you added the file to CODE dir? (yes or no):- ")
    if answer != "yes":
        open_file(".")
        open_file("../MODS")
        exit(
            "You did not select yes to the question, please try again after fixing your issue"
        )

    answer = input("Have you added docstrings and comments? (yes or no):- ")
    if answer != "yes":
        open_file("../CONTRIBUTING.md")
        exit(
            "You did not select yes to the question, please try again after fixing your issue"
        )

    answer = input("Have you made sure you tested your code? (yes or no):- ")
    if answer != "yes":
        open_file("__Test__.py")
        exit(
            "You did not select yes to the question, please try again after fixing your issue"
        )

    answer = input(
        "Have you made sure you have no more than 1 feature per file and the features are non repeated? (yes or no):- "
    )
    if answer != "yes":
        open_file("../CONTRIBUTING.md")
        exit(
            "You did not select yes to the question, please try again after fixing your issue"
        )

    answer = input(
        "Have you made sure you have added a comment to the TOP of your code with the flags you want to be "
        "included in or not? (yes or no):- "
    )
    if answer != "yes":
        open_file("../CONTRIBUTING.md")
        exit(
            "You did not select yes to the question, please try again after fixing your issue"
        )

    answer = input(
        "Have you made sure you DID NOT modify _wrapper.py unless told otherwise? (yes or no):- "
    )
    if answer != "yes":
        open_file("Logicytics.py")
        exit(
            "You did not select yes to the question, please try again after fixing your issue"
        )

    # Usage
    files = Actions.check_current_files(".")
    print(files)
    answer = input(
        "Nearly there! Does the list above include your added files? (yes or no):- "
    )
    if answer != "yes":
        print(
            "Something went wrong! Please contact support, If you are sure the list doesnt contain the proper files"
        )
        exit(
            "You did not select yes to the question, please try again after fixing your issue"
        )
    update_json_file("config.json", files)
    print(
        "Great Job, Please tick the box in github PR request for completing steps in --dev"
    )
