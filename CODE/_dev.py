import os
import json


# Super inefficient - If it works, it works tho ;)


def open_file(file):
    os.startfile(os.path.realpath(file))


def check_current_files(directory):
    file = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(('.py', '.exe', '.ps1', '.bat')):
                files_path = os.path.join(root, filename)
                file.append(files_path.removeprefix('.\\'))
    return file


def update_json_file(filename, new_array):
    with open(filename, 'r+') as f:
        data = json.load(f)
        data['CURRENT_FILES'] = new_array
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


def dev_checks():
    # Checks
    answer = input("Have you made sure you read the required contributing guidlines? (yes or no):- ")
    if answer != "yes":
        open_file("../CONTRIBUTING.md")
        exit("You did not select yes to the question, please try again after fixing your issue")

    answer = input("Have you made files you dont want to be run, start with '_'? (yes or no):- ")
    if answer != "yes":
        open_file(".")
        exit("You did not select yes to the question, please try again after fixing your issue")

    answer = input("Have you added the file to CODE dir? (yes or no):- ")
    if answer != "yes":
        open_file(".")
        open_file("../MODS")
        exit("You did not select yes to the question, please try again after fixing your issue")

    answer = input("Have you added docstrings and comments? (yes or no):- ")
    if answer != "yes":
        open_file("../CONTRIBUTING.md")
        exit("You did not select yes to the question, please try again after fixing your issue")

    answer = input("Have you made sure you tested your code? (yes or no):- ")
    if answer != "yes":
        open_file("../_Test.py")
        exit("You did not select yes to the question, please try again after fixing your issue")

    answer = input(
        "Have you made sure you have no more than 1 feature per file and the features are non repeated? (yes or no):- ")
    if answer != "yes":
        open_file("../CONTRIBUTING.md")
        exit("You did not select yes to the question, please try again after fixing your issue")

    answer = input("Have you made sure you have added a comment to the TOP of your code with the flags you want to be "
                   "included in or not? (yes or no):- ")
    if answer != "yes":
        open_file("../CONTRIBUTING.md")
        exit("You did not select yes to the question, please try again after fixing your issue")

    answer = input("Have you made sure you DID NOT modify _wrapper.py unless told otherwise? (yes or no):- ")
    if answer != "yes":
        open_file("../_wrapper.py")
        exit("You did not select yes to the question, please try again after fixing your issue")

    # Usage
    files = check_current_files('.')
    print(files)
    answer = input("Nearly there! Does the list above include your added files? (yes or no):- ")
    if answer != "yes":
        print("Something went wrong! Please contact support, If you are sure the list doesnt contain the proper files")
        exit("You did not select yes to the question, please try again after fixing your issue")
    update_json_file('config.json', files)
    print("Great Job, Please tick the box in github PR request for completing steps in --dev")
