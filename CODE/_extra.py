import os
import subprocess
import zipfile
import sys


def unzip(zip_path: str) -> None:
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


def menu() -> None:
    """
    Displays a menu of available executable scripts in the '../EXTRA/EXTRA' directory,
    prompts the user to select a script, and runs the selected script using PowerShell.

    Returns:
        None
    """
    try:
        files = [
            f
            for f in os.listdir("../EXTRA/EXTRA")
            if f.endswith(".exe") or f.endswith(".ps1")
        ]
        print("Available scripts:")
        for i, file in enumerate(files):
            print(f"{i+1}. {file}")
    except FileNotFoundError:
        print("Error: ../EXTRA/EXTRA directory not found - Did you unzip it using --unzip-extra flag?")
        exit(1)

    choice = int(input("Enter the number of your chosen script: "))
    if files[choice - 1] == "CMD.ps1":
        print("Redirecting to CMD.ps1...")
        subprocess.run(["powershell.exe", "../EXTRA/EXTRA/CMD.ps1"], check=True)
        command = input("Type the flags you want to execute: ")
        subprocess.run(
            [
                "powershell.exe",
                "../EXTRA/EXTRA/CMD.ps1",
                command.removeprefix(".\\CMD.ps1 "),
            ],
            check=True,
        )
    selected_file = files[choice - 1]
    subprocess.run(["powershell.exe", "../EXTRA/EXTRA/" + selected_file], check=True)
    exit(0)
