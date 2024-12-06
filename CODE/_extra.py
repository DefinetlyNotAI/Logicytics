import os
import subprocess

from logicytics import Log, DEBUG

"""
Files must be executed, and so can't ask or await user input.

Any Extra files in the EXTRA directory 
will be purely executed in the background and logged on the terminal.
"""

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


@log.function
def menu():
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
            if f.endswith((".exe", ".ps1", ".bat", ".cmd", ".py"))]
        print("Available scripts:")
        for i, file in enumerate(files):
            print(f"{i + 1}. {file}")
    except FileNotFoundError:
        log.error(
            "Error: ../EXTRA/EXTRA directory not found - Did you unzip it using --unzip-extra flag?"
        )
        exit(1)

    choice = int(input("Enter the number of your chosen script: "))
    if files[choice - 1] == "CMD.ps1":
        log.info("Redirecting to CMD.ps1...")
        subprocess.run(["powershell.exe", "../EXTRA/EXTRA/CMD.ps1"], check=True)
        command = input("Type the flags you want to Execute: ")
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


menu()
