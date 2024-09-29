from __lib_class import *

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


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
            if f.endswith(".exe") or f.endswith(".ps1")
        ]
        log.info("Available scripts:")
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
