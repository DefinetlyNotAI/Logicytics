import shutil

from __lib_class import *


def backup(directory: str, name: str) -> None:
    """
    Creates a backup of a specified directory by zipping its contents and moving it to a designated backup location.

    Args:
        directory (str): The path to the directory to be backed up.
        name (str): The name of the backup file.

    Returns:
        None
    """
    # Check if backup exists, delete it if so
    if os.path.exists(f"../ACCESS/BACKUP/{name}.zip"):
        os.remove(f"../ACCESS/BACKUP/{name}.zip")

    # Zip the directory and move it to the backup location
    with zipfile.ZipFile(f"{name}.zip", "w") as zip_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(str(file_path), start=os.getcwd())
                zip_file.write(str(file_path), arcname=relative_path)

    shutil.move(f"{name}.zip", "../ACCESS/BACKUP")


def update() -> tuple[str, str]:
    """
    Updates the repository by pulling the latest changes from the remote repository.

    This function navigates to the parent directory, pulls the latest changes using Git,
    and then returns to the current working directory.

    Returns:
        str: The output from the git pull command.
    """
    # Check if git command is available
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        return "Git is not installed or not available in the PATH.", "error"

    # Check if the project is a git repository
    if not os.path.exists(os.path.join(os.getcwd(), ".git")):
        return "This project is not a git repository. The update flag uses git.", "error"

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    output = subprocess.run(["git", "pull"], capture_output=True).stdout.decode()
    os.chdir(current_dir)
    return output, "info"
