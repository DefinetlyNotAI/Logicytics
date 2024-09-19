import shutil
from __lib_class import *
log_health = Log(debug=DEBUG)
log_health_funcs = {
    "INFO": log_health.info,
    "WARNING": log_health.warning,
    "ERROR": log_health.error,
    "CRITICAL": log_health.critical,
    None: log_health.debug,
}

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


def update() -> str:
    """
    Updates the repository by pulling the latest changes from the remote repository.

    This function navigates to the parent directory, pulls the latest changes using Git,
    and then returns to the current working directory.

    Returns:
        None
    """
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    output = subprocess.run(["git", "pull"]).stdout.decode()
    os.chdir(current_dir)
    return output
