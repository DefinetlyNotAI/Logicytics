import os
import shutil
import zipfile
import subprocess


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
    if os.path.exists('../Access/Backup/backup.zip'):
        os.remove('../Access/Backup/backup.zip')

    # Zip the directory and move it to the backup location
    with zipfile.ZipFile(f'{name}.zip', 'w') as zip_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(str(file_path), start=os.getcwd())
                zip_file.write(str(file_path), arcname=relative_path)

    shutil.move('backup.zip', '../Access/Backup')


def update() -> None:
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
    subprocess.run(['git', 'pull'])
    os.chdir(current_dir)
