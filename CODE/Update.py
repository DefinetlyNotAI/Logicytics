from pathlib import Path
import subprocess
import os
import requests


def read_version_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            return content.strip()  # Remove any leading/trailing whitespace
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None


def compare_versions(source_version, target_version):
    if source_version == target_version:
        print("The versions match.")
        return True
    else:
        print(target_version, "does not match with", source_version)
        return False


def compare_logic():
    url = 'https://raw.githubusercontent.com/DefinetlyNotAI/Logicytics/main/SYSTEM/Logicytics.version'
    response = requests.get(url)

    if response.status_code == 200:
        # Determine the current working directory
        current_working_dir = Path.cwd()

        # Save the file locally within the current working directory
        filename = 'Logicytics.version'
        with open(current_working_dir / filename, 'wb') as f:
            f.write(response.content)
    else:
        print("Failed to download the file.")
        exit(1)

    version_number_downloaded = read_version_file(current_working_dir / filename)

    # Now, compare the version number from the downloaded file to the original file in the SYSTEM directory
    # Assuming the original file exists in the parent directory under SYSTEM
    parent_directory = Path(__file__).resolve().parent.parent  # Adjust this path as needed
    original_file_path = parent_directory / 'SYSTEM' / 'Logicytics.version'

    # Read the original file's version number
    version_number_original = read_version_file(original_file_path)

    # Compare the versions
    if compare_versions(version_number_downloaded, version_number_original):
        (Path(current_working_dir / filename)).unlink(
            missing_ok=True)  # Safely delete the file even if it doesn't exist
        return True
    else:
        (Path(current_working_dir / filename)).unlink(
            missing_ok=True)  # Safely delete the file even if it doesn't exist
        return False


def update_project(repo_url, num_levels_up):
    """
    Updates the project from the specified GitHub repository.

    :param repo_url: URL of the GitHub repository.
    :param num_levels_up: Number of directory levels to move up from the current directory.
    """
    try:
        # Determine the local path by moving up the specified number of levels
        local_path = Path(os.getcwd()).parents[num_levels_up]

        # Initialize the directory as a Git repository if it doesn't exist yet
        if not local_path.exists():
            os.makedirs(local_path)
            subprocess.run(["git", "init"], cwd=str(local_path), check=True)

        # Clone the repository if it doesn't exist locally
        if not (local_path / ".git").exists():
            print("Cloning repository...")
            subprocess.run(["git", "clone", repo_url, str(local_path)], check=True)

        # Change directory to the local path within the script
        os.chdir(str(local_path))

        # Pull the latest changes
        print("Updating project...")
        subprocess.run(["git", "pull"], check=True)

        print("Project updated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


if compare_logic() is False:
    Continue = input("Do you want to update Logicytics? (y/n) ")
    if Continue == "y":
        update_project('https://github.com/DefinetlyNotAI/Logicytics.git', 2)
