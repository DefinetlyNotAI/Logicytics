import shutil
from pathlib import Path
import os
import subprocess
import sys
from urllib.request import urlretrieve

# Define paths relative to the script's location
script_dir = Path(__file__).resolve().parent
script_path = os.path.realpath(__file__)


# Function to check if Python is installed
def check_python_installed():
    """
    Function to check if Python is installed.
    """
    try:
        subprocess.run(
            [sys.executable, "--version"], check=True, capture_output=True, text=True
        )
        print("Python is already installed.")
        return True
    except FileNotFoundError:
        print("Error: Python executable not found. Python might not be installed.")
        return False
    except subprocess.CalledProcessError or Exception as e:
        print(f"Error checking Python installation: {e}")
        return False


def download_python_installer(version="3.11.8"):
    """
    Downloads the specified version of the Python installer from the official Python website.

    :param version: The version of Python to download. Default is "3.11.8".
    :type version: str

    :return: The path to the downloaded Python installer.
    :rtype: str or None

    :raises Exception: If there is an error while downloading the installer.
    """
    url = f"https://www.python.org/ftp/python/{version}/python-{version}-amd64.exe"
    installer_path = os.path.join(os.getcwd(), "python-installer.exe")
    print(f"Attempting to download Python {version} installer...")
    try:
        urlretrieve(url, installer_path)
        print(f"Successfully downloaded to {installer_path}")
        return installer_path
    except Exception as e:
        print(f"Failed to download Python {version} installer: {e}")
        return None


def install_python(installer_path):
    """
    Initiates the Python installation using the provided installer path.

    :param installer_path: The path to the Python installer.
    :type installer_path: str

    :return: None
    :rtype: None
    """
    if installer_path is None:
        print("Installer path not found. Cannot proceed with installation.")
        return
    print("Initiating Python installation...")
    try:
        subprocess.run(
            [installer_path, "/quiet", "InstallAllUsers=1", "PrependPath=1"], check=True
        )
        print("Python installation completed successfully.")
    except Exception as e:
        print(f"Error during Python installation: {e}")


# Function to clone the repository if it doesn't already exist
def clone_repo_if_not_exists(repo_url):
    """
    Clones a repository if it doesn't already exist in the current directory.

    :param repo_url: The URL of the repository to clone.
    :type repo_url: str

    :return: None
    :rtype: None
    """
    repo_name = "Logicytics"
    if not any(
        path.name == repo_name and path.is_dir() for path in script_dir.iterdir()
    ):
        try:
            subprocess.run(["git", "clone", repo_url], cwd=script_dir, check=True)
            print(f"Repository {repo_name} cloned successfully.")
        except Exception as e:
            print(f"Failed to clone repository {repo_name}: {e}")
    else:
        print(
            f"The repository {repo_name} already exists in the current directory. Skipping clone."
        )


# Main function
def main():
    """
    A function that initiates the setup process, checks for Python installation,
    clones a repository, installs dependencies, prepares log directories, and moves logs.

    :return: None
    :rtype: None
    """
    print("Starting setup process...")

    # Check if Python is installed
    if not check_python_installed():
        installer_path = download_python_installer()
        install_python(installer_path)

    # Clone the repository into the current directory where Downloader.py is located
    clone_repo_if_not_exists("https://github.com/DefinetlyNotAI/Logicytics.git")

    # Assuming the SETUP directory exists within the cloned repository
    setup_dir = script_dir / "Logicytics" / "SETUP"
    if setup_dir.exists():
        os.chdir(setup_dir)

        # Run pip install
        try:
            result = subprocess.run(
                ["pip", "install", "-e", "."],
                capture_output=True,
                text=True,
                check=True,
            )
            with open("Download.log", "w") as log_file:
                log_file.write(result.stdout)

            print("Dependencies installed successfully.")
        except Exception as e:
            print(f"Error installing dependencies: {e}")

        # Prepare the logs directory
        logs_dir = script_dir / "Logicytics" / "ACCESS" / "LOGS"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Move Download.log to LOGS directory
        shutil.move(setup_dir / "Download.log", logs_dir)

        # Delete this file
        os.system(f'del "{script_path}"')
    else:
        print(f"The SETUP directory does not exist at {setup_dir}. Cannot proceed.")


if __name__ == "__main__":
    main()
