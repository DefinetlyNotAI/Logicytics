import subprocess
from pathlib import Path
import colorlog
import requests


# Configure colorlog
logger = colorlog.getLogger()
logger.setLevel(colorlog.DEBUG)  # Set the log level
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def read_version_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            return content.strip()  # Remove any leading/trailing whitespace
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        return None


def compare_versions(source_version, target_version):
    if source_version == target_version:
        logger.info("The versions match.")
        return True
    else:
        logger.warning("Version", target_version, "does not match with the latest version", source_version)
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
        logger.error("Failed to download the file.")
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


def update_local_repo():
    # Define the commands as a list of strings
    commands = [
        'git fetch origin',
        'git reset --hard origin/main'
    ]

    # Iterate over each command and execute it
    for command in commands:
        try:
            # Execute the command
            subprocess.run(command, shell=True, check=True)
            logger.info(f"Command '{command}' executed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute command '{command}'. Error: {e}")


if compare_logic() is False:
    Continue = input("Do you want to update Logicytics? (y/n) ")
    if Continue == "y":
        update_local_repo()
    else:
        exit(0)
