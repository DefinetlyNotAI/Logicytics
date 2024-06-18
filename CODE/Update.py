import os
import subprocess
from pathlib import Path
import colorlog
import requests


def crash(error_id, function_no, error_content, type):
    """
    Ensure error_id and function_no are strings
    Prepare the data to write to the temporary files
    Write the name of the placeholder script to the temporary file
    Write the error message to the temporary file
    Write the name of the placeholder function to the temporary file
    Write the name of the placeholder language to the temporary file
    Write the name of the placeholder crash to the temporary file
    Write the type to the temporary file
    Open Crash_Reporter.py in a new shell window
    """
    # Ensure error_id and function_no are strings
    error_id = str(error_id)
    function_no = str(function_no)

    # Prepare the data to write to the temporary files
    script_name = os.path.basename(__file__)
    language = os.path.splitext(__file__)[1][1:]  # Extracting the language part

    # Write the name of the placeholder script to the temporary file
    with open("flag.temp", 'w') as f:
        f.write(script_name)

    # Write the error message to the temporary file
    with open("error.temp", 'w') as f:
        f.write(error_id)

    # Write the name of the placeholder function to the temporary file
    with open("function.temp", 'w') as f:
        f.write(function_no)

    # Write the name of the placeholder language to the temporary file
    with open("language.temp", 'w') as f:
        f.write(language)

    # Write the name of the placeholder crash to the temporary file
    with open("error_data.temp", 'w') as f:
        f.write(error_content)

    with open("type.temp", 'w') as f:
        f.write(type)

    # Open Crash_Reporter.py in a new shell window
    # Note: This command works for Command Prompt.
    # Adjust according to your needs.
    process = subprocess.Popen(r'powershell.exe -Command "& .\Crash_Reporter.py"', shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode('utf-8').strip()
        print(decoded_line)
    # Wait for the process to finish and get the final output/error
    stdout, _ = process.communicate()
    # Decode the output from bytes to string
    stdout = stdout.decode('utf-8') if stdout else ""
    print(stdout)


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


# Function to read the version number from a file
def read_version_file(file_path):
    """
    Read the version number from a file and remove any leading/trailing whitespace.

    Args:
        file_path (str): The path to the file containing the version number.

    Returns:
        str: The version number read from the file.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            return content.strip()  # Remove any leading/trailing whitespace
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        crash("FNF", "fun70", f"File not found at {file_path}", "error")
        return None


# Function to compare two versions
def compare_versions(source_version, target_version):
    """
    Compare two versions and log the result.

    Args:
        source_version (str): The version number to compare against.
        target_version (str): The version number to compare.

    Returns:
        bool: True if the versions match, False otherwise.
    """
    if source_version == target_version:
        logger.info(f"The versions match. Your version {target_version} matches with the latest version {source_version}")
        return True
    else:
        logger.warning(f"Version {target_version} does not match with the latest version {source_version}")
        return False


# Function to compare the downloaded and original versions
def compare_logic():
    """
    Compare the downloaded version with the original version and delete the downloaded file if versions match.

    Returns:
        bool: True if versions match, False otherwise.
    """
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
        crash("CE", "fun111", response.status_code, "crash")
        exit(1)

    version_number_downloaded = read_version_file(str(current_working_dir / filename))

    # Now, compare the version number from the downloaded file to the original file in the SYSTEM directory
    # Assuming the original file exists in the parent directory under SYSTEM
    parent_directory = Path(__file__).resolve().parent.parent  # Adjust this path as needed
    original_file_path = parent_directory / 'SYSTEM' / 'Logicytics.version'

    # Read the original file's version number
    version_number_original = read_version_file(str(original_file_path))

    # Compare the versions
    if compare_versions(version_number_downloaded, version_number_original):
        (Path(current_working_dir / filename)).unlink(
            missing_ok=True)  # Safely delete the file even if it doesn't exist
        return True
    else:
        (Path(current_working_dir / filename)).unlink(
            missing_ok=True)  # Safely delete the file even if it doesn't exist
        return False


# Function to update the local repository
def update_local_repo():
    """
    Update the local repository by fetching from origin and resetting to the main branch.
    """
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
            crash("EVE", "fun166", e.returncode, "crash")
            exit(1)


if compare_logic() is False:
    Continue = input("Do you want to update Logicytics? (y/n) ")
    if Continue == "y":
        update_local_repo()
    else:
        exit(0)
