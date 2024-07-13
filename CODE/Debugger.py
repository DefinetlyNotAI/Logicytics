import ctypes
import importlib.metadata as pkg_resources
import re
import shutil
from pathlib import Path
import requests
from datetime import datetime
from local_libraries.Setups import *

time = datetime.now().strftime("%Y-%m-%d")


def read_version_file(file_path):
    """
    Read the content of the file and return it with leading/trailing whitespace removed.

    :param file_path: The path to the file to read.
    :return: The content of the file with leading/trailing whitespace removed or None if the file is not found.
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()
            return content.strip()  # Remove any leading/trailing whitespace
    except FileNotFoundError:
        print_colored(f"File {file_path} not found.", "red")
        return None


def compare_versions(source_version, target_version):
    """
    Compare two versions.

    :param source_version: The source version to compare.
    :param target_version: The target version to compare against.
    :return: True if the versions are the same, False otherwise.
    """
    if source_version == target_version:
        return True
    else:
        return False


def main_compare_logic():
    """
    Compares the version number of the downloaded Logicytics.version file with the version number of the original file.

    Returns:
        bool: True if the versions are the same, False otherwise.
    """
    # Download the Logicytics.version file from GitHub
    url = "https://raw.githubusercontent.com/DefinetlyNotAI/Logicytics/main/SYSTEM/Logicytics.version"
    response = requests.get(url)

    if response.status_code == 200:
        # Determine the current working directory
        current_working_dir = Path.cwd()

        # Save the file locally within the current working directory
        filename = "Logicytics.version"
        with open(current_working_dir / filename, "wb") as f:
            f.write(response.content)
    else:
        print_colored("Failed to download the file.", "red")
        exit(1)

    # Read the downloaded version number
    version_number_downloaded = read_version_file(current_working_dir / filename)

    # Compare the version number with the original file
    parent_directory = (
        Path(__file__).resolve().parent.parent
    )  # Adjust this path as needed
    original_file_path = parent_directory / "SYSTEM" / "Logicytics.version"
    version_number_original = read_version_file(original_file_path)

    if compare_versions(version_number_downloaded, version_number_original):
        (Path(current_working_dir / filename)).unlink(
            missing_ok=True
        )  # Safely delete the file even if it doesn't exist
        return True
    else:
        (Path(current_working_dir / filename)).unlink(
            missing_ok=True
        )  # Safely delete the file even if it doesn't exist
        return False


def check_python_versions():
    """
    Checks if Python and Python3 executables are found in the PATH.

    Writes an error message to DEBUG.md if neither executable is found.
    """
    try:
        result = subprocess.run(
            ["where", "python.exe"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.stdout.strip() == "":
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(
                    '<span style="color:red;">ERROR</span>: Python executable not found.<br><br>'
                )
            exit(1)
        else:
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(
                    f'<span style="color:green;">SYSTEM</span>: Python found in the PATH. Executable(s) located at {result.stdout.strip()}.<br><br>'
                )
    except subprocess.CalledProcessError:
        # Second attempt to find Python3
        try:
            result = subprocess.run(
                ["where", "python3"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.stdout.strip() == "":
                with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                    debug_file.write(
                        '<span style="color:red;">ERROR</span>: Python3 executable not found.<br><br>'
                    )
                exit(1)
            else:
                with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                    debug_file.write(
                        f'<span style="color:green;">SYSTEM</span>: Python3 found in the PATH. Executable located at {result.stdout.strip()}.<br><br>'
                    )
        except subprocess.CalledProcessError:
            # Both attempts failed, log the error
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(
                    "<span style=\"color:red;\">ERROR</span>: Neither 'python' nor 'python3' found in the PATH.<br><br>"
                )


def define_paths():
    """
    Defines the paths for version_file and structure_file.
    Returns:
        version_file_path: Path to the Logicytics.version file
        structure_file_path: Path to the Logicytics.structure file
    """
    version_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "SYSTEM",
        "Logicytics.version",
    )
    structure_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "SYSTEM",
        "Logicytics.structure",
    )
    return version_file_path, structure_file_path


def open_debug_file():
    """
    Opens the DEBUG.md file for writing content.
    """
    debug_file_path = os.path.join(os.getcwd(), "DEBUG.md")
    with open(debug_file_path, "a"):
        pass  # Placeholder for adding content to DEBUG.md


def check_vm():
    """
    Checks for virtual machine indicators in the system model information.
    Writes the result to DEBUG.md file.
    """
    command = 'systeminfo | findstr /C:"System Model"'

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
        )

        if re.search(r"VirtualBox|VBOX|VMWare", result.stdout):
            message = "Running in a virtual machine."
        else:
            message = "Not running in a virtual machine."

        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                f'<span style="color:green;">SYSTEM</span>: {message}<br><br>'
            )
    except subprocess.CalledProcessError as e:
        message = f"Error executing command: {e.stderr}"
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                f'<span style="color:red;">ERROR</span>: {message}<br><br>'
            )


def cmd_raw(command, check):
    """
    Execute a command and process the output based on the check parameter.

    Args:
        command (str): The command to be executed.
        check (str): The type of check to perform.

    Returns:
        str: The command output or an empty string based on the check.

    Raises:
        subprocess.CalledProcessError: If an error occurs during command execution.
    """
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
        )

        if check == "bool":
            output = result.stdout.strip()
            if output:
                return output
            else:
                return ""
        else:
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(
                    f'<span style="color:green;">SYSTEM</span>: {result.stdout}<br><br>'
                )
    except subprocess.CalledProcessError as e:
        if check == "bool":
            return ""
        else:
            message = f"Error executing command: {e.stderr}"
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(
                    f'<span style="color:red;">ERROR</span>: {message}<br><br>'
                )


def check_version_file(version_file_path):
    """
    Check the version file for existence and compare the version.

    Args:
        version_file_path (str): The path to the version file.

    Returns:
        None
    """
    if not os.path.exists(version_file_path):
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                '<span style="color:red;">ERROR</span>: Logicytics.version file not found.<br><br>'
            )
        exit(1)
    else:
        with open(version_file_path, "r") as version_file:
            version = version_file.read().strip()
            if main_compare_logic():
                with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                    debug_file.write(
                        f'<span style="color:green;">SYSTEM</span>: Version: {version}<br><br>'
                    )
            else:
                with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                    debug_file.write(
                        f'<span style="color:yellow;">WARNING</span>: Version: {version} is outdated and can be updated<br><br>'
                    )


def check_structure_file(structure_file_path):
    """
    Check the structure file for existence and validity.

    Args:
        structure_file_path (str): The path to the structure file.

    Returns:
        None
    """
    if not os.path.exists(structure_file_path):
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                '<span style="color:red;">ERROR</span>: Logicytics.structure file not found.<br><br>'
            )
    else:
        with open(structure_file_path, "r") as structure_file:
            for line in structure_file:
                line = line.strip()
                if line:
                    parent_dir = os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                    path = os.path.join(parent_dir, line[1:])
                    if os.path.exists(path):
                        with open(
                            os.path.join(os.getcwd(), "DEBUG.md"), "a"
                        ) as debug_file:
                            debug_file.write(
                                f'<span style="color:blue;">INFO</span>: Success: {path} exists.<br><br>'
                            )
                    else:
                        with open(
                            os.path.join(os.getcwd(), "DEBUG.md"), "a"
                        ) as debug_file:
                            debug_file.write(
                                f'<span style="color:red;">ERROR</span>: {path} does not exist.<br><br>'
                            )


def check_admin_privileges():
    """
    Check if the program is running with administrative privileges.
    Writes the result to DEBUG.md file.
    """
    try:
        subprocess.run(
            ["net", "session"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ctypes.windll.shell32.IsUserAnAdmin()
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                '<span style="color:blue;">INFO</span>: Running with administrative privileges.<br><br>'
            )
    except subprocess.CalledProcessError:
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                '<span style="color:yellow;">WARNING</span>: Not running with administrative privileges.<br><br>'
            )


def check_powershell_execution_policy():
    """
    Check the PowerShell execution policy.
    Writes the result to DEBUG.md file.
    """
    try:
        subprocess.run(
            ["powershell", "Get-ExecutionPolicy"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                '<span style="color:blue;">INFO</span>: PowerShell execution policy is set to Unrestricted.<br><br>'
            )
    except subprocess.CalledProcessError:
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                '<span style="color:red;">ERROR</span>: PowerShell execution policy is not set to Unrestricted.<br><br>'
            )


def check_library_exists(package_name):
    """
    Checks if a specific package is installed.

    :param package_name: Name of the package to check.
    :return: True if the package is installed, False otherwise.
    """
    try:
        pkg_resources.distribution(package_name)
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                f"<span style=\"color:blue;\">INFO</span>: The library '{package_name}' is installed.<br><br>"
            )
        return True
    except pkg_resources.PackageNotFoundError:
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                f"<span style=\"color:red;\">ERROR</span>: The package '{package_name}' is not installed.<br><br>"
            )
        return False


def check_library():
    """
    Checks the requirements.txt file for missing packages and logs errors to DEBUG.md file.

    This function constructs the full path to the requirements.txt file, checks if it exists,
    and then reads the file line by line. It ignores anything after '~=' in each line and checks
    if the package exists using the check_library_exists function. If a package is not found,
    an error message is logged to the DEBUG.md file.

    If an unexpected error occurs while processing the requirements' file, an error message is
    logged to the DEBUG.md file.

    Returns:
        None
    """
    # Construct the full path to the requirements.txt file
    full_path = os.path.join(os.getcwd(), "..", "requirements.txt")

    # Ensure the requirements.txt file exists
    if not os.path.exists(full_path):
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                f'<span style="color:red;">ERROR</span>: Requirements file {full_path} does not exist.<br><br>'
            )
        return

    # Attempt to read the requirements.txt file line by line
    try:
        with open(full_path, "r") as file:
            for line in file:
                # Ignore anything after '~='
                name = line.strip().split("~=", 1)[0]

                # Check if the package exists
                if not check_library_exists(name):
                    with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                        debug_file.write(
                            f"<span style=\"color:red;\">ERROR</span>: The package '{name}' is not installed.<br><br>"
                        )

    except Exception as e:
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                f'<span style="color:red;">ERROR</span>: An unexpected error occurred while processing the requirements file: {e}<br><br>'
            )


def move_debug_file(time):
    """
    Moves the 'DEBUG.md' file from the current directory to the 'ACCESS/LOGS' directory,
    renaming it to include a timestamp.

    This function first checks if the 'DEBUG.md' file exists in the current directory.
    If it does, it ensures the 'ACCESS/LOGS' directory exists before moving the file.
    It does not delete any other files or directories within 'ACCESS/LOGS'.

    Parameters:
        time (str): The timestamp to append to the filename.

    Returns:
        None
    """
    # Define the source and destination paths
    current_dir = os.getcwd()  # Get the current working directory
    source_path = os.path.join(current_dir, "DEBUG.md")
    parent_dir = os.path.dirname(current_dir)
    logs_dir = os.path.join(parent_dir, "ACCESS", "LOGS")

    # Check if the source file exists
    if os.path.exists(source_path):
        # Ensure the destination directory exists
        os.makedirs(logs_dir, exist_ok=True)

        # Generate the new filename with the timestamp
        new_filename = f"{time}_DEBUG.md"
        new_path = os.path.join(logs_dir, new_filename)

        # Move the file and rename it
        shutil.move(source_path, new_path)
        logger.info(f"Moved and renamed DEBUG.md to {new_path} successfully.")
    else:
        logger.error("DEBUG.md does not exist in the current directory.")


def create_directories():
    """
    Creates the necessary directories for the ACCESS, DATA, and LOGS directories.

    This function checks if the ACCESS directory exists in the parent directory and creates it if it doesn't.
    It also checks if the DATA and LOGS directories exist within the ACCESS directory and creates them if they don't.

    Returns:
        None
    """
    # Define the path for the ACCESS directory in the parent directory
    access_dir_path = os.path.join(os.path.dirname(os.getcwd()), "ACCESS")

    # Check if the ACCESS directory exists
    if not os.path.exists(access_dir_path):
        # Create the ACCESS directory
        os.makedirs(access_dir_path)

        # Now, assuming you want to create DATA and LOGS directories inside ACCESS,
        # define their paths relative to ACCESS directory
        data_dir_path = os.path.join(access_dir_path, "DATA")
        logs_dir_path = os.path.join(access_dir_path, "LOGS")

        # Check and create DATA directory
        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path)

        # Check and create LOGS directory
        if not os.path.exists(logs_dir_path):
            os.makedirs(logs_dir_path)


def main():
    """
    This function is the main entry point of the debugger. It performs a series of checks and commands to gather system information.
    """
    # Log the start of the debugging process
    logger.info("Running debugger...")

    # Create the necessary directories
    create_directories()

    # Define the paths to the version file and structure file
    version_file_path, structure_file_path = define_paths()

    # Open the debug file
    open_debug_file()

    # Check the version file for existence and compare the version
    check_version_file(version_file_path)

    # Check the library for compatibility
    check_library()

    # Check the structure file for integrity
    check_structure_file(structure_file_path)

    # Check for administrative privileges
    check_admin_privileges()

    # Check the PowerShell execution policy
    check_powershell_execution_policy()

    # Check for virtual machine indicators
    check_vm()

    # Check the Python versions
    check_python_versions()

    # Execute WMIC commands to gather system information
    cmd_raw("wmic bios get serialnumber", "null")
    cmd_raw("wmic computersystem get model", "null")
    cmd_raw("wmic computersystem get manufacturer", "null")

    # Check for VM drivers
    if cmd_raw('driverquery | findstr /C:"vmxnet"', "bool") == "":
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                '<span style="color:green;">SYSTEM</span>: No VM Drivers Found.<br><br>'
            )
    else:
        cmd_raw('driverquery | findstr /C:"vmxnet"', "null")

    # Execute systeminfo command to gather system information
    cmd_raw('systeminfo | findstr /C:"System Model" /C:"Manufacturer"', "null")

    # Move the debug file
    try:
        move_debug_file(time)
    except Exception as e:
        logger.error(f"Failed to move DEBUG.md. Reason: {e}")

    # Log the completion of the debugging process
    logger.info("Completed debug execution.")


if __name__ == "__main__":
    main()
