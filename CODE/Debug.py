import ctypes
import re
import os
import subprocess
import colorlog
import importlib.metadata as pkg_resources

# Configure colorlog
logger = colorlog.getLogger()
logger.setLevel(colorlog.INFO)  # Set the log level
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


def check_python_versions():
    try:
        result = subprocess.run(['where', 'python.exe'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.stdout.strip() == '':
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write("<span style=\"color:red;\">ERROR</span>: Python executable not found.<br><br>")
            exit(1)
        else:
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(
                    f"<span style=\"color:green;\">SYSTEM</span>: Python found in the PATH. Executable(s) located at {result.stdout.strip()}.<br><br>")
    except subprocess.CalledProcessError:
        # Second attempt to find Python3
        try:
            result = subprocess.run(['where', 'python3'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.stdout.strip() == '':
                with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                    debug_file.write("<span style=\"color:red;\">ERROR</span>: Python3 executable not found.<br><br>")
                exit(1)
            else:
                with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                    debug_file.write(
                        f"<span style=\"color:green;\">SYSTEM</span>: Python3 found in the PATH. Executable located at {result.stdout.strip()}.<br><br>")
        except subprocess.CalledProcessError:
            # Both attempts failed, log the error
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(
                    "<span style=\"color:red;\">ERROR</span>: Neither 'python' nor 'python3' found in the PATH.<br><br>")


def delete_debug_file():
    debug_file_path = os.path.join(os.getcwd(), "DEBUG.md")
    if os.path.exists(debug_file_path):
        os.remove(debug_file_path)
        logger.debug("DEBUG.md file deleted. A new one will be created.")
    else:
        logger.debug("DEBUG.md file does not exist. A new one will be created.")


def define_paths():
    version_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SYSTEM",
                                     "Logicystics.version")
    structure_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SYSTEM",
                                       "Logicystics.structure")
    return version_file_path, structure_file_path


def open_debug_file():
    debug_file_path = os.path.join(os.getcwd(), "DEBUG.md")
    with open(debug_file_path, "a"):
        pass  # Placeholder for adding content to DEBUG.md


def check_vm():
    # Command to check for virtual machine indicators in the system model information
    command = "systeminfo | findstr /C:\"System Model\""

    try:
        # Execute the command and capture the output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)

        # Use regular expressions to check for virtual machine indicators
        if re.search(r"VirtualBox|VBOX|VMWare", result.stdout):
            message = "Running in a virtual machine."
        else:
            message = "Not running in a virtual machine."

        # Write the message to a file only once, after the check is complete
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(f"<span style=\"color:green;\">SYSTEM</span>: {message}<br><br>")
    except subprocess.CalledProcessError as e:
        # Handle errors from the subprocess call
        message = f"Error executing command: {e.stderr}"
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(f"<span style=\"color:red;\">ERROR</span>: {message}<br><br>")


def cmd_raw(command, check):
    # The command to be executed
    try:
        # Execute the command and capture the output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)

        if check == "bool":  # If check is "bool", return the command output or an empty string if it's empty
            output = result.stdout.strip()  # Remove leading/trailing whitespace
            if output:  # If the output is not empty
                return output
            else:
                return ""  # Return an empty string if the output is empty
        else:  # Write the command output to a file
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(f"<span style=\"color:green;\">SYSTEM</span>: {result.stdout}<br><br>")
    except subprocess.CalledProcessError as e:
        if check == "bool":  # If check is "bool", return an empty string or an error message
            return ""  # Return an empty string
        else:  # Handle errors from the subprocess call
            message = f"Error executing command: {e.stderr}"
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(f"<span style=\"color:red;\">ERROR</span>: {message}<br><br>")


def check_version_file(version_file_path):
    if not os.path.exists(version_file_path):
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write("<span style=\"color:red;\">ERROR</span>: Logicystics.version file not found.<br><br>")
        exit(1)
    else:
        with open(version_file_path, "r") as version_file:
            version = version_file.read().strip()
            with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                debug_file.write(f"<span style=\"color:green;\">SYSTEM</span>: Version: {version}<br><br>")


def check_structure_file(structure_file_path):
    if not os.path.exists(structure_file_path):
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write("<span style=\"color:red;\">ERROR</span>: Logicystics.structure file not found.<br><br>")
        exit(1)
    else:
        with open(structure_file_path, "r") as structure_file:
            for line in structure_file:
                line = line.strip()
                if line:  # Check if the line is not empty
                    # Replace {} with the parent working directory
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    path = os.path.join(parent_dir, line[1:])  # Remove the leading = and join with parent_dir
                    if os.path.exists(path):
                        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                            debug_file.write(
                                f"<span style=\"color:blue;\">INFO</span>: Success: {path} exists.<br><br>")
                    else:
                        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                            debug_file.write(f"<span style=\"color:red;\">ERROR</span>: {path} does not exist.<br><br>")


def check_uac_status():
    uac_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System"
    uac_value = "LocalAccountTokenBypassPolicy"
    uac_path = os.path.join(os.environ['WINDIR'], "System32", "config", uac_key)
    uac_value_path = os.path.join(uac_path, uac_value)
    if os.path.exists(uac_value_path):
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                "<span style=\"color:yellow;\">WARNING</span>: User Account Control (UAC) is enabled.<br><br>")
    else:
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                "<span style=\"color:yellow;\">WARNING</span>: User Account Control (UAC) is not enabled.<br><br>")


def check_admin_privileges():
    try:
        subprocess.run(["net", "session"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ctypes.windll.shell32.IsUserAnAdmin()
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write("<span style=\"color:blue;\">INFO</span>: Running with administrative privileges.<br><br>")
    except subprocess.CalledProcessError:
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                "<span style=\"color:yellow;\">WARNING</span>: Not running with administrative privileges.<br><br>")


def check_powershell_execution_policy():
    try:
        subprocess.run(["powershell", "Get-ExecutionPolicy"], check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                "<span style=\"color:blue;\">INFO</span>: PowerShell execution policy is set to Unrestricted.<br><br>")
    except subprocess.CalledProcessError:
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                "<span style=\"color:red;\">ERROR</span>: PowerShell execution policy is not set to Unrestricted.<br><br>")


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
                f"<span style=\"color:blue;\">INFO</span>: The library '{package_name}' is installed.<br><br>")
        return True
    except pkg_resources.PackageNotFoundError:
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                f"<span style=\"color:red;\">ERROR</span>: The package '{package_name}' is not installed.<br><br>")
        return False


def check_library():
    # Construct the full path to the requirements.txt file
    full_path = os.path.join(os.getcwd(), '..', 'requirements.txt')

    # Ensure the requirements.txt file exists
    if not os.path.exists(full_path):
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                f"<span style=\"color:red;\">ERROR</span>: Requirements file {full_path} does not exist.<br><br>")
        return

    # Attempt to read the requirements.txt file line by line
    try:
        with open(full_path, 'r') as file:
            for line in file:
                # Ignore anything after '~='
                name = line.strip().split('~=', 1)[0]

                # Check if the package exists
                if not check_library_exists(name):
                    with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
                        debug_file.write(
                            f"<span style=\"color:red;\">ERROR</span>: The package '{name}' is not installed.<br><br>")

    except Exception as e:
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                f"<span style=\"color:red;\">ERROR</span>: An unexpected error occurred while processing the requirements file: {e}<br><br>")


def main():
    logger.info("Starting debugger...")
    delete_debug_file()
    version_file_path, structure_file_path = define_paths()
    open_debug_file()
    check_version_file(version_file_path)
    check_library()
    check_structure_file(structure_file_path)
    check_uac_status()
    check_admin_privileges()
    check_powershell_execution_policy()
    check_vm()
    check_python_versions()
    cmd_raw("wmic bios get serialnumber", "null")
    cmd_raw("wmic computersystem get model", "null")
    cmd_raw("wmic computersystem get manufacturer", "null")
    if cmd_raw("driverquery | findstr /C:\"vmxnet\"", "bool") == "":
        with open(os.path.join(os.getcwd(), "DEBUG.md"), "a") as debug_file:
            debug_file.write(
                "<span style=\"color:green;\">SYSTEM</span>: No VM Drivers Found.<br><br>")
    else:
        cmd_raw("driverquery | findstr /C:\"vmxnet\"", "null")
    cmd_raw("systeminfo | findstr /C:\"System Model\" /C:\"Manufacturer\"", "null")
    logger.info("Completed debug execution.")


if __name__ == "__main__":
    main()
