import ctypes
import os
import platform
import subprocess
import colorlog
from datetime import datetime


# The First command is part of the startup, the Last 2 commands are part of the cleanup process
# Debug.py, Legal.py, UAC.ps1, UACPY.py, Windows_Defender_Crippler.bat are out of scope.
files = ["./CMD_Disabled_Bypass.py",
         "./Browser_And_Policies_Miner.ps1",
         "./Window_Features_Lister.ps1",
         "./Antivirus_Finder.ps1",
         "./IP_Scanner.py",
         "./Device_Data.bat",
         "./Sys_Tools.py",
         "./Tree_Command.bat",
         "./Simple_Password_Miner.py",
         "./Copy_Media.py",
         "./System_Info_Grabber.py",
         "./Zipper.py",
         "./Clean.ps1"]

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


def timestamp(reason):
    # Get the current date and time
    now = datetime.now()

    # Format the timestamp as a string
    time = reason + now.strftime("%Y-%m-%d %H:%M:%S")

    # Print the formatted timestamp
    logger.debug(time)


def check_tos():
    # Get the absolute path to the parent directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # Construct the path to the SYSTEM directory within the parent directory
    system_dir_path = os.path.join(parent_dir, 'SYSTEM')

    # Construct the full path to the ToS.accept file within the SYSTEM directory
    file_path = os.path.join(system_dir_path, 'ToS.accept')

    # Check if the file exists
    if os.path.exists(file_path):
        logger.info(f"Found ToS.accept file at {file_path}")
        return True  # File found, exit the function
    else:
        logger.warning(f"ToS.accept file not found in {system_dir_path}")
        execute_code(r".\Legal.py")
    return False  # No file found


def create_empty_data_directory():
    """
    Creates an empty 'DATA' directory in the current working directory.
    """
    current_working_dir = os.getcwd()
    data_dir_path = os.path.join(current_working_dir, "DATA")

    try:
        os.makedirs(data_dir_path, exist_ok=True)
        logger.info(f"'{data_dir_path}' has been created.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def execute_code(script_path):
    if os.path.splitext(script_path)[1].lower() == '.ps1':
        unblock_command = f'powershell.exe -Command "Unblock-File -Path {script_path}"'
        subprocess.run(unblock_command, shell=True, check=True)
        logger.info("PS1 Script unblocked.")

    command = f'powershell.exe -Command "& {script_path}"'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if os.path.splitext(script_path)[1].lower() == '.ps1' or os.path.splitext(script_path)[1].lower() == '.bat':
        # Print the output in real-time
        for line in iter(process.stdout.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            logger.debug(decoded_line)
    elif os.path.splitext(script_path)[1].lower() == '.py':
        # Print the output in real-time
        for line in iter(process.stdout.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            print(decoded_line)

    # Wait for the process to finish and get the final output/error
    stdout, _ = process.communicate()

    # Decode the output from bytes to string
    stdout = stdout.decode('utf-8') if stdout else ""

    # Return the output and error messages
    return stdout, ""


def set_execution_policy():
    command = "powershell.exe Set-ExecutionPolicy Unrestricted -Scope CurrentUser -Force"
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info("Execution policy has been set to Unrestricted.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to set execution policy to Unrestricted. Error: {e}")


def checks():
    def is_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    if platform.system() == 'Windows':
        if is_admin():
            logger.info("code.py is running with administrative privileges.")
        else:
            logger.critical("code.py is running without administrative privileges.")
    else:
        logger.critical("This script is intended to run on Windows.")
        exit()


def main():
    timestamp("Started Logicytics at ")
    if check_tos():
        create_empty_data_directory()
        set_execution_policy()
        checks()
        timestamp("Completed Checks at ")
        for script_path in files:
            print()
            timestamp(f"Running Script {script_path} at ")
            execute_code(script_path)


if __name__ == "__main__":
    main()
    print()
    timestamp("Completed Logicytics at ")
    input("Press Any Button to continue: ")
