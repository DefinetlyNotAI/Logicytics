import ctypes
import os
import platform
import subprocess
import colorlog
from datetime import datetime

# The First 2 commands are part of the startup, the Last 2 commands are part of the cleanup process
# Debug.py, Legal.py, UAC.ps1, UACPY.py, Windows_Defender_Crippler.bat, APIGen.py are out of scope.
files = ["./CMD_Disabled_Bypass.py",
         "./Simple_Password_Miner.py",
         "./Browser_Policy_Miner.ps1",
         "./Window_Features_Lister.ps1",
         "./IP_Scanner.py",
         "./API_IP_Scraper.py",
         "./Device_Data.bat",
         "./Registry_miner.bat",
         "./Sys_Tools.py",
         "./Tree_Command.bat",
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


def check_file(name):
    # Get the absolute path to the parent directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # Construct the path to the SYSTEM directory within the parent directory
    system_dir_path = os.path.join(parent_dir, 'SYSTEM')

    # Construct the full path to the ToS.accept file within the SYSTEM directory
    file_path = os.path.join(system_dir_path, name)

    # Check if the file exists
    if os.path.exists(file_path) and name == "ToS.accept":
        logger.info(f"Found ToS.accept file at {file_path}")
        return True  # File found, exit the function
    elif os.path.exists(file_path) and name == "API.KEY":
        logger.info(f"Found API.KEY file at {file_path}")
        return True  # File found, exit the function
    else:
        logger.warning(f"{name} file not found in {system_dir_path}, quitting.")
        if name == "ToS.accept":
            execute_code(r".\Legal.py", "Script")
        else:
            execute_code(r".\APIGen.py", "Script")
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


def execute_code(script, type):
    if os.path.splitext(script)[1].lower() == '.ps1':
        unblock_command = f'powershell.exe -Command "Unblock-File -Path {script}"'
        subprocess.run(unblock_command, shell=True, check=True)
        logger.info("PS1 Script unblocked.")

    if type == "Command":
        command = f'powershell.exe -Command "& {script}"'
        process = subprocess.Popen(command, shell=True)
    elif type == "Script":
        command = f'powershell.exe -Command "& {script}"'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        logger.critical(f"Script Failure, Unknown entry type: {type}")
        exit(1)

    if os.path.splitext(script)[1].lower() == '.ps1' or os.path.splitext(script)[1].lower() == '.bat':
        # Initialize Identifier variable
        Identifier = None
        decoded_line = ""
        # Read the first word until :
        for line in iter(process.stdout.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if ':' in decoded_line:
                words = decoded_line.split(':', 1)
                Identifier = words[0].strip().upper()
                decoded_line = words[1].strip()
                break

        # Log the output based on the Identifier
        if Identifier == "INFO":
            logger.info(decoded_line)
        elif Identifier == "ERROR":
            logger.error(decoded_line)
        elif Identifier == "WARNING":
            logger.warning(decoded_line)
        else:
            logger.debug(decoded_line)
    elif os.path.splitext(script)[1].lower() == '.py':
        # Print the output in real-time
        for line in iter(process.stdout.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            print(decoded_line)
    else:
        for line in iter(process.stdout.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            logger.info(decoded_line)

    # Wait for the process to finish and get the final output/error
    stdout, _ = process.communicate()

    # Decode the output from bytes to string
    stdout = stdout.decode('utf-8') if stdout else ""

    # Return the output
    print()
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
            logger.info("Logicytics.py is running with administrative privileges.")
        else:
            logger.critical("Logicytics.py is running without administrative privileges.")
    else:
        logger.critical("This script is intended to run on Windows.")
        exit()


def main():
    timestamp("Started Logicytics at ")
    if check_file("ToS.accept") and check_file("API.KEY"):
        create_empty_data_directory()
        set_execution_policy()
        checks()
        timestamp("Completed Checks at ")
        print()
        for script_path in files:
            timestamp(f"Running Script {script_path} at ")
            execute_code(script_path, "Script")
        logger.info(r"Attempting to decypher the C:\\")
        try:
            execute_code(r"cd C:\\", "Command")
            execute_code(r"cipher -c", "Command")
        except Exception as e:
            logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
    timestamp("Completed Logicytics at ")
