import ctypes
import os
import platform
import random
import shutil
import subprocess
import colorlog
from datetime import datetime
import argparse

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


# Function to print usage instructions and examples
def show_usage_and_examples(parser):
    """
    Prints the usage instructions, examples, and descriptions of flags.
    """
    print("Usage: Logicytics.py [options]")
    print("\nOptions:")
    parser.print_help()


# Function to process command-line flags
def flagger():
    """
    Processes command-line flags, validates them, and returns a dictionary of flag values.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process command line flags.")

    # Define flags with descriptions
    flags = [
        ("--legacy", "Runs only the legacy scripts and required scripts that are made from the stable v1 project."),
        ("--unzip-extra", "Unzips all the extra files in the EXTRA directory == ONLY DO THIS ON YOUR OWN MACHINE, MIGHT TRIGGER ANTIVIRUS ==."),
        ("--backup",
         "Creates a backup of all the files in the CODE directory in a ZIP file in a new BACKUP Directory, == ONLY DO THIS ON YOUR OWN MACHINE =="),
        ("--restore", "Restores all files from the BACKUP directory, == ONLY DO THIS ON YOUR OWN MACHINE =="),
        ("--update",
         "Updates from the latest stable version in the GitHub repository, == ONLY DO THIS ON YOUR OWN MACHINE =="),
        ("--debug", "All Variables/Lists in the main project only are displayed with a DEBUG tag."),
        ("--extra", "Opens a menu for the EXTRA directory files == USE ON YOUR OWN RISK ==."),
        ("--onlypy", "Runs only the python scripts and required scripts."),
        ("--setup-only", "Runs all prerequisites then quits."),
        ("--setup", "Runs all prerequisites then Logicytics normally."),
        ("--minimum",
         "Runs the bare minimum where no external API or files are used, as well as running only quick programs."),
        ("--only-native", "Only runs PowerShell and Batch plus clean-up and setup script."),
        ("--debugger-only", "Runs the debugger then quits."),
        ("--debugger", "Runs the debugger then Logicytics."),
        ("--run", "Runs with default settings."),
        ("--mini-log", "Runs the log without feedback from the software."),
        ("--silent", "Runs without showing any log"),
        ("--shutdown", "After completing, ejects disk then shuts down the entire system."),
        ("--reboot", "After completing, ejects disk then restarts the entire system."),
        ("--bios", "After completing, ejects disk then restarts the entire system with instructions for you to follow.")
    ]

    compulsory_flags = ['onlypy', 'setup_only', 'setup', 'minimum', 'only_native', 'debugger_only', 'debugger', 'run',
                        'legacy', 'unzip_extra', 'backup', 'restore', 'update', 'extra']

    # Add flags to the parser
    for flag, desc in flags:
        parser.add_argument(flag, action="store_true", help=desc)

    # Parse arguments
    args = parser.parse_args()

    # Check for compulsory flags
    if not any(args.__dict__.get(flag, False) for flag in compulsory_flags):
        show_usage_and_examples(parser)
        return {}

    # Count the number of compulsory flags that are set
    set_compulsory_flags_count = sum(args.__dict__.get(flag, False) for flag in compulsory_flags)

    # Enforce that exactly one compulsory flag is set
    if set_compulsory_flags_count != 1:
        logger.critical("Exactly one of the compulsory flags must be set.")
        show_usage_and_examples(parser)
        exit(1)

    # Check for conflicting flags
    conflicts = {
        ('mini_log', 'silent'): "Both 'mini-log' and 'silent' are used.",
        ('shutdown', 'silent'): "Both 'shutdown' and 'silent' are used.",
        ('shutdown', 'reboot'): "Both 'shutdown' and 'reboot' are used.",
        ('shutdown', 'bios'): "Both 'shutdown' and 'bios' are used.",
        ('reboot', 'silent'): "Both 'reboot' and 'silent' are used.",
        ('reboot', 'bios'): "Both 'reboot' and 'bios' are used.",
        ('bios', 'silent'): "Both 'bios' and 'silent' are used.",
        ('debug', 'silent'): "Both 'debug' and 'silent' are used.",
        ('debug', 'mini_log'): "Both 'debug' and 'mini-log' are used.",
    }
    for conflict, msg in conflicts.items():
        if all(args.__dict__.get(flag, False) for flag in conflict):
            logger.critical(msg)
            exit(1)

    # Return the parsed arguments
    return vars(args)


def timestamp(reason: str) -> None:
    """
    Print a formatted timestamp with a given reason.

    Args:
        reason (str): The reason for the timestamp.

    Returns:
        None
    """
    # Get the current date and time
    now = datetime.now()

    # Format the timestamp as a string
    time = f"{reason}{now.strftime('%Y-%m-%d %H:%M:%S')}"

    # Print the formatted timestamp
    logger.debug(time)


def check_file(name):
    """
    Check if a specific file exists within the SYSTEM directory.

    Args:
        name (str): The name of the file to check.

    Returns:
        bool: True if the file is found, False otherwise.
    """
    # Get the absolute path to the parent directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # Construct the path to the SYSTEM directory within the parent directory
    system_dir_path = os.path.join(parent_dir, 'SYSTEM')

    # Construct the full path to the file within the SYSTEM directory
    file_path = os.path.join(system_dir_path, name)

    # Check if the file exists
    if os.path.exists(os.path.join(system_dir_path, "DEV.pass")):
        # Log info if the file is found
        logger.info(f"Dev File Found, ignoring checks for {name}")
        return True  # File found, exit the function
    elif os.path.exists(file_path) and name == "ToS.accept":
        # Log info if ToS.accept file is found
        logger.info(f"Found ToS.accept file at {file_path}")
        return True  # File found, exit the function
    elif os.path.exists(file_path) and name == "API-IP.KEY":
        # Log's info if the API-IP.KEY file is found
        logger.info(f"Found API-IP.KEY file at {file_path}")
        return True  # File found, exit the function
    else:
        # Log a warning if the file is not found
        logger.warning(f"{name} file not found in {system_dir_path}, quitting.")
        # Execute the corresponding code based on the file name
        if name == "ToS.accept":
            execute_code(r".\Legal.py", "Script", "")
        else:
            execute_code(r".\APIGen.py", "Script", "")
    return False


def create_empty_data_directory(Silent):
    """
    Creates an empty 'DATA' directory in the current working directory.

    Args:
        Silent (str): A parameter to control the logging output.

    Returns:
        None
    """
    current_working_dir = os.getcwd()
    data_dir_path = os.path.join(current_working_dir, "DATA")

    try:
        os.makedirs(data_dir_path, exist_ok=True)
        if Silent != "Silent":
            logger.info(f"'{data_dir_path}' has been created.")
        if Silent == "Debug":
            logger.debug("Current working directory: " + current_working_dir)
            logger.debug("Data directory path: " + data_dir_path)
            logger.debug(f"Data directory contents: {os.listdir(data_dir_path)}")
            logger.debug(f"Data directory exists: {os.path.exists(data_dir_path)}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def install_libraries(command: str) -> None:
    """
    Executes a command to install libraries and prints the output.

    Args:
        command (str): The command to execute.

    Returns:
        None
    """
    try:
        # Execute the command and capture the output
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
        stdout, stderr = process.communicate()

        # Check for errors
        if process.returncode != 0:
            # Log the error and the standard error
            logger.error(f"Error occurred while executing command: {command}")
            return

        # Process the output and log each line
        for line in stdout.decode('utf-8').splitlines():
            logger.info(line)

    except Exception as e:
        # Log any exceptions that occur
        logger.error(f"An error occurred: {e}")


def execute_code(script: str, type: str, silence: str) -> tuple[str, str]:
    """
    Executes a script and logs the output based on the script type and silence level.

    Args:
        script (str): The path to the script to execute.
        type (str): The type of script to execute. It Can be either "Command" or "Script".
        silence (str): The level of silence. Can be either "Silent" or any other value.

    Returns:
        Tuple[str, str]: A tuple containing the output of the script and an empty string.
    """
    global words, unblock_command

    # Unblock the script if it is a PowerShell script
    if os.path.splitext(script)[1].lower() == '.ps1':
        unblock_command = f'powershell.exe -Command "Unblock-File -Path {script}"'
        subprocess.run(unblock_command, shell=True, check=True)
        if silence != "Silent":
            logger.info("PS1 Script unblocked.")

    if silence == "Debug":
        logger.debug("Unblocking script: " + unblock_command)
        logger.debug("Script: " + script)
        logger.debug("Script Type: " + type)

    # Execute the script based on the type
    if type == "Command":
        command = f'powershell.exe -Command "& {script}"'
        process = subprocess.Popen(command, shell=True)
    elif type == "Script":
        command = f'powershell.exe -Command "& {script}"'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        logger.critical(f"Script Failure, Unknown entry type: {type}")
        exit(1)

    if silence == "Debug":
        logger.debug(f"Process: {process}")
        logger.debug("Command: " + command)
        logger.debug("Script Type: " + type)

    # Log the output based on the script type and silence level
    if silence != "Silent":
        if os.path.splitext(script)[1].lower() in ['.ps1', '.bat']:
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

                if silence == "Debug":
                    logger.debug("Decoded Line: " + decoded_line)
                    logger.debug("Identifier: " + Identifier)
                    logger.debug(f"Words: {words}")
                    logger.debug(f"Line Value: {line}")
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


def set_execution_policy(Silent: str) -> None:
    """
    Sets the PowerShell execution policy to Unrestricted for the current process.

    Args:
        Silent (str): If "Silent", suppresses logging of success or failure.

    Raises:
        subprocess.CalledProcessError: If there was an error setting the execution policy.

    Returns:
        None
    """
    # Define the command to set the execution policy
    command = 'powershell.exe -Command "Set-ExecutionPolicy Unrestricted -Scope Process -Force"'

    if Silent == "Debug":
        logger.debug("Command: " + command)

    try:
        # Execute the command
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if Silent == "Debug":
            logger.debug(f"Result: {result}")

        # Check the output for success
        if 'SUCCESS' in result.stdout:
            if Silent != "Silent":
                logger.info("Execution policy has been set to Unrestricted.")
        else:
            logger.error("An error occurred while trying to set the execution policy.")

    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while trying to set the execution policy: {e}")


def checks():
    """
    Checks if the script is running with administrative privileges on Windows.

    Raises:
        SystemExit: If the script is not running on Windows or if it's not running with administrative privileges.
    """

    def is_admin():
        """
        Checks if the script is running with administrative privileges on Windows.

        Returns:
            bool: True if the script is running with administrative privileges, False otherwise.
        """
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    if platform.system() == 'Windows':
        if is_admin():
            logger.info("Logicytics.py is running with administrative privileges.")
            return True
        else:
            logger.critical("Logicytics.py is running without administrative privileges.")
            return False
    else:
        logger.critical("This script is intended to run on Windows.")
        return False


def parse_true_values(flagger_result):
    """
    Parses the given dictionary and returns a list of keys whose values are True.

    Args:
        flagger_result (dict): The dictionary to parse.

    Returns:
        list: A list of keys with True values.
    """
    # Initialize an empty list to store keys with True values
    true_keys = []

    # Iterate over the items in the dictionary
    for key, value in flagger_result.items():
        # Check if the value is True
        if value:
            # If True, append the key to the list
            true_keys.append(key)

    # Return the list of keys with True values
    return true_keys


def print_random_logo():
    """
    Prints the content of a random logo file from the './logo' directory.
    If the directory or logo file is not found, appropriate messages are displayed.

    Raises:
        IOError: If there is an error reading the logo file.
    """
    logo_dir = './logo'

    # Check if the logo directory exists
    if not os.path.exists(logo_dir):
        logger.warning(f"The directory '{logo_dir}' does not exist, attempting to create it.")
        return

    # Create the logo directory if it doesn't exist
    os.makedirs(logo_dir, exist_ok=True)

    # Get a list of all .txt files in the logo directory
    logo_files = [f for f in os.listdir(logo_dir) if f.endswith('.txt')]

    # Check if there are any .txt files in the logo directory
    if not logo_files:
        logger.critical("No .txt files found in the logo directory.")
        exit(1)

    # Choose a random logo file
    random_file = random.choice(logo_files)
    file_path = os.path.join(logo_dir, random_file)

    try:
        # Read and print the content of the random logo file
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f.read())
    except IOError as e:
        logger.warning(f"An error occurred while trying to read the file {file_path}: {e}")


def logicytics(log, quit_var):
    """
    Executes different actions based on the 'log' and 'quit_var' parameters as well as preset 'run' parameter.
    """
    current_dir = os.getcwd()  # Get the current working directory
    directory_path = os.path.join(current_dir, "DATA")  # Construct the full path

    try:
        # Check if the specified path exists
        if os.path.exists(directory_path):
            # Remove the directory and all its contents
            shutil.rmtree(directory_path)
            print(f"Directory {directory_path} has been deleted.")
    except Exception as e:
        print(f"An error occurred while trying to delete the directory: {e}")

    if log == "normal":
        timestamp("Started Logicytics at ")
        print_random_logo()
        if check_file("ToS.accept") and check_file("API-IP.KEY") and checks():
            set_execution_policy("")
            create_empty_data_directory("")
            timestamp("Completed Checks at ")
            print()
            for script_path in files:
                timestamp(f"Running Script {script_path} at ")
                execute_code(script_path, "Script", "")

            if quit_var == "shutdown":
                os.system('shutdown /s /t 0')
            elif quit_var == "restart":
                os.system('shutdown /r /t 0')
            elif quit_var == "bios":
                logger.warning(
                    "Sorry, this is a impossible task, we will restart the device for you in 10 seconds, and you have to mash the (esc) or (f10) button, thanks for understanding")
                os.system('shutdown /r /t 10')
            elif quit_var == "normal":
                print()
            else:
                logger.critical("No Valid Flag")
                exit(1)
        else:
            logger.critical("Unexpected Error Occurred due to Missing permissions while Checking")
            exit(1)

    elif log == "debug":
        timestamp("Started Logicytics at ")
        print_random_logo()
        if check_file("ToS.accept") and check_file("API-IP.KEY") and checks():
            set_execution_policy("Debug")
            create_empty_data_directory("Debug")
            timestamp("Completed Checks at ")
            print()
            for script_path in files:
                timestamp(f"Running Script {script_path} at ")
                execute_code(script_path, "Script", "Debug")

            if quit_var == "shutdown":
                os.system('shutdown /s /t 0')
            elif quit_var == "restart":
                os.system('shutdown /r /t 0')
            elif quit_var == "bios":
                logger.warning(
                    "Sorry, this is a impossible task, we will restart the device for you in 10 seconds, and you have to mash the (esc) or (f10) button, thanks for understanding")
                os.system('shutdown /r /t 10')
            elif quit_var == "normal":
                print()
            else:
                logger.critical("No Valid Flag")
                exit(1)
        else:
            logger.critical("Unexpected Error Occurred due to Missing permissions while Checking")
            exit(1)

    elif log == "mini_log":
        timestamp("Started Logicytics at ")
        if check_file("ToS.accept") and check_file("API-IP.KEY") and checks():
            set_execution_policy("Silent")
            create_empty_data_directory("Silent")
            timestamp("Completed Checks at ")
            print()
            for script_path in files:
                timestamp(f"Running Script {script_path} at ")
                execute_code(script_path, "Script", "Silent")

            if quit_var == "shutdown":
                os.system('shutdown /s /t 0')
            elif quit_var == "restart":
                os.system('shutdown /r /t 0')
            elif quit_var == "bios":
                logger.warning(
                    "Sorry, this is a impossible task, we will restart the device for you in 10 seconds, and you have to mash the (esc) or (f10) button, thanks for understanding")
                os.system('shutdown /r /t 10')
            elif quit_var == "normal":
                print()
            else:
                logger.critical("No Valid Flag")
                exit(1)
        else:
            logger.critical("Unexpected Error Occurred due to Missing permissions while Checking")
            exit(1)

    elif log == "silent":
        if check_file("ToS.accept") and check_file("API-IP.KEY"):
            set_execution_policy("Silent")
            create_empty_data_directory("Silent")
            for script_path in files:
                execute_code(script_path, "Script", "Silent")
    else:
        logger.critical("No Valid Flag")
        exit(1)


keys = parse_true_values(flagger())

run = "normal"
log = "normal"
quit_var = "normal"

# Initialize variables with default values
run_actions = {
    'onlypy': 'onlypy',
    'setup_only': 'setup_only',
    'setup': 'setup',
    'minimum': 'minimum',
    'only_native': 'only_native',
    'debugger_only': 'debugger_only',
    'debugger': 'debugger',
    'run': 'run',
    'legacy': 'legacy',
    'unzip_extra': 'unzip_extra',
    'backup': 'backup',
    'restore': 'restore',
    'update': 'update',
    'extra': 'extra',
}

log_actions = {
    'mini_log': 'mini_log',
    'silent': 'silent',
    'debug': 'debug',
}

quit_actions = {
    'shutdown': 'shutdown',
    'reboot': 'reboot',
    'bios': 'bios',
}

# Update variables based on keys found
for key in keys:
    if key in run_actions:
        run = run_actions[key]
    if key in log_actions:
        log = log_actions[key]
    if key in quit_actions:
        quit_var = quit_actions[key]

files = []
Continue = ""

if run == "run":
    # The First 2 commands are part of the startup, the Last 2 commands are part of the cleanup process
    # Debugger.py, Legal.py, UAC.ps1, UACPY.py, Windows_Defender_Crippler.bat, APIGen.py are out of scope.
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

if run == "onlypy":
    files = ["./CMD_Disabled_Bypass.py",
             "./Simple_Password_Miner.py",
             "./IP_Scanner.py",
             "./API_IP_Scraper.py",
             "./Sys_Tools.py",
             "./Copy_Media.py",
             "./System_Info_Grabber.py",
             "./Zipper.py",
             "./Clean.ps1"]

if run == "setup_only":
    install_libraries("pip install -r../requirements.txt")
    files = ["./CMD_Disabled_Bypass.py",
             "./UACPY.py",
             "./Window_Defender_Crippler.bat"]

if run == "setup":
    install_libraries("pip install -r../requirements.txt")
    files = ["./CMD_Disabled_Bypass.py",
             "./UACPY.py",
             "./Window_Defender_Crippler.bat",
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

if run == "minimum":
    files = ["./CMD_Disabled_Bypass.py",
             "./Simple_Password_Miner.py",
             "./Browser_Policy_Miner.ps1",
             "./Window_Features_Lister.ps1",
             "./IP_Scanner.py",
             "./Device_Data.bat",
             "./Registry_miner.bat",
             "./Tree_Command.bat",
             "./Copy_Media.py",
             "./System_Info_Grabber.py",
             "./Zipper.py",
             "./Clean.ps1"]

if run == "only_native":
    files = ["./CMD_Disabled_Bypass.py",
             "./Simple_Password_Miner.py",
             "./Browser_Policy_Miner.ps1",
             "./Window_Features_Lister.ps1",
             "./Device_Data.bat",
             "./Registry_miner.bat",
             "./Tree_Command.bat",
             "./Zipper.py",
             "./Clean.ps1"]

if run == "debugger_only":
    files = ["./Debugger.py"]

if run == "debugger":
    files = ["./Debugger.py",
             "./CMD_Disabled_Bypass.py",
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

if run == "legacy":
    files = ["./CMD_Disabled_Bypass.py",
             "./Simple_Password_Miner.py",
             "./Device_Data.bat",
             "./Tree_Command.bat",
             "./Copy_Media.py",
             "./Zipper.py",
             "./Clean.ps1"]

if run == "unzip_extra":
    Continue = input(
        "This flag will unzip all the extra files in the EXTRA directory. Only do this if you know what you are doing and want to use the extra feature, Do this only on your own machine, Might trigger antivirus, Best to backup your files first. Press `Enter` to continue, press anything else to cancel... ")
    if Continue == "":
        files = ["./Unzip_Extra.py"]

if run == "backup":
    Continue = input(
        "This flag will zip all the files in the CODE directory for backup uses. Do this only on your own machine, Press `Enter` to continue, press anything else to cancel... ")
    if Continue == "":
        files = ["./Backup.py"]

if run == "restore":
    Continue = input(
        "This flag will unzip the backed up files in the BACKUP directory. Used to restore old files in case of breaking or unexpected errors, a menu might appear if more than 1 Backup is found, Do this only on your own machine. Press `Enter` to continue, press anything else to cancel... ")
    if Continue == "":
        files = ["./Restore.py"]

if run == "update":
    Continue = input(
        "This flag will update this project from the latest version in the GitHub repository. Do this only on your own machine as you may need to download extra features, Best to backup your files first. Press `Enter` to continue, press anything else to cancel... ")
    if Continue == "":
        files = ["./Update.py"]

if run == "extra":
    Continue = input(
        "This flag will open a menu to run any of the extra files in the EXTRA directory. Only do this if you know what you are doing and want to use the extra feature, Might trigger antivirus. Press `Enter` to continue, press anything else to cancel... ")
    if Continue == "":
        files = ["./Extra_Menu.py"]

if log == "debug":
    logger.debug("Log: " + run)
    for file in files:
        logger.debug("File's from List: " + file)
    logger.debug("Quit Type: " + quit_var)
    logger.debug(f"Keys to be used: {keys}")
    logger.debug("Continue value: " + Continue)

logicytics(log, quit_var)
