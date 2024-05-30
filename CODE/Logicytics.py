import ctypes
import os
import platform
import subprocess
import colorlog
from datetime import datetime
import argparse
import sys

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


def show_usage_and_examples():
    """Prints the usage instructions, examples, and descriptions of flags."""
    print("Usage: Logicytics.py [options]")
    print("\nOptions:")
    parser.print_help()


def flagger():
    # Create the parser
    global parser
    parser = argparse.ArgumentParser(description="Process command line flags.")

    # Add the flags with descriptions
    parser.add_argument("--onlypy", action="store_true", help="Runs only the python scripts and required scripts.")
    parser.add_argument("--setup-only", action="store_true", help="Runs all prerequisites then quits.")
    parser.add_argument("--setup", action="store_true", help="Runs all prerequisites then Logicytics normally.")
    parser.add_argument("--minimum", action="store_true",
                        help="Runs the bare minimum where no external API or files are used, as well as running only quick programs.")
    parser.add_argument("--only-native", action="store_true",
                        help="Only runs PowerShell and Batch plus clean-up and setup script.")
    parser.add_argument("--debug-only", action="store_true", help="Runs the debugger then quits.")
    parser.add_argument("--debug", action="store_true", help="Runs the debugger then Logicytics.")
    parser.add_argument("--run", action="store_true", help="Runs with default settings.")
    parser.add_argument("--mini-log", action="store_true", help="Runs the log without feedback from the software.")
    parser.add_argument("--silent", action="store_true",
                        help="Runs without showing any log")
    parser.add_argument("--shutdown", action="store_true",
                        help="After completing, ejects disk then shuts down the entire system.")
    parser.add_argument("--reboot", action="store_true",
                        help="After completing, ejects disk then restarts the entire system.")
    parser.add_argument("--bios", action="store_true",
                        help="After completing, ejects disk then restarts the entire system with instructions for you to follow.")

    # Check if no arguments were provided and show usage
    if len(sys.argv) == 1:
        show_usage_and_examples()
        exit(0)

    try:
        # Attempt to parse the arguments
        args = parser.parse_args()
    except SystemExit:
        # This exception is raised when no arguments are provided
        show_usage_and_examples()
        exit(1)

    # Set variables based on parsed arguments
    flags = {
        'onlypy': args.onlypy,
        'setup_only': args.setup_only,
        'setup': args.setup,
        'minimum': args.minimum,
        'only_native': args.only_native,
        'debug_only': args.debug_only,
        'debug': args.debug,
        'run': args.run,
        'mini_log': args.mini_log,
        'silent': args.silent,
        'shutdown': args.shutdown,
        'reboot': args.reboot,
        'bios': args.bios
    }

    # Define compulsory flags
    compulsory_flags = ['onlypy', 'setup_only', 'setup', 'minimum', 'only_native',
                        'debug_only', 'debug', 'run']

    # Check for compulsory flags
    if not any(flags.get(flag, False) for flag in compulsory_flags):
        logger.critical(
            "Error: One of these flags: --onlypy, --setup-only, --setup, --minimum, --only-native, --debug-only, --debug, --run must be used.")
        return

    # Check for combinations of true flags and output an error message if any condition is met
    if flags['mini_log'] and flags['silent']:
        logger.critical("Error: Both 'mini-log' and 'silent' are used.")
        exit()
    elif flags['shutdown'] and flags['silent']:
        logger.critical("Error: Both 'shutdown' and 'silent' are used.")
        exit()
    elif flags['shutdown'] and flags['reboot']:
        logger.critical("Error: Both 'shutdown' and 'reboot' are used.")
        exit()
    elif flags['shutdown'] and flags['bios']:
        logger.critical("Error: Both 'shutdown' and 'bios' are used.")
        exit()
    elif flags['reboot'] and flags['silent']:
        logger.critical("Error: Both 'reboot' and 'silent' are used.")
        exit()
    elif flags['reboot'] and flags['bios']:
        logger.critical("Error: Both 'reboot' and 'bios' are used.")
        exit()
    elif flags['bios'] and flags['silent']:
        logger.critical("Error: Both 'bios' and 'silent' are used.")
        exit()
    else:
        # Define compulsory flags
        compulsory_flags = ['onlypy', 'setup_only', 'setup', 'minimum', 'only_native',
                            'debug_only', 'debug', 'run']

        # Check for compulsory flags
        if not any(flags.get(flag, False) for flag in compulsory_flags):
            logger.critical(
                "Error: One of these flags: --onlypy, --setup-only, --setup, --minimum, --only-native, --debug-only, --debug, --run must be used.")
            return

        return flags


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
            execute_code(r".\Legal.py", "Script", "")
        else:
            execute_code(r".\APIGen.py", "Script", "")
    return False  # No file found


def create_empty_data_directory(Silent):
    """
    Creates an empty 'DATA' directory in the current working directory.
    """
    current_working_dir = os.getcwd()
    data_dir_path = os.path.join(current_working_dir, "DATA")

    try:
        os.makedirs(data_dir_path, exist_ok=True)
        if Silent != "Silent":
            logger.info(f"'{data_dir_path}' has been created.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def execute_code(script, type, silence):
    global words
    if os.path.splitext(script)[1].lower() == '.ps1':
        unblock_command = f'powershell.exe -Command "Unblock-File -Path {script}"'
        subprocess.run(unblock_command, shell=True, check=True)
        if silence != "Silent":
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

    if silence != "Silent":
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


def set_execution_policy(Silent):
    # Define the command to set the execution policy
    command = 'powershell.exe -Command "Set-ExecutionPolicy Unrestricted -Scope Process -Force"'

    try:
        # Execute the command
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check the output for success
        if 'SUCCESS' in result.stdout:
            if Silent != "Silent":
                logger.info("Execution policy has been set to Unrestricted.")
        else:
            logger.error(f"Failed to set execution policy to Unrestricted. Output: {result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while trying to set the execution policy: {e}")


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


def parse_true_values(flagger_result):
    """
    Parses the given dictionary and returns a list of keys whose values are True.

    :param flagger_result: The dictionary to parse.
    :return: A list of keys with True values.
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


def logicytics(log, quit_var):
    if log == "normal":
        timestamp("Started Logicytics at ")
        if check_file("ToS.accept") and check_file("API.KEY"):
            create_empty_data_directory("")
            set_execution_policy("")
            checks()
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
                logger.debug(
                    "Sorry, this is a impossible task, we will restart the device for you in 10 seconds, and you have to mash the (esc) button, thanks for understanding")
                os.system('shutdown /r /t 10')
            elif quit_var == "normal":
                print()
            else:
                logger.critical("No Valid Flag")

    elif log == "mini_log":
        timestamp("Started Logicytics at ")
        if check_file("ToS.accept") and check_file("API.KEY"):
            create_empty_data_directory("Silent")
            set_execution_policy("Silent")
            checks()
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
                logger.debug(
                    "Sorry, this is a impossible task, we will restart the device for you in 10 seconds, and you have to mash the (esc) button, thanks for understanding")
                os.system('shutdown /r /t 10')
            elif quit_var == "normal":
                print()
            else:
                logger.critical("No Valid Flag")

    elif log == "silent":
        if check_file("ToS.accept") and check_file("API.KEY"):
            create_empty_data_directory("Silent")
            set_execution_policy("Silent")
            print()
            for script_path in files:
                execute_code(script_path, "Script", "Silent")
    else:
        logger.critical("No Valid Flag")


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
    'debug_only': 'debug_only',
    'debug': 'debug',
    'run': 'run'
}

log_actions = {
    'mini_log': 'mini_log',
    'silent': 'silent'
}

quit_actions = {
    'shutdown': 'shutdown',
    'reboot': 'reboot',
    'bios': 'bios'
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

if run == "run":
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
    execute_code("pip install -r../requirements.txt", "Command", "")
    files = ["./CMD_Disabled_Bypass.py",
             "./UACPY.py",
             "./Window_Defender_Crippler.bat"]

if run == "setup":
    execute_code("pip install -r../requirements.txt", "Command", "")
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

if run == "debug_only":
    files = ["./Debug.py"]

if run == "debug":
    files = ["./Debug.py",
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

logicytics(log, quit_var)
