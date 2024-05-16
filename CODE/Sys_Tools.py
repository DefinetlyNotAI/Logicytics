import os
import subprocess
from pathlib import Path
import colorlog

# Configure colorlog
logger = colorlog.getLogger()
logger.setLevel(colorlog.INFO)  # Set the log level
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def check_service_exists(service_name):
    try:
        # PowerShell command to check if the service exists
        ps_cmd = f"Get-Service -Name '{service_name}'"
        ps_result = subprocess.run(['powershell', '-Command', ps_cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True)
        # If the service exists, the command will succeed, and we can parse the output
        if "Status" in ps_result.stdout:
            return True
    except Exception as e:
        logger.error(f"Error checking if service '{service_name}' exists: {e}")
    return False


def suspend_windows_security():
    # Define the path to PsSuspend.exe
    ps_suspend_path = os.path.join(os.getcwd(), 'sys', 'PsSuspend.exe')

    # Check if PsSuspend.exe exists at the specified path
    if not os.path.exists(ps_suspend_path):
        logger.error(f"PsSuspend.exe not found at {ps_suspend_path}. Please check the file location.")
        return

    # Define the process name of Windows Security (assuming MsMpSvc is still correct)
    process_name = "MsMpSvc"

    # Check if the Windows Security service exists
    if not check_service_exists(process_name):
        logger.error(f"The service '{process_name}' does not exist on this system.")
        return

    try:
        # Construct the command to suspend the process
        cmd = [ps_suspend_path, process_name]

        # Run the command to suspend the process
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        logger.info(result)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return

    # Query the status of the Windows Security service
    try:
        # PowerShell command to get the status of the Windows Security service
        ps_cmd = r"Get-Service -Name MsMpSvc | Select-Object DisplayName, Status"
        ps_result = subprocess.run(['powershell', '-Command', ps_cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True)

        # Simplified extraction of the status from the PowerShell output
        status_lines = ps_result.stdout.strip().split('\n')
        if len(status_lines) > 1:
            # Assuming the second line contains the status
            status_line = status_lines[-1]  # Take the last line as the status
            logger.info(f"Windows Security service status: {status_line}")
            if "Running" in status_line:
                logger.info("Windows Security appears to be running normally.")
            elif "Stopped" in status_line:
                logger.info("Windows Security appears to be stopped, possibly due to the suspension attempt.")
            else:
                logger.warning("Unknown status for Windows Security service.")
        else:
            logger.warning("Could not determine the status of Windows Security service.")

    except Exception as e:
        logger.error(f"Failed to query Windows Security service status: {e}")


def generate_services_file():
    # Get the current working directory
    current_dir = os.getcwd()

    # Define the name of the output file
    output_file_name = "Services_SysInternal.txt"

    # Construct the full path to the output file in the current working directory
    output_file_path = os.path.join(current_dir, output_file_name)

    try:
        # Assuming PsService.exe is directly in the current working directory,
        # Adjust the path if PsService.exe is located elsewhere
        ps_service_path = os.path.join(current_dir, "sys", "PsService.exe")

        # Execute the PowerShell command to get the list of services
        result = subprocess.run([ps_service_path], stdout=subprocess.PIPE)

        # Attempt to decode the output using 'cp1252'
        output_str = result.stdout.decode('cp1252')

        # Write the output to the specified file
        with open(output_file_path, 'w', encoding='cp1252') as file:
            file.write(output_str)

        logger.info(f"Services information has been written to {output_file_path}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def set_admin_password(password):
    """
    Sets the administrator account password using PsPasswd.exe located in the current working directory/sys.

    :param password: The new password for the administrator account.
    """
    # Construct the path to PsPasswd.exe
    pspasswd_path = os.path.join(os.getcwd(), 'sys', 'PsPasswd.exe')

    # Check if PsPasswd.exe exists
    if not os.path.exists(pspasswd_path):
        raise FileNotFoundError(f"PsPasswd.exe not found in {os.getcwd()}/sys")

    # Command to run PsPasswd.exe
    command = [pspasswd_path, 'Administrator', f'"{password}"']

    try:
        # Execute the command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Print the output
        logger.info(result.stdout.decode())
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def generate_log_list_txt():
    # Get the current working directory
    cwd = os.getcwd()

    # Construct the full path to PsLogList.exe
    psloglist_path = Path(cwd) / 'sys' / 'psloglist.exe'

    # Execute PsLogList.exe and capture its output
    result = subprocess.run([str(psloglist_path)], stdout=subprocess.PIPE)

    # Ensure the output is not empty before writing to a file
    if result.stdout:
        # Write the output to a text file
        with open(Path(cwd) / 'LogList_SysInternal.txt', 'w') as log_file:
            log_file.write(result.stdout.decode('utf-8'))
        logger.info("LogList_SysInternal.txt has been created.")
    else:
        logger.warning("No output received from PsLogList.exe.")


def log_sys_internal_users():
    # Get the current working directory
    cwd = os.getcwd()

    # Construct the path to PsLoggedOn.exe
    ps_logged_on_path = os.path.join(cwd, 'sys', 'PsLoggedOn.exe')

    # Execute PsLoggedOn.exe and capture its output
    result = subprocess.run([ps_logged_on_path], stdout=subprocess.PIPE)

    # Decode the output from bytes to string
    output_str = result.stdout.decode('utf-8')

    # Write the output to a text file
    with open('LoggedUsers_SysInternal.txt', 'w') as f:
        f.write(output_str)

    logger.info("LoggedUsers_SysInternal.txt has been created.")


def generate_system_data_txt():
    try:
        # Get the current working directory
        cwd = os.getcwd()

        # Construct the path to PsList.exe
        pslist_path = Path(cwd) / 'sys' / 'pslist.exe'

        # Check if PsList.exe exists
        if not pslist_path.exists():
            logger.error(f"PsList.exe not found at {pslist_path}")
            return

        # Execute PsList.exe and capture its output
        result = subprocess.run([str(pslist_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Write the captured output to a text file
        with open(Path(cwd) / 'SystemData_Advanced_SysInternal.txt', 'w') as f:
            f.write(result.stdout.decode())

        logger.info("System data successfully written to SystemData_Advanced_SysInternal.txt")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


def generate_system_info():
    # Step 1: Get the current working directory
    current_working_directory = os.getcwd()

    # Step 2: Construct the path to PsInfo.exe
    psinfo_path = os.path.join(current_working_directory, 'sys', 'PsInfo.exe')

    # Ensure PsInfo.exe exists at the specified path
    if not os.path.exists(psinfo_path):
        logger.error(f"PsInfo.exe not found at {psinfo_path}")
        return

    # Step 3: Execute PsInfo.exe and capture its output
    try:
        result = subprocess.run([psinfo_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode('utf-8')

        # Step 4: Write the output to a text file
        with open(os.path.join(current_working_directory, 'SystemInfo_Advanced_SysInternal.txt'), 'w') as f:
            f.write(output)

        logger.info("System information successfully saved.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def generate_sid_data():
    # Step 1: Get the current working directory
    current_working_directory = os.getcwd()

    # Step 2: Construct the path to PsGetSid.exe
    ps_get_sid_path = os.path.join(current_working_directory, 'sys', 'PsGetSid.exe')

    # Ensure the path exists
    if not os.path.exists(ps_get_sid_path):
        logger.error(f"PsGetSid.exe not found at {ps_get_sid_path}")
        return

    # Step 3: Execute PsGetSid.exe and capture output
    try:
        result = subprocess.run([ps_get_sid_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode('utf-8')

        # Check if execution was successful
        if result.returncode != 0:
            logger.error("Error executing PsGetSid.exe")
            return

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return

    # Step 4: Write the output to a text file
    with open(os.path.join(current_working_directory, 'SID_Data_SysInternal.txt'), 'w') as file:
        file.write(output)


def generate_running_processes_report():
    # Step 1: Determine the current working directory
    current_working_directory = os.getcwd()

    # Step 2: Construct the path to PsFile.exe
    psfile_path = os.path.join(current_working_directory, 'sys', 'PsFile.exe')

    # Ensure PsFile.exe exists at the constructed path
    if not os.path.exists(psfile_path):
        logger.error(f"PsFile.exe not found at {psfile_path}")
        return

    # Step 3: Execute PsFile.exe and capture output
    try:
        result = subprocess.run([psfile_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode('utf-8')

        # Step 4: Write the output to a text file
        with open(os.path.join(current_working_directory, 'RunningProcesses_SysInternal.txt'), 'w') as file:
            file.write(output)

        logger.info("Report generated successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


generate_running_processes_report()
generate_sid_data()
generate_system_info()
generate_system_data_txt()
log_sys_internal_users()
generate_log_list_txt()
set_admin_password('')  # To set the password to blank
generate_services_file()
suspend_windows_security()
