import datetime
import re
import subprocess
from typing import LiteralString

from logicytics import log


# Utility function to log data to a file
def save_to_file(filename: str, section_title: str, data: str):
    """
    Appends data to a file with a section title.
    
    Args:
        filename (str): Path to the file where data will be written. Must be a valid file path.
        section_title (str): Title describing the section being added to the file.
        data (str or list): Content to be written. Accepts either a single string or a list of strings.
    
    Raises:
        IOError: If the file cannot be opened or written to due to permission or path issues.
        Exception: For any unexpected errors during file writing.
    
    Notes:
        - Uses UTF-8 encoding for file writing
        - Adds decorative section separators around the content
        - Automatically handles single string or list of strings input
        - Logs any errors encountered during file writing
    """
    try:
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(f"\n{'=' * 50}\n{section_title}\n{'=' * 50}\n")
            file.write(f"{data}\n" if isinstance(data, str) else "\n".join(data) + "\n")
            file.write(f"{'=' * 50}\n")
    except Exception as err:
        log.error(f"Error writing to file {filename}: {err}")


# Utility function to run PowerShell commands
def run_powershell_command(command: str) -> None | list[LiteralString]:
    """
    Runs a PowerShell command and returns the output as a list of lines.
    
    Args:
        command (str): The PowerShell command to execute.
    
    Returns:
        list: A list of strings representing each line of the command output.
              Returns an empty list if the command execution fails or an exception occurs.
    
    Raises:
        subprocess.CalledProcessError: If the PowerShell command returns a non-zero exit status.
        Exception: For any unexpected errors during command execution.
    
    Notes:
        - Uses subprocess.run() with capture_output=True to capture command output
        - Logs errors for failed commands or exceptions
        - Splits command output into lines for easier processing
    """
    try:
        result = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"PowerShell command failed with return code {result.returncode}")
            return []
        return result.stdout.splitlines()
    except Exception as err:
        log.error(f"Error running PowerShell command: {err}")
        return []


# Unified parsing function for PowerShell output
def parse_output(lines: list[LiteralString], regex: str, group_names: list[str]):
    """
    Parses the output lines using the provided regex and group names.
    
    Parameters:
        lines (list): A list of strings representing command output lines.
        regex (str): Regular expression pattern to match each line.
        group_names (list): List of group names to extract from matched regex.
    
    Returns:
        list: Dictionaries containing extracted group names and their values.
    
    Raises:
        Exception: If parsing the output encounters an unexpected error.
    
    Notes:
        - Skips lines that do not match the provided regex pattern
        - Logs debug messages for unrecognized lines
        - Logs error if parsing fails completely
    """
    results = []
    try:
        for line in lines:
            match = re.match(regex, line)
            if match:
                results.append({name: match.group(name) for name in group_names})
            else:
                log.debug(f"Skipping unrecognized line: {line}")
        return results
    except Exception as err:
        log.error(f"Parsing output failed: {err}")


# Function to get paired Bluetooth devices
def get_paired_bluetooth_devices() -> list[str]:
    """
    Retrieves a list of paired Bluetooth devices with their names and MAC addresses.
    
    This function executes a PowerShell command to fetch Bluetooth devices with an "OK" status, 
    parses the output to extract device details, and attempts to retrieve MAC addresses from device IDs.
    
    Returns:
        list: A list of formatted strings containing device names and MAC addresses. 
        Each string follows the format "Name: <device_name>, MAC: <mac_address>".
    
    Raises:
        Exception: If there are issues running the PowerShell command or parsing the output.
    """
    command = (
        'Get-PnpDevice -Class Bluetooth | Where-Object { $_.Status -eq "OK" } | Select-Object Name, DeviceID'
    )
    output = run_powershell_command(command)
    log.debug(f"Raw PowerShell output for paired devices:\n{output}")

    devices = parse_output(
        output,
        regex=r"^(?P<Name>.+?)\s+(?P<DeviceID>.+)$",
        group_names=["Name", "DeviceID"]
    )

    # Extract MAC addresses
    for device in devices:
        mac_match = re.search(r"BLUETOOTHDEVICE_(?P<MAC>[A-F0-9]{12})", device["DeviceID"], re.IGNORECASE)
        device["MAC"] = mac_match.group("MAC") if mac_match else "Address Not Found"

    return [f"Name: {device['Name']}, MAC: {device['MAC']}" for device in devices]


# Function to log all Bluetooth data
@log.function
def log_bluetooth():
    """
    Logs comprehensive Bluetooth data including paired devices and system event logs.
    
    This function performs the following actions:
    - Captures the current timestamp
    - Retrieves and logs paired Bluetooth devices
    - Collects Bluetooth connection/disconnection event logs
    - Captures Bluetooth file transfer logs
    - Saves all collected data to 'bluetooth_data.txt'
    
    The function uses internal utility functions to run PowerShell commands, parse outputs, and save results to a file. It provides a systematic approach to logging Bluetooth-related system information.
    
    Logs are saved with descriptive section titles, making the output easily readable and organized. If no data is found for a specific section, a default "No logs found" message is recorded.
    
    Note:
        - Requires administrative or sufficient system permissions to access Windows event logs
        - Logs are appended to the file, allowing historical tracking of Bluetooth events
    """
    log.info("Starting Bluetooth data logging...")
    filename = "bluetooth_data.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_to_file(filename, "Bluetooth Data Collection - Timestamp", timestamp)

    # Collect and log paired devices
    log.info(f"Collecting paired devices...")
    paired_devices = get_paired_bluetooth_devices()
    section_title = "Paired Bluetooth Devices"
    save_to_file(filename, section_title, paired_devices or ["No paired Bluetooth devices found."])
    log.debug(f"{section_title}: {paired_devices}")

    # Collect and log event logs
    def collect_logs(title: str, command: str):
        """
        Collects and logs event logs by executing a PowerShell command and saving the results.
        
        Args:
            title (str): The title or description of the log section being collected.
            command (str): The PowerShell command to execute for retrieving event logs.
        
        Behavior:
            - Runs the specified PowerShell command using `run_powershell_command()`
            - Saves the log results to a file using `save_to_file()`
            - Logs an informational message about the log collection
            - If no logs are found, saves a default "No logs found." message
            - Uses the global `filename` variable for log file destination
        
        Raises:
            Potential exceptions from `run_powershell_command()` and `save_to_file()` which are handled internally
        """
        logs = run_powershell_command(command)
        save_to_file(filename, title, logs or ["No logs found."])
        log.info(f"Getting {title}...")

    collect_logs(
        "Bluetooth Connection/Disconnection Logs",
        'Get-WinEvent -LogName "Microsoft-Windows-Bluetooth-BthLEServices/Operational" '
        '| Select-Object TimeCreated, Id, Message | Format-Table -AutoSize'
    )

    collect_logs(
        "Bluetooth File Transfer Logs",
        'Get-WinEvent -LogName "Microsoft-Windows-Bluetooth-BthLEServices/Operational" '
        '| Select-String -Pattern "file.*transferred" | Format-Table -AutoSize'
    )

    log.info("Finished Bluetooth data logging.")


if __name__ == "__main__":
    try:
        log_bluetooth()
    except Exception as e:
        log.error(f"Failed to log Bluetooth data: {e}")
