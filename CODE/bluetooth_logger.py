import datetime
import re
import subprocess

from logicytics import Log, DEBUG

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


# Utility function to log data to a file
def save_to_file(filename, section_title, data):
    """
    Appends data to a file with a section title.

    Args:
        filename (str): The name of the file to write to.
        section_title (str): The title of the section to be added.
        data (str or list): The data to be written. Can be a string or a list of strings.

    Raises:
        Exception: If there is an error writing to the file.
    """
    try:
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(f"\n{'=' * 50}\n{section_title}\n{'=' * 50}\n")
            file.write(f"{data}\n" if isinstance(data, str) else "\n".join(data) + "\n")
            file.write(f"{'=' * 50}\n")
    except Exception as e:
        log.error(f"Error writing to file {filename}: {e}")


# Utility function to run PowerShell commands
def run_powershell_command(command):
    """
    Runs a PowerShell command and returns the output as a list of lines.

    Args:
        command (str): The PowerShell command to run.

    Returns:
        list: A list of strings, each representing a line of the command output.
              Returns an empty list if the command fails or an exception occurs.

    Raises:
        Exception: If there is an error running the PowerShell command.
    """
    try:
        result = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"PowerShell command failed with return code {result.returncode}")
            return []
        return result.stdout.splitlines()
    except Exception as e:
        log.error(f"Error running PowerShell command: {e}")
        return []


# Unified parsing function for PowerShell output
def parse_output(lines, regex, group_names):
    """
    Parses the output lines using the provided regex and group names.

    Args:
        lines (list): A list of strings, each representing a line of the command output.
        regex (str): The regular expression pattern to match each line.
        group_names (list): A list of group names to extract from the matched regex.

    Returns:
        list: A list of dictionaries, each containing the extracted group names and their corresponding values.
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
    except Exception as e:
        log.error(f"Parsing output failed: {e}")


# Function to get paired Bluetooth devices
def get_paired_bluetooth_devices():
    """
    Retrieves a list of paired Bluetooth devices.

    This function runs a PowerShell command to get the list of paired Bluetooth devices,
    parses the output, and extracts the device names and MAC addresses.

    Returns:
        list: A list of strings, each containing the name and MAC address of a paired Bluetooth device.
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
    Logs all Bluetooth data including paired devices and event logs.

    This function collects and logs the following data:
    - Paired Bluetooth devices
    - Bluetooth connection/disconnection logs
    - Bluetooth file transfer logs

    The data is saved to a file named 'bluetooth_data.txt'.
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
    def collect_logs(title, command):
        """
        Collects and logs event logs based on the provided PowerShell command.

        Args:
            title (str): The title of the log section.
            command (str): The PowerShell command to run for collecting logs.
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
    log_bluetooth()
