import subprocess
import json
from logicytics import Log, DEBUG

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


def get_bluetooth_device_details():
    """
    Retrieves and logs detailed information about Bluetooth devices on the system.

    The function runs a PowerShell command to query devices whose names contain the term 'Bluetooth'.
    It writes the information to a text file named 'Bluetooth Info.txt'.

    Information for each device includes:
    - Name
    - Device ID
    - Description
    - Manufacturer
    - Status
    - PNP Device ID

    Logs errors if any issues are encountered during the process.

    Returns:
        None
    """
    log.info("Fetching detailed info for Bluetooth devices...")
    try:
        devices = _query_bluetooth_devices()
        _write_device_info_to_file(devices, "Bluetooth Info.txt")
    except Exception as e:
        log.error(f"Error: {e}")
        exit(1)


def _query_bluetooth_devices():
    """
    Queries the system for Bluetooth devices using PowerShell commands.

    Returns:
        list: A list of device information dictionaries.
    """
    try:
        # Run PowerShell command to get Bluetooth devices
        command = (
            "Get-PnpDevice | Where-Object { $_.FriendlyName -like '*Bluetooth*' } | "
            "Select-Object FriendlyName, DeviceID, Description, Manufacturer, Status, PnpDeviceID | "
            "ConvertTo-Json -Depth 3"
        )
        result = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True, check=True)
        devices = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to query Bluetooth devices: {e}")
        exit(1)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse device information: {e}")
        exit(1)

    if isinstance(devices, dict):
        devices = [devices]  # Handle single result case

    device_info_list = []
    for device in devices:
        device_info = {
            'Name': device.get('FriendlyName', 'Unknown'),
            'Device ID': device.get('DeviceID', 'Unknown'),
            'Description': device.get('Description', 'Unknown'),
            'Manufacturer': device.get('Manufacturer', 'Unknown'),
            'Status': device.get('Status', 'Unknown'),
            'PNP Device ID': device.get('PnpDeviceID', 'Unknown')
        }
        log.debug(f"Retrieved device: {device_info['Name']}")
        device_info_list.append(device_info)

    return device_info_list


def _write_device_info_to_file(devices, filename):
    """
    Writes the details of the Bluetooth devices to a file.

    Args:
        devices (list): List of device information dictionaries.
        filename (str): Name of the file to write to.

    Returns:
        None
    """
    try:
        with open(filename, "w", encoding="UTF-8") as file:
            for device_info in devices:
                _write_single_device_info(file, device_info)
    except Exception as e:
        log.error(f"Failed to write device information to file: {e}")
        exit(1)


def _write_single_device_info(file, device_info):
    """
    Writes information for a single Bluetooth device to the file.

    Args:
        file (TextIO): File object to write to.
        device_info (dict): Dictionary containing device information.

    Returns:
        None
    """
    file.write(f"Name: {device_info.get('Name', 'Unknown')}\n")
    for key, value in device_info.items():
        if key != 'Name':
            file.write(f"    {key}: {value}\n")
    file.write("\n")  # Separate devices with a blank line


if __name__ == "__main__":
    get_bluetooth_device_details()
