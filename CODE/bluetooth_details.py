from __future__ import annotations

import json
import subprocess

from logicytics import log


@log.function
def get_bluetooth_device_details():
    """
    Retrieves and logs detailed information about Bluetooth devices on the system.
    
    Executes a PowerShell query to collect Bluetooth device details and writes the information to a text file. 
    The function performs the following key actions:
    - Logs the start of the device information retrieval process
    - Queries Bluetooth devices using an internal helper function
    - Writes device details to 'Bluetooth Info.txt' if devices are found
    
    Returns:
        None: No return value; results are written to a file and logged
    """
    log.info("Fetching detailed info for Bluetooth devices...")
    devices = _query_bluetooth_devices()
    if devices:
        _write_device_info_to_file(devices, "Bluetooth Info.txt")


def _query_bluetooth_devices() -> bool | list[dict[str, str]]:
    """
    Queries the system for Bluetooth devices using PowerShell commands.
    
    Executes a PowerShell command to retrieve detailed information about Bluetooth devices connected to the system. 
    The function handles potential errors during command execution and JSON parsing, providing fallback values 
    for device information.
    
    Returns:
        bool | list[dict[str, str]]: A list of device information dictionaries or False if an error occurs. 
        Each dictionary contains details such as Name, Device ID, Description, Manufacturer, Status, and PNP Device ID.
    
    Raises:
        No direct exceptions are raised. Errors are logged and the function returns False.
    
    Example:
        devices = _query_bluetooth_devices()
        if devices:
            for device in devices:
                print(device['Name'])
    """
    try:
        # Run PowerShell command to get Bluetooth devices
        command = (
            "Get-PnpDevice | Where-Object { $_.FriendlyName -like '*Bluetooth*' } | "
            "Select-Object FriendlyName, DeviceID, Description, Manufacturer, Status, PnpDeviceID | "
            "ConvertTo-Json -Depth 3"
        )
        result = subprocess.run(["powershell", "-Command", command],
                                capture_output=True, text=True, check=True)
        devices = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to query Bluetooth devices with command '{command}': {e}")
        return False
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse device information: {e}")
        return False

    if isinstance(devices, dict):
        devices = [devices]  # Handle single result case

    device_info_list = []
    for device in devices:
        FALLBACK_MSG = 'Unknown (Fallback due to failed Get request)'
        device_info = {
            'Name': device.get('FriendlyName', FALLBACK_MSG),
            'Device ID': device.get('DeviceID', FALLBACK_MSG),
            'Description': device.get('Description', FALLBACK_MSG),
            'Manufacturer': device.get('Manufacturer', FALLBACK_MSG),
            'Status': device.get('Status', FALLBACK_MSG),
            'PNP Device ID': device.get('PnpDeviceID', FALLBACK_MSG)
        }
        log.debug(f"Retrieved device: {device_info['Name']}")
        device_info_list.append(device_info)

    return device_info_list


def _write_device_info_to_file(devices, filename):
    """
    Writes the details of Bluetooth devices to a specified file.
    
    Args:
        devices (list): A list of dictionaries containing Bluetooth device information.
        filename (str): The path and name of the file where device details will be written.
    
    Raises:
        IOError: If there is an error opening or writing to the specified file.
        OSError: If there are file system related issues during file writing.
    
    Notes:
        - Uses UTF-8 encoding for file writing
        - Logs an error if file writing fails
        - Calls _write_single_device_info() for each device in the list
    """
    try:
        with open(filename, "w", encoding="UTF-8") as file:
            for device_info in devices:
                _write_single_device_info(file, device_info)
    except Exception as e:
        log.error(f"Failed to write device information to file: {e}")


def _write_single_device_info(file, device_info):
    """
    Writes detailed information for a single Bluetooth device to the specified file.
    
    Parameters:
        file (TextIO): An open file object to which device information will be written.
        device_info (dict): A dictionary containing key-value pairs of Bluetooth device attributes.
    
    Writes the device name followed by all other device attributes, with each device's information separated by a blank line. Uses `.get()` method to provide a fallback 'Unknown' value if the device name is missing.
    
    Example:
        If device_info is {'Name': 'Wireless Headset', 'Address': '00:11:22:33:44:55', 'Connected': 'True'}
        The file will contain:
        Name: Wireless Headset
            Address: 00:11:22:33:44:55
            Connected: True
    
        If no name is provided:
        Name: Unknown
            Address: 00:11:22:33:44:55
            Connected: True
    """
    file.write(f"Name: {device_info.get('Name', 'Unknown')}\n")
    for key, value in device_info.items():
        if key != 'Name':
            file.write(f"    {key}: {value}\n")
    file.write("\n")  # Separate devices with a blank line


if __name__ == "__main__":
    get_bluetooth_device_details()
