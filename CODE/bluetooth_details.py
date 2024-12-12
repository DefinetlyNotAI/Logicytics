import win32com.client

from logicytics import Log, DEBUG

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


def get_bluetooth_device_details():
    """
    Retrieves and logs detailed information about Bluetooth devices on the system.

    The function connects to the Windows Management Instrumentation (WMI) service and queries for devices
    whose names contain the term 'Bluetooth'. It writes the information to a text file named 'Bluetooth Info.txt'.

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
    Queries the WMI service for Bluetooth devices.

    Returns:
        list: A list of device information dictionaries.
    """
    try:
        locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        service = locator.ConnectServer(".", "root\\cimv2")
        devices = service.ExecQuery("SELECT * FROM Win32_PnPEntity WHERE Name LIKE '%Bluetooth%'")
    except Exception as e:
        log.error(f"Failed to query Bluetooth devices: {e}")
        exit(1)

    device_info_list = []
    for device in devices:
        device_info = {
            'Name': getattr(device, 'Name', 'Unknown'),
            'Device ID': getattr(device, 'DeviceID', 'Unknown'),
            'Description': getattr(device, 'Description', 'Unknown'),
            'Manufacturer': getattr(device, 'Manufacturer', 'Unknown'),
            'Status': getattr(device, 'Status', 'Unknown'),
            'PNP Device ID': getattr(device, 'PNPDeviceID', 'Unknown')
        }
        log.info(f"Retrieved device: {device_info['Name']}")
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
        with open(filename, "w", encoding="UTF-8") as f:
            for device_info in devices:
                for key, value in device_info.items():
                    formatted_line = f"{key}: {value}\n" if key == 'Name' else f"    {key}: {value}\n"
                    f.write(formatted_line)
                f.write("\n")  # Separate devices with a blank line
    except Exception as e:
        log.error(f"Failed to write device information to file: {e}")
        exit(1)


get_bluetooth_device_details()
