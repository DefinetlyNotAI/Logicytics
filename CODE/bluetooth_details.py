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
        locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        service = locator.ConnectServer(".", "root\\cimv2")

        devices = service.ExecQuery("SELECT * FROM Win32_PnPEntity WHERE Name LIKE '%Bluetooth%'")

        # Making sure it exists first
        with open("Bluetooth Info.txt", "w", encoding="UTF-8") as f:
            for device in devices:
                device_info = {
                    'Name': getattr(device, 'Name', 'Unknown'),
                    'Device ID': getattr(device, 'DeviceID', 'Unknown'),
                    'Description': getattr(device, 'Description', 'Unknown'),
                    'Manufacturer': getattr(device, 'Manufacturer', 'Unknown'),
                    'Status': getattr(device, 'Status', 'Unknown'),
                    'PNP Device ID': getattr(device, 'PNPDeviceID', 'Unknown')
                }
                log.info(f"Name: {device_info['Name']}")
                for key, value in device_info.items():
                    f.write(f"{key}: {value}\n" if key == 'Name' else f"    {key}: {value}\n")
    except Exception as e:
        log.error(f"Error: {e}")


get_bluetooth_device_details()
