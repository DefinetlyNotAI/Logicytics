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

        # Make sure it exists first
        with open("Bluetooth Info.txt", "w", encoding="UTF-8") as f:
            for device in devices:
                log.info(f"Name: {device.Name}")
                f.write(f"Name: {device.Name}\n")
                f.write(f"    Device ID: {device.DeviceID}\n")
                f.write(f"    Description: {device.Description}\n")
                f.write(f"    Manufacturer: {device.Manufacturer}\n")
                f.write(f"    Status: {device.Status}\n")
                f.write(f"    PNP Device ID: {device.PNPDeviceID}\n")
                f.write("-" * 50)
                f.write("\n\n")
    except Exception as e:
        log.error(f"Error: {e}")


get_bluetooth_device_details()
