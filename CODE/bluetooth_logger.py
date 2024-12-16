import winreg
import subprocess
import re
import datetime
from logicytics import Log, DEBUG

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})

# Utility function to log collected data to a file
def log_to_file(filename, section_title, data):
    """Logs collected data to a text file with a section title."""
    try:
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(f"\n{'=' * 50}\n{section_title}\n{'=' * 50}\n")
            if isinstance(data, list):
                for item in data:
                    file.write(f"{item}\n")
            else:
                file.write(f"{data}\n")
            file.write(f"\n{'=' * 50}\n")
    except Exception as e:
        log.error(f"Error writing to file {filename}: {e}")


# Function to collect paired Bluetooth devices
def get_paired_bluetooth_devices():
    """Retrieves paired Bluetooth devices from the Windows Registry."""
    devices = []
    try:
        reg_path = r"SYSTEM\CurrentControlSet\Services\BTHPORT\Parameters\Devices"
        registry_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
        i = 0
        while True:
            try:
                device_mac = winreg.EnumKey(registry_key, i)
                device_key_path = f"{reg_path}\\{device_mac}"
                device_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, device_key_path)

                try:
                    device_name, _ = winreg.QueryValueEx(device_key, "Name")
                except FileNotFoundError:
                    device_name = "Unknown"

                devices.append(f"Device Name: {device_name}, MAC: {device_mac}")
                winreg.CloseKey(device_key)
                i += 1
            except OSError:
                break  # No more devices
    except Exception as e:
        log.error(f"Error reading Bluetooth devices from registry: {e}")

    return devices


# Function to execute PowerShell command and return results
def run_powershell_command(command):
    """Runs a PowerShell command and returns the output."""
    try:
        result = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True)
        return result.stdout.split('\n')
    except Exception as e:
        log.error(f"Error running PowerShell command: {e}")
        return []


# Function to get connection/disconnection logs from Event Viewer
def get_bluetooth_event_logs():
    """Extracts connection/disconnection logs from Event Viewer (Bluetooth)."""
    powershell_command = (
        'Get-WinEvent -LogName "Microsoft-Windows-Bluetooth-BthLEServices/Operational" '
        '| Select-Object TimeCreated, Id, Message | Format-Table -AutoSize'
    )
    logs = run_powershell_command(powershell_command)
    return logs


# Function to get Bluetooth file transfer logs
def get_bluetooth_file_transfer_logs():
    """Extracts Bluetooth file transfer logs from Event Viewer."""
    powershell_command = (
        'Get-WinEvent -LogName "Microsoft-Windows-Bluetooth-BthLEServices/Operational" '
        '| Select-Object TimeCreated, Id, Message | Format-Table -AutoSize'
    )
    log_output = run_powershell_command(powershell_command)

    transfer_logs = []
    try:
        transfer_logs = re.findall(r'.*Bluetooth.*file.*transferred.*', '\n'.join(log_output), re.IGNORECASE)
    except Exception as e:
        log.error(f"Error parsing file transfer logs: {e}")

    return transfer_logs


# Main function to collect and log all Bluetooth data


def main():
    filename = "bluetooth_data.txt"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(filename, "Bluetooth Data Collection - Timestamp", current_time)

    log.info("Collecting Paired Bluetooth Devices...")
    paired_devices = get_paired_bluetooth_devices()
    if paired_devices:
        log_to_file(filename, "Paired Bluetooth Devices", paired_devices)
        for device in paired_devices:
            log.debug(device)
    else:
        log.warning("No paired Bluetooth devices found.")
        log_to_file(filename, "Paired Bluetooth Devices", "No paired Bluetooth devices found.")

    log.info("Collecting Bluetooth Connection/Disconnection Logs...")
    bluetooth_logs = get_bluetooth_event_logs()
    if bluetooth_logs:
        log_to_file(filename, "Bluetooth Connection/Disconnection Logs", bluetooth_logs)
        for log_for in bluetooth_logs:
            log.debug(log_for)
    else:
        log.warning("No Bluetooth connection/disconnection logs found.")
        log_to_file(filename, "Bluetooth Connection/Disconnection Logs", "No Bluetooth connection/disconnection logs found.")

    log.info("Collecting Bluetooth File Transfer Logs...")
    file_transfers = get_bluetooth_file_transfer_logs()
    if file_transfers:
        log_to_file(filename, "Bluetooth File Transfer Logs", file_transfers)
        for transfer in file_transfers:
            log.warning(transfer)
    else:
        log.warning("No Bluetooth file transfers found.")
        log_to_file(filename, "Bluetooth File Transfer Logs", "No Bluetooth file transfers found.")  # Ensure we log even if no transfers are found


if __name__ == "__main__":
    main()