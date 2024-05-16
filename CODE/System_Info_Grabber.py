import getpass
import socket
import subprocess
import re
import time
import uuid
import psutil
import wmi
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
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
handler.setFormatter(formatter)
logger.addHandler(handler)
USER_NAME = getpass.getuser()
DESTINATION_PREFIX = "DATA\\" + USER_NAME


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
        if (p['name'] in system_processes or p['name'] in network_processes
                or p['name'] in web_browser_processes or p['name'] in email_clients or p['name'] in office_processes or
                p[
                    'name'] in antivirus_security_processes):
            logger.info(f"INFO: PID = {p['pid']}, Program Name = {p['name']}")
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


ipv4, ipv6, mac_address = get_network_info()
version_number, type = get_windows_version_info()

logger.info(f"INFO: Raw Processes Running Suitable to dump:- {filter_processes()}")
print()
logger.info(f"INFO: Computer Model: {get_computer_model()}")
print()
logger.info("INFO: CPU", execute_command('wmic cpu get Name').splitlines()[2].strip())
print()
logger.info("INFO: GPU", execute_command('wmic path win32_VideoController get Name').splitlines()[2].strip())
print()
logger.info("INFO: RAM", execute_command('wmic MEMORYCHIP get BankLabel, Capacity, MemoryType').splitlines()[2].strip())
print()
logger.info("INFO: SSD", execute_command('wmic diskdrive get Model, MediaType, Size').splitlines()[2].strip())
print()
logger.info(f"INFO: Windows Version Number: {version_number}")
print()
logger.info(f"INFO: Windows Type: {type}")
print()
logger.info(f"INFO: IPv4: {ipv4}")
print()
logger.info(f"INFO: IPv6: {ipv6}")
print()
logger.info(f"INFO: MAC Address: {mac_address}")
print()
time.sleep(3)
