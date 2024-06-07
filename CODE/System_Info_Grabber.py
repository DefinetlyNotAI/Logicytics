import getpass
import os
import re
import socket
import subprocess
import uuid
import colorlog
import wmi

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
DESTINATION_PREFIX = f"DATA\\{USER_NAME}"

def extract_version_number(output):
    pattern = r'\b\d+(\.\d+){2,}\b'
    match = re.search(pattern, output)
    return match.group(0) if match else None

def get_windows_version_info():
    command = 'wmic os get Caption, Version'
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, shell=True)
    version_number = extract_version_number(result.stdout)
    type_ = re.search(r'(\bHome\b|\bEnterprise\b)', result.stdout, re.IGNORECASE).group(1) if re.search(r'(\bHome\b|\bEnterprise\b)', result.stdout, re.IGNORECASE) else None
    return version_number, type_

def get_network_info():
    hostname = socket.gethostname()
    ipv4 = socket.gethostbyname(hostname)
    ipv6 = [item[4][0] for item in socket.getaddrinfo(hostname, None, socket.AF_INET6)]
    mac_address = ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff) for i in range(0, 8 * 6, 8)][::-1])
    return ipv4, ipv6, mac_address

def get_computer_model():
    c = wmi.WMI()
    for computer_system in c.Win32_ComputerSystem():
        return computer_system.Model
    return None

def execute_command(command):
    logger.info(f"Executing command: {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode!= 0:
        logger.error(f"Command '{command}' failed with error: {stderr}")
        return ""
    return stdout.strip()

def write_to_file(filename, content):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved text file in {filename}")
    except IOError as e:
        logger.error(f"Error writing to file: {e}")

def remove_prefix_from_lines(content, prefix):
    return '\n'.join(line.lstrip(prefix) for line in content.split('\n'))

# Main execution block
ipv4, ipv6, mac_address = get_network_info()
version_number, type_ = get_windows_version_info()
commands = ['wmic cpu get Name', 'wmic path win32_VideoController get Name', 'wmic MEMORYCHIP get BankLabel, Capacity, MemoryType', 'wmic diskdrive get Model, MediaType, Size']
hardware_info = [execute_command(cmd).splitlines()[2].strip() for cmd in commands]

content = [
    f"Computer Model: {get_computer_model()}\n",
    *hardware_info,
    f"Windows Version Number: {version_number}\n",
    f"Windows Type: {type_}\n",
    f"IPv4: {ipv4}\n",
    f"IPv6: {ipv6}\n",
    f"MAC Address: {mac_address}"
]

filename = os.path.join(os.getcwd(), "system_info.txt")
write_to_file(filename, remove_prefix_from_lines(', '.join(content), ', '))
