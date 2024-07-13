import re
import socket
import uuid
import wmi
from local_libraries.Setups import *


def extract_version_number(output):
    """
    Extracts the version number from the output string.

    Args:
        output (str): The output string to extract the version number from.

    Returns:
        str: The extracted version number or None if no match is found.
    """
    pattern = r"\b\d+(\.\d+){2,}\b"
    match = re.search(pattern, output)
    return match.group(0) if match else None


def get_windows_version_info():
    """
    Retrieves the Windows version number and type.

    Returns:
        tuple: A tuple containing the version number (str) and type (str) or None if no match is found.
    """
    command = "wmic os get Caption, Version"
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, shell=True)
    version_number = extract_version_number(result.stdout)
    type_ = (
        re.search(r"(\bHome\b|\bEnterprise\b)", result.stdout, re.IGNORECASE).group(1)
        if re.search(r"(\bHome\b|\bEnterprise\b)", result.stdout, re.IGNORECASE)
        else None
    )
    return version_number, type_


def get_network_info():
    """
    Retrieves the IPv4, IPv6, and MAC address of the current machine.

    Returns:
        tuple: A tuple containing the IPv4 address (str), IPv6 address (list of str), and MAC address (str).
    """
    hostname = socket.gethostname()
    ipv4 = socket.gethostbyname(hostname)
    ipv6 = [item[4][0] for item in socket.getaddrinfo(hostname, None, socket.AF_INET6)]
    mac_address = ":".join(
        ["{:02x}".format((uuid.getnode() >> i) & 0xFF) for i in range(0, 8 * 6, 8)][
            ::-1
        ]
    )
    return ipv4, ipv6, mac_address


def get_computer_model():
    """
    Retrieves the model of the computer system.

    Returns:
        str: The model of the computer system or None if no computer system is found.
    """
    c = wmi.WMI()
    for computer_system in c.Win32_ComputerSystem():
        return computer_system.Model
    return None


def execute_command(command):
    """
    Executes a command and returns the output.

    Args:
        command (str): The command to execute.

    Returns:
        str: The output of the command or an empty string if the command fails.
    """
    logger.info(f"Executing command: {command}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logger.error(f"Command '{command}' failed with error: {stderr}")
        crash("UKN", "fun65", process.returncode, "error")
        return ""
    return stdout.strip()


def write_to_file(filename, content):
    """
    Writes the content to a file.

    Args:
        filename (str): The name of the file to write to.
        content (str): The content to write to the file.

    Returns:
        None
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved text file in {filename}")
    except IOError as e:
        logger.error(f"Error writing to file: {e}")
        crash("IOE", "fun85", e, "error")


def remove_prefix_from_lines(content: str, prefix: str) -> str:
    """
    Removes a prefix from each line in a string.

    Args:
        content (str): The string containing the lines to modify.
        prefix (str): The prefix to remove from each line.

    Returns:
        str: The modified string with the prefix removed from each line.
    """
    # Split the content into lines and remove the prefix from each line
    modified_lines = [line.lstrip(prefix) for line in content.split("\n")]

    # Join the modified lines back into a single string
    modified_content = "\n".join(modified_lines)

    return modified_content


# Main execution block
ipv4, ipv6, mac_address = get_network_info()
version_number, type_ = get_windows_version_info()
commands = [
    "wmic cpu get Name",
    "wmic path win32_VideoController get Name",
    "wmic MEMORYCHIP get BankLabel, Capacity, MemoryType",
    "wmic diskdrive get Model, MediaType, Size",
]
hardware_info = [execute_command(cmd).splitlines()[2].strip() for cmd in commands]

content = [
    f"Computer Model: {get_computer_model()}\n",
    *hardware_info,
    f"Windows Version Number: {version_number}\n",
    f"Windows Type: {type_}\n",
    f"IPv4: {ipv4}\n",
    f"IPv6: {ipv6}\n",
    f"MAC Address: {mac_address}",
]

filename = os.path.join(os.getcwd(), "system_info.txt")
write_to_file(filename, remove_prefix_from_lines(", ".join(content), ", "))
