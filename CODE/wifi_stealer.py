from __future__ import annotations

from __lib_class import *

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


def get_password(ssid: str) -> str | None:
    """
    Retrieves the password associated with a given Wi-Fi SSID.

    Args:
        ssid (str): The SSID of the Wi-Fi network.

    Returns:
        str or None: The password associated with the SSID, or None if no password is found.

    Raises:
        Exception: If an error occurs while executing the command.
    """
    try:
        command_output = Execute.command(
            f'netsh wlan show profile name="{ssid}" key=clear'
        )
        if command_output:
            key_content = command_output.splitlines()
            for line in key_content:
                if "Key Content" in line:
                    return line.split(":")[1].strip()
    except Exception as err:
        log.error(err)


def parse_wifi_names(command_output: str) -> list:
    wifi_names = []

    for line in command_output.split("\n"):
        if "All User Profile" in line:
            start_index = line.find("All User Profile") + len("All User Profile")
            wifi_name = line[start_index:].strip()
            wifi_names.append(wifi_name)

    return wifi_names


def get_wifi_names() -> list:
    """
    Retrieves the names of all Wi-Fi profiles on the system.

    This function executes the command "netsh wlan show profile" to retrieve the list of Wi-Fi profiles.
    It then iterates over each line of the command output and checks if the line contains the string "All User Profile".
    If it does, it extrActions()s the Wi-Fi profile name and appends it to the list of Wi-Fi names.

    Returns:
        list: A list of Wi-Fi profile names.
    """
    try:
        log.info("Retrieving Wi-Fi names...")
        wifi_names = parse_wifi_names(Execute.command("netsh wlan show profile"))
        log.info(f"Retrieved {len(wifi_names)} Wi-Fi names.")
        return wifi_names
    except Exception as err:
        log.error(err)


def get_wifi_passwords():
    """
    Retrieves the passwords for all Wi-Fi profiles on the system.

    This function retrieves the names of all Wi-Fi profiles on the system using the get_wifi_names() function.
    It then iterates over each Wi-Fi profile name and retrieves the password associated with the profile using the get_password() function.
    The Wi-Fi profile names and passwords are stored in a dictionary where the key is the Wi-Fi profile name and the value is the password.
    """
    with open("WiFi.txt", "w") as file:
        for name in get_wifi_names():
            try:
                log.info(f"Retrieving password for {name.removeprefix(': ')}")
                file.write(
                    f"Name: {name.removeprefix(': ')}, Password: {get_password(name.removeprefix(': '))}\n"
                )
            except UnicodeDecodeError as e:
                log.error(e)
            except Exception as e:
                log.error(e)


get_wifi_passwords()
