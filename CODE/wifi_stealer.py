from __lib_class import *
from __lib_log import Log

act = Actions()
log = Log(debug=DEBUG)


def get_password(ssid: str) -> str or None:
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
        command_output = act.run_command(
            f'netsh wlan show profile name="{ssid}" key=clear'
        )
        key_content = command_output.splitlines()
        for line in key_content:
            if "Key Content" in line:
                return line.split(":")[1].strip()
        return None
    except Exception as e:
        log.error(e)


def get_wifi_names() -> list:
    """
    Retrieves the names of all Wi-Fi profiles on the system.

    This function executes the command "netsh wlan show profile" to retrieve the list of Wi-Fi profiles.
    It then iterates over each line of the command output and checks if the line contains the string "All User Profile".
    If it does, it extracts the Wi-Fi profile name and appends it to the list of Wi-Fi names.

    Returns:
        list: A list of Wi-Fi profile names.
    """
    try:
        log.info("Retrieving Wi-Fi names...")
        command_output = act.run_command("netsh wlan show profile")
        wifi_names = []

        for line in command_output.split("\n"):
            if "All User Profile" in line:
                start_index = line.find("All User Profile") + len("All User Profile")
                wifi_name = line[start_index:].strip()
                wifi_names.append(wifi_name)
        log.info(f"Retrieved {len(wifi_names)} Wi-Fi names.")
        return wifi_names
    except Exception as e:
        log.error(e)


with open("WiFi.txt", "w") as file:
    for name in get_wifi_names():
        log.info(f"Retrieving password for {name.removeprefix(': ')}")
        file.write(
            f"Name: {name.removeprefix(': ')}, Password: {get_password(name.removeprefix(': '))}\n"
        )
