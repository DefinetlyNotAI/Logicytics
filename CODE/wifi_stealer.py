from __future__ import annotations

from logicytics import log, Execute


def get_password(ssid: str) -> str | None:
    """
    Retrieves the password for a specified Wi-Fi network.
    
    Args:
        ssid (str): The name (SSID) of the Wi-Fi network to retrieve the password for.
    
    Returns:
        str or None: The Wi-Fi network password if found, otherwise None.
    
    Raises:
        Exception: If an error occurs during command execution or password retrieval.
    
    Notes:
        - Uses the Windows `netsh` command to extract network profile details
        - Searches command output for "Key Content" to find the password
        - Logs any errors encountered during the process
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
    """
    Parses the output of the command to extract Wi-Fi profile names.
    
    Args:
        command_output (str): The output of the command "netsh wlan show profile" containing Wi-Fi profile information.
    
    Returns:
        list: A list of extracted Wi-Fi profile names, stripped of whitespace.
    
    Raises:
        No explicit exceptions are raised by this function.
    
    Example:
        >>> output = "All User Profile     : HomeNetwork\\nAll User Profile     : WorkWiFi"
        >>> parse_wifi_names(output)
        ['HomeNetwork', 'WorkWiFi']
    """
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
    
    Executes the "netsh wlan show profile" command to list available Wi-Fi network profiles. 
    Parses the command output to extract individual profile names.
    
    Returns:
        list: A list of Wi-Fi network profile names discovered on the system.
    
    Raises:
        Exception: If an error occurs during the retrieval of Wi-Fi names.
    
    Example:
        wifi_profiles = get_wifi_names()  # Returns ['HomeNetwork', 'CoffeeShop', ...]
    """
    try:
        log.info("Retrieving Wi-Fi names...")
        wifi_names = parse_wifi_names(Execute.command("netsh wlan show profile"))
        log.info(f"Retrieved {len(wifi_names)} Wi-Fi names.")
        return wifi_names
    except Exception as err:
        log.error(err)


@log.function
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


if __name__ == "__main__":
    get_wifi_passwords()
