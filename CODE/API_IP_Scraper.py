import requests

from CODE.local_libraries.Setups import *


def get_public_ip():
    """
    Fetches the public IP address using the ipify API.

    Returns:
        str: Public IP address as a string.
        None: If there's an error fetching the IP.
    """
    try:
        response = requests.get("https://api.ipify.org?format=json")
        response.raise_for_status()  # Raises an HTTPError if the response was unsuccessful
        return response.json()["ip"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching public IP: {e}")
        crash("CE", "fun5", e, "error")
        return None


def save_to_file(filename, content):
    """
    saves the provided content to a file.

    Args:
        filename (str): Name of the file to save to.
        content (str): Content to write to the file.

    Raises:
        IOError: If there's an issue writing to the file.
    """
    try:
        with open(filename, "w") as file:
            file.write(content)
    except IOError as e:
        logger.error(f"Error writing to file: {e}")
        crash("IOE", "fun23", e, "error")


def ip_scraper():
    """
    Main function to fetch and save public IP information using the VPNAPI service.
    """

    # Get the script directory and parent directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.join(script_dir, "..")

    # Construct the path to the API key file
    api_key_file_path = os.path.join(parent_dir, "SYSTEM", "API-IP.key")

    # Check if the API key file exists before proceeding
    if not os.path.exists(api_key_file_path):
        logger.error("Exiting: The API-IP.key file does not exist.")
        crash("FNF", "fun42", "API-IP.key file not found", "error")
        return

    # Read the API key from the file
    with open(api_key_file_path, "r") as file:
        api_key = file.read().strip()
        if api_key == "API-NO":
            exit()

    # Attempt to fetch the public IP
    public_ip = get_public_ip()
    if not public_ip:
        logger.error("Exiting: Could not fetch your public IP address.")
        crash("CE", "fun42", "Could not fetch your public IP address", "error")
        return

    # Construct the URL for the request
    url = f"https://vpnapi.io/api/{public_ip}?key={api_key}"

    # Make the request to the VPNAPI service
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response was unsuccessful
    except requests.exceptions.HTTPError as e:
        logger.error(f"Exiting: Failed to retrieve data from VPNAPI. Error: {e}")
        crash("HTP", "fun42", e, "error")
        return

    # Parse the JSON response
    data = response.json()

    # Format the output string
    output = (
        f"Country: {data['location']['country']}\n"
        f"City: {data['location']['city']}\n"
        f"ISP: {data['network']['autonomous_system_organization']}\n"
        f"Organization: {data['network']['autonomous_system_organization']}\n\n"
        f"VPN Used: {'Yes' if data['security']['vpn'] else 'No'}\n"
        f"Proxy Used: {'Yes' if data['security']['proxy'] else 'No'}\n"
        f"Tor Used: {'Yes' if data['security']['tor'] else 'No'}\n"
    )

    # Save the formatted output to a file
    save_to_file("API_Output.txt", output)
    logger.info("Operation completed successfully.")


if __name__ == "__main__":
    ip_scraper()
