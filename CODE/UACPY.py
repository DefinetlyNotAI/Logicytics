import subprocess
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


def get_uac_setting():
    """
    Retrieves the current UAC setting from the Windows registry.

    Returns:
        str: The current UAC setting value.
    """
    # Query the current UAC setting using PowerShell
    uac_setting = subprocess.run(["powershell", "-Command",
                                  "Get-ItemProperty -Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System' -Name 'EnableLUA'"],
                                 capture_output=True, text=True, check=True)
    # Extract the value
    value = uac_setting.stdout.strip()
    return value


def set_uac_setting(value):
    """
    Sets the UAC setting in the Windows registry.

    Args:
        value (str): The new UAC setting value.
    """
    # Set the UAC setting using PowerShell
    subprocess.run(["powershell", "-Command",
                    f"Set-ItemProperty -Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System' -Name 'EnableLUA' -Value {value}"],
                   check=True)


def main():
    """
    The Main function retrieves the current UAC setting, changes it to the opposite value,
    and prompts the user to restart their computer.
    """
    logger.info("Script started executing.")
    # Get the current UAC setting
    old_uac_setting = get_uac_setting()
    logger.info(f"Old UAC setting: {old_uac_setting}")

    # Change the UAC setting to the opposite value
    new_uac_setting = '0' if old_uac_setting == '1' else '1'
    set_uac_setting(new_uac_setting)
    logger.info(f"New UAC setting: {new_uac_setting}")

    # Ask the user to restart their computer
    logger.info("Please restart your computer for the changes to take effect.")
    # Prompt the user to restart with confirmation
    user_input = input("Do you want to restart your computer now? (yes/no): ")
    if user_input.lower() == 'yes':
        subprocess.run(["powershell", "-Command", "shutdown /r /t 0"], check=True)
    else:
        logger.info("Restart cancelled by the user.")


if __name__ == "__main__":
    main()
