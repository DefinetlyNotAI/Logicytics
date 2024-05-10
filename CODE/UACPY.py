import subprocess


def get_uac_setting():
    # Query the current UAC setting using PowerShell
    uac_setting = subprocess.run(["powershell", "-Command",
                                  "Get-ItemProperty -Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System' -Name 'EnableLUA'"],
                                 capture_output=True, text=True, check=True)
    # Extract the value
    value = uac_setting.stdout.strip()
    return value


def set_uac_setting(value):
    # Set the UAC setting using PowerShell
    subprocess.run(["powershell", "-Command",
                    "Set-ItemProperty -Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System' -Name 'EnableLUA' -Value " + value],
                   check=True)


def main():
    print("INFO: Script started executing.")
    # Get the current UAC setting
    old_uac_setting = get_uac_setting()
    print(f"INFO: Old UAC setting: {old_uac_setting}")

    # Change the UAC setting to the opposite value
    new_uac_setting = '0' if old_uac_setting == '1' else '1'
    set_uac_setting(new_uac_setting)
    print(f"INFO: New UAC setting: {new_uac_setting}")

    # Ask the user to restart their computer
    print("INFO: Please restart your computer for the changes to take effect.")
    # Prompt the user to restart with confirmation
    user_input = input("Do you want to restart your computer now? (yes/no): ")
    if user_input.lower() == 'yes':
        subprocess.run(["powershell", "-Command", "shutdown /r /t 0"], check=True)
    else:
        print("INFO: Restart cancelled by the user.")


main()
