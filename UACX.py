import ctypes
import sys
import os
import subprocess


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def request_elevation():
    # Get the full path of the current script
    script_path = os.path.abspath(sys.argv[0])
    # Request elevation
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script_path}"', None, 1)
    # Exit the current instance of the script
    sys.exit()


def main():
    print("INFO: Script started executing.")
    if is_admin():
        print("INFO: Running with administrative privileges.")
        # Launch UAC.ps1 with administrative privileges
        ps_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UAC.ps1")
        # Ensure the path is correctly formatted for PowerShell
        ps_script_path = ps_script_path.replace("\\", "/")
        print(f"INFO: Executing PowerShell script: {ps_script_path}")
        # Execute the PowerShell script
        subprocess.run(["powershell", "-File", ps_script_path], check=True)
        print("INFO: PowerShell script execution completed.")
    else:
        print("INFO: Requesting elevation.")
        request_elevation()
        print("INFO: Running with administrative privileges.")
        # Launch UAC.ps1 with administrative privileges
        ps_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UAC.ps1")
        # Ensure the path is correctly formatted for PowerShell
        ps_script_path = ps_script_path.replace("\\", "/")
        print(f"INFO: Executing PowerShell script: {ps_script_path}")
        # Execute the PowerShell script
        subprocess.run(["powershell", "-File", ps_script_path], check=True)
        print("INFO: PowerShell script execution completed.")
    print("INFO: Script execution completed.")


main()
