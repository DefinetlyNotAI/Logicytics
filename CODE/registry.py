import os
from __lib_log import Log
from __lib_actions import *


def backup_registry():
    # Define the path where the registry will be exported
    export_path = os.path.join(os.getcwd(), "RegistryBackup.reg")

    # Command to export the entire registry (HKEY_LOCAL_MACHINE)
    cmd = f"echo Y | reg export HKLM {export_path}"

    try:
        # Execute the command
        subprocess.run(cmd, shell=True, check=True)
        Log(debug=DEBUG).info(f"Registry backed up successfully to {export_path}")
    except subprocess.CalledProcessError as e:
        Log(debug=DEBUG).error(f"Failed to back up the registry: {e}")
    Log(debug=DEBUG).info(f"Registry back-up executed")
