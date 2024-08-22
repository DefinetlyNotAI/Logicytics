import subprocess
import os

def backup_registry():
    # Define the path where the registry will be exported
    export_path = os.path.join(os.getcwd(), 'RegistryBackup.reg')

    # Command to export the entire registry (HKEY_LOCAL_MACHINE)
    cmd = f'echo Y | reg export HKLM {export_path}'

    try:
        # Execute the command
        subprocess.run(cmd, shell=True, check=True)
        print(f'Registry backed up successfully to {export_path}')
    except subprocess.CalledProcessError as e:
        print(f'Failed to back up the registry: {e}')

backup_registry()
