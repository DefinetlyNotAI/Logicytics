from __lib_class import *


def backup_registry():
    """
    Backs up the Windows registry to a file named 'RegistryBackup.reg' in the current working directory.

    This function uses the reg export command to export the entire registry (HKEY_LOCAL_MACHINE) and logs the result.

    Parameters:
    None

    Returns:
    None
    """
    # Define the path where the registry will be exported
    export_path = os.path.join(os.getcwd(), "RegistryBackup.reg")

    # Command to export the entire registry (HKEY_LOCAL_MACHINE)
    cmd = f"echo Y | reg export HKLM {export_path}"

    try:
        # Execute the command
        subprocess.run(cmd, shell=False, check=True)
        log.info(f"Registry backed up successfully to {export_path}")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to back up the registry: {e}")
    log.info(f"Registry back-up executed")


backup_registry()
