from logicytics import *

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


@log.function
def backup_registry():
    """
    Backs up the Windows registry to a file named 'RegistryBackup.reg' in the current working directory.

    This function uses the reg export command to export the entire
    registry (HKEY_LOCAL_MACHINE) and logs the result.
    """
    export_path = os.path.join(os.getcwd(), "RegistryBackup.reg")
    reg_path = r"C:\Windows\System32\reg.exe"
    cmd = [reg_path, "export", "HKLM", export_path]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info(f"Registry backed up successfully to {export_path}. Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to back up the registry: {e}. More details: {result.stderr}")
    except Exception as e:
        log.error(f"Failed to back up the registry: {e}")


backup_registry()
