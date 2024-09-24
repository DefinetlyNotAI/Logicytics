from __lib_class import *

if __name__ == "__main__":
    log = Log(debug=DEBUG)


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
        subprocess.run(cmd, check=True)
        log.info(f"Registry backed up successfully to {export_path}")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to back up the registry: {e}")


backup_registry()
