from __lib_log import *
from __lib_actions import *
import ctypes
import os


class Checks:
    def __init__(self):
        self.Actions = Actions()

    @staticmethod
    def admin() -> bool:
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except AttributeError:
            return False

    def uac(self) -> bool:
        value = self.Actions.run_command(
            "powershell (Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System).EnableLUA"
        )
        return int(value.strip("\n")) == 1

def wrapper_execution(script: str, type="Script"):
    """
    Executes a script and logs the output based on the script type and silence level.

    Args:
        script (str): The path to the script to execute.
        type (str): The type of script to execute. It Can be either "Command" or "Script".

    Returns:
        Tuple[str, str]: A tuple containing the output of the script and an empty string.
    """
    script = fr".\{script}"
    # Unblock the script if it is a PowerShell script
    if os.path.splitext(script)[1].lower() == ".ps1":
        try:
            unblock_command = f'powershell.exe -Command "Unblock-File -Path {script}"'
            subprocess.run(unblock_command, shell=True, check=True)
            log.info("PS1 Script unblocked.")
        except Exception as err:
            log.critical(f"Failed to unblock script: {err}")

        if DEBUG:
            log.debug("Unblocking script: " + unblock_command)
            log.debug("Script: " + script)
            log.debug("Script Type: " + type)

    # Execute the script based on the type
    if type == "Command":
        command = f'powershell.exe -Command "& {script}"'
        process = subprocess.Popen(command, shell=True)
    elif type == "Script":
        command = fr'.\{script}'
        process =  subprocess.run(fr'.\{script}', capture_output=True, text=True)
    else:
        log.critical(f"Script Failure, Unknown entry type: {type}")
        exit(1)

    if DEBUG:
        log.debug(f"Process: {process}")
        log.debug("Command: " + command)
        log.debug("Script Type: " + type)

    if os.path.splitext(script)[1].lower() == ".py":
        # Print the output in real-time
        for line in iter(process.stdout.readline, b""):
            decoded_line = line.decode("utf-8").strip()
            print(decoded_line)
    else:
        # Initialize Identifier variable
        Identifier = None
        decoded_line = ""
        # Read the first word until :
        for line in iter(process.stdout.readline, b""):
            decoded_line = line.decode("utf-8").strip()
            if ":" in decoded_line:
                words = decoded_line.split(":", 1)
                Identifier = words[0].strip().upper()
            decoded_line = words[1].strip()
            if DEBUG:
                log.debug("Decoded Line: " + decoded_line)
                log.debug("Identifier: " + Identifier)
                log.debug(f"Words: {words}")
                log.debug(f"Line Value: {line}")
            break
        # Log the output based on the Identifier
        if Identifier == "INFO":
            log.info(decoded_line)
        elif Identifier == "ERROR":
            log.error(decoded_line)
        elif Identifier == "WARNING":
            log.warning(decoded_line)
        elif Identifier == "DEBUG":
            log.debug(decoded_line)
        else:
            log.debug(decoded_line)

    # Wait for the process to finish and get the final output/error
    stdout, _ = process.communicate()

    # Decode the output from bytes to string
    stdout = stdout.decode("utf-8") if stdout else ""
    # Return the output
    print(stdout)

if __name__ == "__main__":
    # Initialization
    os.makedirs("../ACCESS/LOGS/", exist_ok=True)
    os.makedirs("../ACCESS/BACKUP/", exist_ok=True)
    log = Log(debug=DEBUG)
    try:
        action, sub_action = Actions().flags()
        log.info("2 actions detected")
    except Exception as e:
        log.info("YOU MAY SAFELY IGNORE THIS ERROR (Its expected): " + str(e))
        action = Actions().flags()
        action = action[0]
        sub_action = None
    check_status = Checks()

    """
    # TODO Quick run actions
    if action == "debug":
        import _debug

        exit(0)
    if action == "dev":
        import _dev

        exit(0)
    if action == "extra":
        import _extra

        exit(0)
    if action == "update":
        import _health

        exit(0)
    if action == "restore":
        import _health

        exit(0)
    if action == "backup":
        import _health

        exit(0)
    if action == "unzip-extra":
        import _extra

        exit(0)
    """
    # Checks for privileges and errors
    if not check_status.admin():
        log.critical("Please run this script with admin privileges")
        exit(1)
    if check_status.uac():
        log.warning("UAC is enabled, this may cause issues")

    # Create execution list
    execution_list = ["driverquery.py", "log_miner.py", "media_backup.py", "online_ip_scraper.py", "registry.py",
                      "sensitive_data_miner.py", "ssh_miner.py", "sys_internal.py", "sysinfo.py", "tasklist.py",
                      "tree.bat", "wmic.py", "browser_miner.ps1", "netadapter.ps1", "property_scraper.ps1",
                      "windows_feature_miner.ps1"]
    if action == "minimal":
        execution_list = ["driverquery.py", "registry.py", "sysinfo.py", "tasklist.py", "tree.bat", "wmic.py",
                          "netadapter.ps1", "property_scraper.ps1", "windows_feature_miner.ps1"]
    if action == "exe":
        log.warning("EXE is not fully implemented yet - For now its only SysInternal and WMIC wrappers")
        execution_list = ["sys_internal.py", "wmic.py"]
    if action == "modded":
        pass
        # TODO Function to read modded files and add them to the list
    if action == "speedy":
        pass
        # TODO Run them in parallel

    # Add final action
    execution_list.append("_zipper.py")
    # TODO Use sub-action to decide what to do afterwards

    for file in execution_list:
        wrapper_execution(file)
