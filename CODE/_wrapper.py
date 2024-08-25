import ctypes
from __lib_actions import *
from __lib_log import *


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

    @staticmethod
    def execute(script):
        if script.endswith(".ps1"):
            try:
                unblock_command = (
                    f'powershell.exe -Command "Unblock-File -Path {script}"'
                )
                subprocess.run(unblock_command, shell=True, check=True)
                log.info("PS1 Script unblocked.")
            except Exception as err:
                log.critical(f"Failed to unblock script: {err}")

        if script.endswith(".py"):
            result = subprocess.Popen(
                ["python", script], stdout=subprocess.PIPE
            ).communicate()[0]
            print(result.decode())
        else:
            result = subprocess.Popen(
                ["powershell.exe", ".\\" + script], stdout=subprocess.PIPE
            ).communicate()[0]
            lines = result.decode().splitlines()
            ID = next(
                (line.split(":")[0].strip() for line in lines if ":" in line), None
            )
            if ID == "INFO":
                log.info("\n".join(lines))
            if ID == "WARNING":
                log.warning("\n".join(lines))
            if ID == "ERROR":
                log.error("\n".join(lines))
            if ID == "CRITICAL":
                log.critical("\n".join(lines))
            else:
                log.debug("\n".join(lines))

    @staticmethod
    def set_execution_policy():
        # Define the command to set the execution policy
        command = 'powershell.exe -Command "Set-ExecutionPolicy Unrestricted -Scope Process -Force"'

        if DEBUG:
            log.debug("Command: " + command)

        try:
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if DEBUG:
                log.debug(f"Result: {result}")

            # Check the output for success
            if "SUCCESS" in result.stdout:
                if DEBUG:
                    log.info("Execution policy has been set to Unrestricted.")
            else:
                log.error("An error occurred while trying to set the execution policy.")

        except subprocess.CalledProcessError as err:
            log.error(
                f"An error occurred while trying to set the execution policy: {err}"
            )


# Initialization
os.makedirs("../ACCESS/LOGS/", exist_ok=True)
os.makedirs("../ACCESS/BACKUP/", exist_ok=True)
log = Log(debug=DEBUG)
log.info("Starting...")
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
try:
    check_status.set_execution_policy()
except Exception as e:
    log.warning("Failed to set execution policy: " + str(e))

# Create execution list
execution_list = [
    "driverquery.py",
    "log_miner.py",
    "media_backup.py",
    "online_ip_scraper.py",
    "registry.py",
    "sensitive_data_miner.py",
    "ssh_miner.py",
    "sys_internal.py",
    "sysinfo.py",
    "tasklist.py",
    "tree.bat",
    "wmic.py",
    "browser_miner.ps1",
    "netadapter.ps1",
    "property_scraper.ps1",
    "window_feature_miner.ps1",
]
if action == "minimal":
    execution_list = [
        "driverquery.py",
        "registry.py",
        "sysinfo.py",
        "tasklist.py",
        "tree.bat",
        "wmic.py",
        "netadapter.ps1",
        "property_scraper.ps1",
        "window_feature_miner.ps1",
    ]
if action == "exe":
    log.warning(
        "EXE is not fully implemented yet - For now its only SysInternal and WMIC wrappers"
    )
    execution_list = ["sys_internal.py", "wmic.py"]
if action == "modded":
    pass
    # TODO Function to read modded files and add them to the list
if action == "speedy":
    pass
    # TODO Run them in parallel

# Add final action
execution_list.append("_zipper.py")

for file in range(len(execution_list)):  # Loop through execution_list
    check_status.execute(execution_list[file])
    log.info(f"{execution_list[file]} executed")

# TODO Use sub-action to decide what to do afterwards
log.info("Completed successfully")
