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

if __name__ == "__main__":
    # Initialization
    os.makedirs("../ACCESS/LOGS/", exist_ok=True)
    os.makedirs("../ACCESS/BACKUP/", exist_ok=True)
    log = Log(debug=DEBUG)
    try:
        action, sub_action = Actions().flags()
        log.info("2 actions detected")
    except Exception as e:
        log.info("YOU MAY SAFELY IGNORE THIS: Only one action detected as we tried unpacking and got " + str(e))
        action = Actions().flags()
        sub_action = None
    check_status = Checks()

    # Quick run actions
    if action == "debug":
        Actions().run_command("python _debug.py", new_shell=True)
        exit(0)
    if action == "dev":
        Actions().run_command("python _dev.py", new_shell=True)
        exit(0)
    if action == "extra":
        Actions().run_command("python _extra.py --extra", new_shell=True)
        exit(0)
    if action == "update":
        Actions().run_command("python _health.py --update", new_shell=True)
        exit(0)
    if action == "restore":
        Actions().run_command("python _health.py --restore", new_shell=True)
        exit(0)
    if action == "backup":
        Actions().run_command("python _health.py --backup", new_shell=True)
        exit(0)
    if action == "unzip-extra":
        Actions().run_command("python _extra.py --unzip", new_shell=True)
        exit(0)

    # Checks for privileges etc
    if not check_status.admin():
        log.critical("Please run this script with admin privileges")
        exit(1)
    if check_status.uac():
        log.warning("UAC is enabled, this may cause issues")

    # Create execution list
    execution_list = []
    if action == "basic":
        execution_list = ["driverquery.py", "log_miner.py", "media_backup.py", "online_ip_scraper.py", "registry.py", "sensitive_data_miner.py", "ssh_miner.py", "sys_internal.py", "sysinfo.py", "tasklist.py", "tree.py", "wmic.py"]
    if action == "minimal":
        execution_list = ["driverquery.py", "registry.py", "sysinfo.py", "tasklist.py", "tree.py", "wmic.py"]
    if action == "exe":
        log.warning("EXE is not fully supported yet - For now its only SysInternal and WMIC")
        execution_list = ["sys_internal.py", "wmic.py"]
    if action == "modded":
        execution_list = ["driverquery.py", "log_miner.py", "media_backup.py", "online_ip_scraper.py", "registry.py", "sensitive_data_miner.py", "ssh_miner.py", "sys_internal.py", "sysinfo.py", "tasklist.py", "tree.py", "wmic.py"]
        # TODO Function to read modded files and add them to the list
    if action == "speedy":
        log.warning("This uses multiprocessing - Highly experimental and unstable")
        execution_list = ["driverquery.py", "log_miner.py", "media_backup.py", "online_ip_scraper.py", "registry.py", "sensitive_data_miner.py", "ssh_miner.py", "sys_internal.py", "sysinfo.py", "tasklist.py", "tree.py", "wmic.py"]
        # TODO Run them in parallel


    # Add final action
    execution_list.append("_zipper.py")

    # Additional Sub Actions
    if sub_action == "reboot":
        execution_list.append("R")  # Reboot
    elif sub_action == "shutdown":
        execution_list.append("S")  # Shutdown
    elif sub_action == "webhook":
        execution_list.append("W")  # Send Webhook
    else:
        execution_list.append("N")  # Nothing
