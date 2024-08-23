from Custom_Libraries.Log import *
from Custom_Libraries.Actions import *
import ctypes
import os
import \
    driverquery,\
    log_miner,\
    media_backup,\
    online_ip_scraper,\
    registry,\
    sensitive_data_miner,\
    ssh_miner,\
    sys_internal,\
    sysinfo,\
    tasklist,\
    wmic


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
    action, sub_action = Actions().flags()
    check_status = Checks()

    # Checks for privileges etc
    if not check_status.admin():
        log.critical("Please run this script with admin privileges")
        exit(1)
    if check_status.uac():
        log.warning("UAC is enabled, this may cause issues")

    # Create execution list
    execution_list = []
