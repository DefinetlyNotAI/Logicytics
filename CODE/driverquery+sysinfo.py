from __lib_actions import *
from __lib_log import Log


def command(file: str, com: str, message: str):
    try:
        output = Actions.run_command(com)
        open(file, "w").write(output)
        log.info(f"{message} Successful")
    except Exception as e:
        log.error(f"Error while getting {message}: {e}")
    log.info(f"{message} Executed")


log = Log(debug=DEBUG)
command("Drivers.txt", "driverquery /v", "Driver Query")
command("SysInfo.txt", "systeminfo", "System Info")
