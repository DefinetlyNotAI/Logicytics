from __lib_actions import *
from __lib_log import Log


def driverquery():
    try:
        output = Actions.run_command("driverquery /v")
        open("Drivers.txt", "w").write(output)
        log.info("Driver Query Successful")
    except Exception as e:
        log.error(e)
    log.info("Driver Query Executed")


log = Log(debug=DEBUG)
driverquery()
