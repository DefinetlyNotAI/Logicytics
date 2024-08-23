from __lib_actions import *
from __lib_log import Log


def sysinfo():
    try:
        data = Actions.run_command("systeminfo")
        open("SysInfo.txt", "w").write(data)
        Log(debug=DEBUG).info("System Info Successful")
    except Exception as e:
        Log(debug=DEBUG).error("Error while getting system info: " + str(e))
