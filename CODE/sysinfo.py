from CODE.Custom_Libraries.Actions import Actions
from CODE.Custom_Libraries.Log import Log

def sysinfo():
    try:
        data = Actions.run_command("systeminfo")
        open("SysInfo.txt", "w").write(data)
        Log().info("System Info Successful")
    except Exception as e:
        Log().error("Error while getting system info: " + str(e))
