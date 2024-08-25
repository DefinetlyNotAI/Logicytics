from __lib_actions import *
from __lib_log import Log


def sysinfo():
    """
    Retrieves system information using the 'systeminfo' command and saves the output to a file named 'SysInfo.txt'.

    Parameters:
    None

    Returns:
    None

    Raises:
    Exception: If an error occurs while running the 'systeminfo' command or writing to the file.
    """
    try:
        data = Actions.run_command("systeminfo")
        open("SysInfo.txt", "w").write(data)
        log.info("System Info Successful")
    except Exception as e:
        log.error("Error while getting system info: " + str(e))


log = Log(debug=DEBUG)
sysinfo()
