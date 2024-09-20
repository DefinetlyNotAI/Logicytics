from __lib_class import *

log = Log(debug=DEBUG)
log_funcs = {
    "INFO": log.info,
    "WARNING": log.warning,
    "ERROR": log.error,
    "CRITICAL": log.critical,
    None: log.debug,
}


def command(file: str, com: str, message: str):
    """
    Executes a command and writes the output to a file.

    Args:
        file (str): The name of the file to write the command output to.
        com (str): The command to be executed.
        message (str): A message to be logged.

    Returns:
        None
    """
    try:
        output = Actions.run_command(com)
        open(file, "w").write(output)
        log.info(f"{message} Successful")
    except Exception as e:
        log.error(f"Error while getting {message}: {e}")
    log.info(f"{message} Executed")


command("Drivers.txt", "driverquery /v", "Driver Query")
command("SysInfo.txt", "systeminfo", "System Info")
