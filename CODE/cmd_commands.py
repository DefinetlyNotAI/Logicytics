from __lib_class import *

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


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
command("GPResult.txt", "GPResult /r", "GPResult")
command("Dir_Root.txt", "Get-ChildItem C:\\ -Recurse", "Root Directory Listing")