from logicytics import log, Execute


@log.function
def command(file: str, commands: str, message: str, encoding: str = "UTF-8") -> None:
    """
    Executes a command and writes the output to a file.

    Args:
        file (str): The name of the file to write the command output to.
        commands (str): The command to be executed.
        message (str): A message to be logged.
        encoding (str): The encoding to write the file in.

    Returns:
        None
    """
    log.info(f"Executing {message}")
    try:
        output = Execute.command(commands)
        with open(file, "w", encoding=encoding) as f:
            f.write(output)
        log.info(f"{message} Successful - {file}")
    except Exception as e:
        log.error(f"Error while getting {message}: {e}")


if __name__ == "__main__":
    command("Drivers.txt", "driverquery /v", "Driver Query")
    command("SysInfo.txt", "systeminfo", "System Info")
    command("GPResult.txt", "GPResult /r", "GPResult", "windows-1252")
