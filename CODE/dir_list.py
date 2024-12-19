import os
from concurrent.futures import ThreadPoolExecutor

from logicytics import Log, DEBUG, Execute

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


def run_command_threaded(directory: str, file: str, message: str, encoding: str = "UTF-8") -> None:
    """
    Executes a command for a specific directory and writes the output to a file.

    Args:
        directory (str): The directory to run the command on.
        file (str): The name of the file to write the command output to.
        message (str): A message to be logged.
        encoding (str): The encoding to write the file in.

    Returns:
        None
    """
    log.info(f"Executing {message} for {directory}")
    try:
        command = f"powershell -Command Get-ChildItem {directory} -Recurse"
        output = Execute.command(command)
        open(file, "a", encoding=encoding).write(output)
        log.info(f"{message} Successful for {directory} - {file}")
    except Exception as e:
        log.error(f"Error while getting {message} for {directory}: {e}")


@log.function
def command_threaded(base_directory: str, file: str, message: str, encoding: str = "UTF-8") -> None:
    """
    Splits the base directory into subdirectories and runs the command concurrently.

    Args:
        base_directory (str): The base directory to split and run the command on.
        file (str): The name of the file to write the command output to.
        message (str): A message to be logged.
        encoding (str): The encoding to write the file in.

    Returns:
        None
    """
    with ThreadPoolExecutor() as executor:
        subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if
                          os.path.isdir(os.path.join(base_directory, d))]
        futures = [executor.submit(run_command_threaded, subdir, file, message, encoding) for subdir in subdirectories]
        for future in futures:
            future.result()


if __name__ == "__main__":
    log.warning("Running dir_list.py - This is very slow - We will use threading to speed it up")
    command_threaded("C:\\", "Dir_Root.txt", "Root Directory Listing")
