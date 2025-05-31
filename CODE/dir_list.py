import os
from concurrent.futures import ThreadPoolExecutor

from logicytics import log, execute


def run_command_threaded(directory: str, file: str, message: str, encoding: str = "UTF-8") -> None:
    """
    Executes a PowerShell command to recursively list directory contents and writes the output to a specified file.
    
    Args:
        directory (str): The target directory path to list contents from.
        file (str): The output file path where directory contents will be appended.
        message (str): A descriptive message for logging the operation.
        encoding (str, optional): File writing encoding. Defaults to "UTF-8".
    
    Raises:
        Exception: If command execution or file writing fails.
    
    Notes:
        - Uses PowerShell's Get-ChildItem with recursive flag
        - Appends output to the specified file
        - Logs operation start and result/error
    """
    log.info(f"Executing {message} for {directory}")
    try:
        safe_directory = directory.replace('"', '`"')  # Escape quotes
        command = f'powershell -NoProfile -Command "Get-ChildItem \\""{safe_directory}\\"" -Recurse"'
        output = execute.command(command)
        open(file, "a", encoding=encoding).write(output)
        log.info(f"{message} Successful for {directory} - {file}")
    except Exception as e:
        log.error(f"Error while getting {message} for {directory}: {e}")


@log.function
def command_threaded(base_directory: str, file: str, message: str, encoding: str = "UTF-8") -> None:
    """
    Concurrently lists contents of subdirectories within a base directory using thread pooling.
    
    Args:
        base_directory (str): Root directory to explore and list subdirectories from.
        file (str): Output file path to write directory listing results.
        message (str): Descriptive logging message for the operation.
        encoding (str, optional): File writing character encoding. Defaults to "UTF-8".
    
    Raises:
        Exception: Logs and captures any errors during thread pool execution.
    
    Notes:
        - Uses ThreadPoolExecutor for parallel directory content listing
        - Processes each subdirectory concurrently
        - Writes results to the specified file
        - Handles potential errors during thread execution
    """
    try:
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4)) as executor:
            subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if
                              os.path.isdir(os.path.join(base_directory, d))]
            futures = [executor.submit(run_command_threaded, subdir, file, message, encoding) for subdir in
                       subdirectories]
            for future in futures:
                future.result()
    except Exception as e:
        log.error(f"Thread Pool Error: {e}")


if __name__ == "__main__":
    log.warning("Running dir_list.py - This is very slow - We will use threading to speed it up")
    command_threaded("C:\\", "Dir_Root.txt", "Root Directory Listing")
