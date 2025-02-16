import configparser
import functools
import os
import traceback

from logicytics.Checks import Check
from logicytics.Execute import Execute
from logicytics.FileManagement import FileManagement
from logicytics.Flag import Flag
from logicytics.Get import Get
from logicytics.Logger import Log

Execute = Execute()
Get = Get()
Check = Check()
FileManagement = FileManagement()
Flag = Flag()


def config_data() -> tuple[str, str, list[str], bool]:
    """
    Retrieves configuration data from the 'config.ini' file.

    If the configuration file is not found in any of these locations,
    the program exits with an error message.

    Returns:
        tuple[str, str, list[str], bool]: A tuple containing:
            - Log level (str): Either "DEBUG" or "INFO"
            - Version (str): System version from configuration
            - Files (list[str]): List of files specified in configuration
            - Delete old logs (bool): Flag indicating whether to delete old log files

    Raises:
        SystemExit: If the 'config.ini' file cannot be found in any of the attempted locations
    """

    def config_path():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        configs_path = os.path.join(parent_dir, "config.ini")

        if os.path.exists(configs_path):
            return configs_path
        else:
            print("The config.ini file is not found in the expected location.")
            exit(1)

    config = configparser.ConfigParser()
    path = config_path()
    config.read(path)

    log_using_debug = config.getboolean("Settings", "log_using_debug")
    delete_old_logs = config.getboolean("Settings", "delete_old_logs")
    version = config.get("System Settings", "version")
    files = config.get("System Settings", "files").split(", ")

    log_using_debug = "DEBUG" if log_using_debug else "INFO"

    return log_using_debug, version, files, delete_old_logs


DEBUG, VERSION, CURRENT_FILES, DELETE_LOGS = config_data()

__show_trace = DEBUG == "DEBUG"


def deprecated(removal_version: str, reason: str, show_trace: bool = __show_trace) -> callable:
    """
    Decorator function that marks a function as deprecated
    and provides a warning when the function is called.

    Args:
        removal_version (str): The version when the function will be removed.
        reason (str): The reason for deprecation.
        show_trace (bool): Whether to show the stack trace when the function is called. Default is based on DEBUG set by user.

    Returns:
        callable: A decorator that marks a function as deprecated.

    Notes:
        - Uses a nested decorator function to preserve the original function's metadata
        - Prints a colorized deprecation warning
    """

    def decorator(func: callable) -> callable:
        """
        Decorator function that marks a function as deprecated and provides a warning when the function is called.
        
        Args:
            func (callable): The function to be decorated with a deprecation warning.
        
        Returns:
            callable: A wrapper function that preserves the original function's metadata and prints a deprecation warning.
        
        Notes:
            - Uses functools.wraps to preserve the original function's metadata
            - Prints a colorized deprecation warning to stderr
            - Allows the original function to continue executing
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> callable:
            """
            Wraps a deprecated function to print a warning message before execution.
            
            Args:
                *args: Positional arguments passed to the original function.
                **kwargs: Keyword arguments passed to the original function.
            
            Returns:
                The return value of the original function after printing a deprecation warning.
            
            Warns:
                Prints a colored deprecation warning to stderr with details about:
                - Function name being deprecated
                - Reason for deprecation
                - Version when the function will be removed
            """
            message = f"\033[91mDeprecationWarning: A call to the deprecated function {func.__name__}() has been called, {reason}. Function will be removed at version {removal_version}\n"
            if show_trace:
                stack = ''.join(traceback.format_stack()[:-1])
                message += f"Called from:\n{stack}\033[0m"
            else:
                message += "\033[0m"
            print(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


FileManagement.mkdir()
log = Log({"log_level": DEBUG})
