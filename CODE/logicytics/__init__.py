import functools
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

DEBUG, VERSION, CURRENT_FILES, DELETE_LOGS = Get.config_data()


def deprecated(removal_version: str, reason: str, show_trace: bool = True if DEBUG == "DEBUG" else False) -> callable:
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
