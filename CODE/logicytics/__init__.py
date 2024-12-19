import functools

from logicytics.Checks import Check
from logicytics.Execute import Execute
from logicytics.FileManagement import FileManagement
from logicytics.Flag import Flag
from logicytics.Get import Get
from logicytics.Logger import Log


def deprecated(removal_version: str, reason: str) -> callable:
    """
    A decorator to mark functions as deprecated.

    Args:
        removal_version (str): The version at which the function will be removed.
        reason (str): The reason why the function is deprecated.

    Returns:
        callable: The decorated function.
    """
    def decorator(func: callable) -> callable:
        """
        The actual decorator function.

        Args:
            func (callable): The function to be decorated.

        Returns:
            callable: The wrapper function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> callable:
            """
            The wrapper function that prints a deprecation warning.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                callable: The result of the original function.
            """
            print(
                f"\033[91mDeprecationWarning: A call to the deprecated function {func.__name__}() has been called, {reason}. Function will be removed at version {removal_version}\033[0m")
            return func(*args, **kwargs)

        return wrapper

    return decorator


Execute = Execute()
Get = Get()
Check = Check()
FileManagement = FileManagement()
Flag = Flag()

DEBUG, VERSION, CURRENT_FILES, DELETE_LOGS = Get.config_data()
