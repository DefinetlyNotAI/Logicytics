from logicytics.Execute import *
from logicytics.Get import *
from logicytics.Logger import *
from logicytics.Checks import *
from logicytics.FileManagement import *
from logicytics.Flag import *
import functools


def deprecated(removal_version: str, reason: str) -> callable:
    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> callable:
            print(f"\033[91mDeprecationWarning: A call to the deprecated function {func.__name__}() has been called, {reason}. Function will be removed at version {removal_version}\033[0m")
            return func(*args, **kwargs)
        return wrapper

    return decorator


Execute = Execute()
Get = Get()
Check = Check()
FileManagement = FileManagement()
Flag = Flag()

DEBUG, VERSION, CURRENT_FILES, DELETE_LOGS = Get.config_data()
