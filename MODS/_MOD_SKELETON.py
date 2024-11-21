# If using the future annotations, it should be ontop of the file
# from __future__ import annotations

# Other Imports if needed or necessary go here

# This imports everything needed including the unique logger called by log - It is not optional
# To know more check the WiKi
from __lib_class import *

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})

# Your actual code, must be able to run without any interference by outside actions
# USE log.debug, log.info, log.error, log.warning and log.critical and log.string as well
# You can choose to use any other of the code without issues
# Example of said code:-


# You can enable this decorator to log the function name and the time it took to run,
# It is recommended to use this,
# as it only logs the function and the time it took to run in debug mode
# @log.function
def MOD_EXAMPLE() -> None:
    """
    This function MOD is used to log different types of messages.

    It logs an error message, a warning message, an info message, and a debug message.

    Parameters:
    None

    Returns:
    None
    """
    log.error("This is an error")
    log.warning("This is a warning")
    log.info("This is a info message")
    log.debug("This is a debug message")
    log.critical("This is a critical message")
    # This is special, allows you to use strings to specify the log level, it is not recommended to use this
    # Options are error, warning, info, debug, critical - It is case-insensitive and can be used with any of the log levels
    # Defaults with the log level of debug
    log.string("This is a random message", "ERROR")
    pass  # Your code here with proper logging like the above log options


MOD_EXAMPLE()

# Always remember to call your function at the end of the file and then leave a new line
# This is to ensure that the function is called and the file is not empty
