# If using the future annotations, it should be ontop of the file
# from __future__ import annotations

# Other Imports if needed or necessary go here

# To know more check the WiKi
from logicytics import log  # And more if needed

# Your actual code, must be able to run without any interference by outside actions
# USE log.debug, log.info, log.error, log.warning and log.critical and log.string as well
# You can choose to use any other of the code without issues
# Example of said code:-


# This log decorator logs the function name and the time it took to run,
# It is recommended to use this,
# as it only logs the function and the time it took to run
# in debug mode thus helping when people enable debug mode
# Do note however, if you are using multiple decorators, this should be the last one
# check the WiKi for more information
# Do not use this decorator if you are running a function that is part of another function
@log.function
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
    # This is special, allows you to use strings to specify the log level, it is not recommended to use these
    # Options are error, warning, info, debug, critical - It is case-insensitive and can be used with any of the log levels
    # Defaults with the log level of debug
    log.string("This is a random message", "ERROR")
    pass  # Your code here with proper logging like the above log options


# It is recommended to call your function at the end of the file using the following code
# This is to ensure that the function is called only when directly executed and not when imported
if __name__ == "__main__":
    MOD_EXAMPLE()

# Always remember to call your function at the end of the file and then leave a new line
# This is to ensure that the function is called and the file is not empty
