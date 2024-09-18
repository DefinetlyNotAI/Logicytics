# This imports everything needed including the unique logger called by log
from __lib_class import *


# Your actual code, must be able to run without any interference by outside actions
# USE log.info, log.error, log.warning and log.debug as well
# You can choose to use any other of the code without issues

# Example

def MOD() -> None:
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
    pass  # Your code here with proper logging


MOD()
