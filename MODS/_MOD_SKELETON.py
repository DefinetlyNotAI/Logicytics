# Check the WiKi [Section 2, Coding Rules and Tips, Custom LOG Mechanism] for more information
# OPTIONAL - Only if using the Custom LOG Mechanism feature
# open("CUSTOM.LOG.MECHANISM", "w").close()

# This imports everything needed including the unique logger called by log- It is not optional
# To know more check the WiKi [Section 2, Coding Rules and Tips, Custom Libraries, __lib_class.py]
# from __lib_class import *

# Other Imports if needed or necessary go here

# Check the WiKi [Section 2, Coding Rules and Tips, Custom LOG Mechanism]
# OPTIONAL - Only if using the Custom LOG Mechanism feature
# log = Log(# PUT YOUR CUSTOM PARAMS IN HERE #)

# Check the WiKi [Section 2, Coding Rules and Tips, Custom LOG Mechanism & Text based logging]
# OPTIONAL - Only if using the 2 special features together at the same time
# log_funcs = {
#     "INFO": log.info,
#     "WARNING": log.warning,
#     "ERROR": log.error,
#     "CRITICAL": log.critical,
#     None: log.debug,
# }

# Your actual code, must be able to run without any interference by outside actions
# USE log.info, log.error, log.warning and log.debug as well
# You can choose to use any other of the code without issues
# Example of said code:-
#
# def MOD_EXAMPLE() -> None:
#     """
#     This function MOD is used to log different types of messages.
#
#     It logs an error message, a warning message, an info message, and a debug message.
#
#     Parameters:
#     None
#
#     Returns:
#     None
#     """
#     log.error("This is an error")
#     log.warning("This is a warning")
#     log.info("This is a info message")
#     log.debug("This is a debug message")
#     pass  # Your code here with proper logging
#
#
# MOD_EXAMPLE()

# Check the WiKi [Section 2, Coding Rules and Tips, Custom LOG Mechanism] for more information
# OPTIONAL - Only if using the Custom LOG Mechanism feature
# os.remove("CUSTOM.LOG.MECHANISM")
