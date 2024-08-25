from __lib_actions import *
from __lib_log import Log


def driverquery():
    """
    Retrieves detailed information about the drivers installed on the system.

    This function runs the 'driverquery /v' command, captures the output, and writes it to a file named 'Drivers.txt'.
    It logs a success message if the operation is completed successfully, and an error message if any exception occurs.
    Finally, it logs a message indicating that the driver query has been executed.

    Parameters:
    None

    Returns:
    None
    """
    try:
        output = Actions.run_command("driverquery /v")
        open("Drivers.txt", "w").write(output)
        log.info("Driver Query Successful")
    except Exception as e:
        log.error(e)
    log.info("Driver Query Executed")


log = Log(debug=DEBUG)
driverquery()
