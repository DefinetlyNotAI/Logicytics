import os
from __lib_actions import *
from __lib_log import Log


def sys_internal():
    """
    This function runs a series of system internal executables and logs their output.

    It iterates over a list of executable names, constructs the command to run each one,
    captures the output, and writes it to a file named 'SysInternal.txt'.

    The function also logs information and warning messages for each executable,
    including any errors that occur during execution.
    """
    executables = [
        "psfile.exe",
        "PsGetsid.exe",
        "PsInfo.exe",
        "pslist.exe",
        "PsLoggedon.exe",
        "psloglist.exe",
    ]
    with open("SysInternal.txt", "a") as outfile:
        # Iterate over each executable
        for executable in executables:
            try:
                # Construct the command to run the executable
                command = f"{os.path.join('SysInternal_Suite', executable)}"

                # Execute the command and capture the output
                result = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                # Write the output to the File
                outfile.write("-" * 190)
                outfile.write(f"{executable} Output:\n{result.stdout.decode()}")
                log.info(f"{executable}: Successfully executed")

                # Optionally, handle errors if any
                if (
                    result.stderr.decode() != ""
                    and result.returncode != 0
                    and result.stderr.decode() is not None
                ):
                    log.warning(f"{executable}: {result.stderr.decode()}")
                    outfile.write(f"{executable}:\n{result.stderr.decode()}")

            except Exception as e:
                log.error(f"Error executing {executable}: {str(e)}")
                outfile.write(f"Error executing {executable}: {str(e)}\n")
    log.info("SysInternal: Successfully executed")


log = Log(debug=DEBUG)
sys_internal()
