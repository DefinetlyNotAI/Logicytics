import os
from CODE.Custom_Libraries.Log import Log
from CODE.Custom_Libraries.Actions import *


def sys_internal():
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

                # Write the output to the file
                outfile.write("-" * 190)
                outfile.write(f"{executable} Output:\n{result.stdout.decode()}")
                Log(debug=DEBUG).info(f"{executable}: Successfully executed")

                # Optionally, handle errors if any
                if (
                    result.stderr.decode() != ""
                    and result.returncode != 0
                    and result.stderr.decode() is not None
                ):
                    Log(debug=DEBUG).warning(f"{executable}: {result.stderr.decode()}")
                    outfile.write(f"{executable}:\n{result.stderr.decode()}")

            except Exception as e:
                Log(debug=DEBUG).error(f"Error executing {executable}: {str(e)}")
                outfile.write(f"Error executing {executable}: {str(e)}\n")
    Log(debug=DEBUG).info("SysInternal: Successfully executed")
