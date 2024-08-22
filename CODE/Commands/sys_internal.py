import subprocess
import os

# Define the path to your Sysinternals folder
sysinternals_path = 'SysInternal_Suite'

# List of Sysinternals executables to run
executables = [
    'psfile.exe',
    'PsGetsid.exe',
    'PsInfo.exe',
    'pslist.exe',
    'PsLoggedon.exe',
    'psloglist.exe',
]

# Open the output file in append mode
with open('SysInternal.txt', 'a') as outfile:
    # Iterate over each executable
    for executable in executables:
        try:
            # Construct the command to run the executable
            command = f'"{os.path.join(sysinternals_path, executable)}"'

            # Execute the command and capture the output
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Write the output to the file
            outfile.write('-' * 190)
            outfile.write(f'{executable} Output:\n{result.stdout.decode()}')

            # Optionally, handle errors if any
            if result.stderr.decode() != '' or result.returncode != 0:
                outfile.write(f'{executable} Extra Detail (IMPORTANT):\n{result.stderr.decode()}')

        except Exception as e:
            outfile.write(f'Error executing {executable}: {str(e)}\n')

print("Script completed.")
