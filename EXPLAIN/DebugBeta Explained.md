
# Python Script Explanation

This Python script is designed to perform a series of checks and operations related to system configuration and file existence, logging its findings to a `DEBUG.md` file. It includes functionality to delete a `DEBUG.md` file if it exists, check for the existence of specific version and structure files, verify User Account Control (UAC) status, determine if the current user has administrative privileges, and check the PowerShell execution policy. Here's a breakdown of the script:

## Import Required Modules

```python
import os
import subprocess
import winreg
```

The script imports necessary modules for file and directory operations, subprocess management, and Windows Registry access.

## Define Functions

### `delete_debug_file_if_exists()`

```python
def delete_debug_file_if_exists():
    current_dir = os.getcwd()
    debug_file_path = os.path.join(current_dir, "DEBUG.md")
    if os.path.exists(debug_file_path):
        os.remove(debug_file_path)
        print("DEBUG.md file has been deleted, to create a new one")
    else:
        print("DEBUG.md file does not exist, creating a new one")
```

This function checks if a `DEBUG.md` file exists in the current working directory. If it does, the file is deleted and a message is printed. If the file does not exist, a message is printed indicating that a new one will be created.

### Main Script Logic

```python
# Call the function to check and delete the DEBUG.md file if it exists
delete_debug_file_if_exists()

# Define the paths to the reference files
version_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SYSTEM", "Logicystics.version")
structure_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SYSTEM", "Logicystics.structure")

# Open the DEBUG.md file in appending mode
with open("DEBUG.md", "a") as debug_file:
    # Check if the version file exists
    if not os.path.exists(version_file_path):
        debug_file.write("<span style='color:red;'>Error</span>: Version file not found.\n")
        debug_file.write("\n")  # Explicitly add a newline
        exit(1)
    else:
        # Read the version number from the version file
        with open(version_file_path, 'r') as file:
            version_number = file.read().strip()
        debug_file.write("<span style='color:green;'>INFO</span>: Version number is {version_number}\n".format(
            version_number=version_number))
        debug_file.write("\n")  # Explicitly add a newline

    # Check if the structure file exists
    if not os.path.exists(structure_file_path):
        debug_file.write("<span style='color:red;'>Error</span>: Structure file not found.\n")
        debug_file.write("\n")  # Explicitly add a newline
        exit(1)

    # Read the structure file line-by-line and check if each line exists on the drive
    with open(structure_file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any leading/trailing whitespace
            item_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SYSTEM", line)
            if not os.path.isfile(item_path):
                debug_file.write(
                    "<span style='color:red;'>Error</span>: File {line} not found.\n".format(line=line))
                debug_file.write("\n")  # Explicitly add a newline
            else:
                debug_file.write(
                    "<span style='color:green;'>INFO</span>: File exists and is found at {item_path}.\n".format(
                        line=line,
                        item_path=item_path))
                debug_file.write("\n")  # Explicitly add a newline

    # Check if UAC is enabled
    try:
        uac_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System", 0, winreg.KEY_READ)
        uac_enabled = winreg.QueryValueEx(uac_key, "EnableLUA")[0] == 1
        winreg.CloseKey(uac_key)
        if uac_enabled:
            debug_file.write("<span style='color:orange;'>Warning</span>: UAC is enabled.\n")
            debug_file.write("\n")  # Explicitly add a newline
        else:
            debug_file.write("<span style='color:green;'>INFO</span>: UAC is not enabled.\n")
            debug_file.write("\n")  # Explicitly add a newline
    except WindowsError:
        debug_file.write("<span style='color:red;'>Error</span>: UAC status could not be determined.\n")
        debug_file.write("\n")  # Explicitly add a newline

    # Check if the current user is an admin
    try:
        result = subprocess.run(["net", "session"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if result.returncode!= 0:
            debug_file.write(
                "<span style='color:orange;'>Warning</span>: The current user does not have administrative privileges.\n")
            debug_file.write("\n")  # Explicitly add a newline
        else:
            debug_file.write(
                "<span style='color:green;'>INFO</span>: The current user has administrative privileges.\n")
            debug_file.write("\n")  # Explicitly add a newline

    # Check the PowerShell execution policy
    try:
        execution_policy = subprocess.check_output(["powershell", "-Command", "Get-ExecutionPolicy"], text=True).strip()
        if execution_policy!= "Unrestricted":
            debug_file.write(
                "<span style='color:red;'>Error</span>: PowerShell execution policy is not set to 'Unrestricted'.\n")
            debug_file.write("\n")  # Explicitly add a newline

    # Check the PowerShell execution policy
    try:
        if execution_policy!= "Unrestricted".\n")
            debug_file.write(
                "<span style='color:green;'>INFO</span>: PowerShell execution policy is not set to 'Unrestricted'.\