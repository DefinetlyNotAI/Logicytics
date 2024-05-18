# Batch Script Explanation

This batch script is designed to collect and organize system information on a Windows machine. It captures detailed system information using the `systeminfo` command, extracts specific pieces of data, cleans up the data, and finally writes the collected information into a file named `PC_Info.txt`. The script is structured to handle potential issues gracefully, such as failing to capture system information, and it organizes the output neatly for easy reading.

## Code Breakdown

### Initial Setup

```batch
@echo off
setlocal enabledelayedexpansion
```

- `@echo off` disables the echoing of commands in the command prompt, making the output cleaner.
- `setlocal enabledelayedexpansion` enables delayed variable expansion, allowing variables to be updated within loops and immediately reflected outside those loops.

### Variables Initialization

```batch
set "processor_architecture=x86"
set "username=JohnDoe"
set "computername=PC123"
set "number_of_processors=4"
```

- These variables are initialized with default values. They represent key pieces of system information that will be captured and possibly modified later in the script.

### Capturing System Information

```batch
echo Capturing system information...
systeminfo >%temp%\info.txt
```

- The script captures system information using the `systeminfo` command and redirects the output to a temporary file named `info.txt` in the `%temp%` directory.

### Verifying System Info Capture

```batch
if exist "%temp%\info.txt" (
    echo System information captured successfully.
) else (
    echo Failed to capture system information.
    exit /b 1
)
```

- This section checks if the `info.txt` file exists. If it does, it indicates success; otherwise, it logs a failure message and exits the script with a non-zero exit code to indicate an error.

### Extracting Specific Lines

```batch
for /f "tokens=2 delims=:" %%a in ('type %temp%\info.txt ^| find "Registered Owner"') do set owner=%%a
...
```

- Using a loop, the script searches the `info.txt` file for specific lines (like "Registered Owner", "OS Name", etc.) and extracts the second token (the actual value) from each line, assigning it to a variable.

### Data Cleanup

```batch
del %temp%\info.txt
set owner=!owner: =!
...
```

- After extracting the needed information, the script deletes the temporary `info.txt` file to free up space.
- It then performs cleanup on the extracted data by removing spaces from the beginning and end of the variable values.

### Determining Architecture

```batch
if /I "%processor_architecture%"=="x86" set arch=x32
...
```

- Based on the `processor_architecture` variable, the script determines the system architecture and assigns it to the `arch` variable.

### Writing Information to File

```batch
(
    echo SYSTEM: Username:%username%
  ...
) > PC_Info.txt
```

- The script formats the collected information into a readable format and writes it to `PC_Info.txt`.

### Final Messages

```batch
echo System information saved to PC_Info.txt.
echo Script completed successfully.
endlocal
```

- Upon completion, the script confirms that the system information has been saved and that the script has finished running. It then ends the local environment changes initiated by `setlocal`.

## Conclusion

This batch script is a useful tool for quickly gathering and organizing system information on a Windows machine. It demonstrates effective use of batch scripting techniques, including variable manipulation, conditional logic, and file handling, to extract and present system details in a structured and accessible manner.
