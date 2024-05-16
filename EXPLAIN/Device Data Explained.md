# Batch Script Explanation

This batch script is designed to collect and format system information about a Windows computer, including details such as the registered owner, OS name, system manufacturer, and product ID. It then cleans up the collected data, determines the processor architecture, and outputs the gathered information into a file named `PC_Info.txt`. Here's a detailed breakdown of its functionality:

## Script Breakdown

### Initial Setup

- `@echo off`: Disables the display of commands in the command prompt window, making the output cleaner.
- `setlocal enabledelayedexpansion`: Enables delayed variable expansion, which allows the use of variables within loops and conditions that depend on their values being updated during the same iteration or evaluation.

### Collecting System Information

- `systeminfo >%temp%\info.txt`: Executes the `systeminfo` command, which gathers detailed system information, and redirects the output to a temporary file named `info.txt` in the `%temp%` directory.

### Extracting Specific Information

- The script uses multiple `for /f` loops to parse the `info.txt` file and extract specific pieces of information:
  - Registered Owner
  - OS Name
  - System Manufacturer
  - Product ID

Each piece of information is extracted using the `find` command to locate lines containing specific keywords and then parsing those lines to isolate the relevant data.

### Cleaning Up and Formatting Data

- `del %temp%\info.txt`: Deletes the temporary `info.txt` file to clean up.
- The script then trims spaces from the extracted strings using substring manipulation to clean up the data.
- It also determines the processor architecture based on predefined strings and sets the `arch` variable accordingly.

### Outputting Formatted Information

- The script constructs a formatted output string that includes the cleaned-up and determined information.
- This output is then redirected to a file named `PC_Info.txt`.

## Example Output

The final output in `PC_Info.txt` would look something like this:

```
Username:YourUsername
Hostname:YourComputerName
OS:Windows 10 Pro
Owner:John Doe
Product Key:XXXXX-XXXXX-XXXXX-XXXXX
Processors:4
Manufacturer:Dell Inc.
```

Note: The actual output will vary based on the specific system information of the computer where the script is run.

## Usage

This script is useful for quickly gathering and formatting detailed system information on a Windows computer. It can be particularly handy for IT professionals or system administrators who need to document or compare system specifications across multiple machines.