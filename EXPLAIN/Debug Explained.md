# Python Script for Logicytics Project Setup

This Python script automates several checks and operations crucial for setting up the Logicytics project environment. It ensures the correct system configuration and file management by verifying the presence of essential files, checking system properties, and logging all actions in a `DEBUG.md` file.

## Overview

The script is designed to streamline the initial setup of the Logicytics project by:

- Deleting any existing debug file.
- Defining paths to critical version and structure files.
- Checking for the existence of these files and logging their contents.
- Verifying system properties such as User Account Control (UAC) status, administrative privileges, and PowerShell execution policy.
- Logging system information and hardware specifications.

## Functionality

### `check_python_versions()`
Check for the existence of python on PATH, and log them down.


### `delete_debug_file()`
Deletes the `DEBUG.md` file in the current working directory, preparing for a fresh log.

### `define_paths()`
Constructs and returns paths to the `Logicystics.version` and `Logicystics.structure` files relative to the script's location.

### `open_debug_file()`
Opens the `DEBUG.md` file in appending mode for logging purposes.

### `check_version_file(version_file_path)`
Verifies the existence of the `Logicystics.version` file. Logs an error and exits the script if missing; otherwise, logs its content.

### `check_structure_file(structure_file_path)`
Checks for the `Logicystics.structure` file. Logs an error and exits the script if not found; otherwise, validates each path listed in the file and logs the results.

### `check_uac_status()`
Determines the UAC status by examining the `LocalAccountTokenBypassPolicy` registry key. Log whether UAC is enabled or disabled.

### `check_admin_privileges()`
Confirms if the script is running with administrative privileges. Log the privilege level accordingly.

### `check_powershell_execution_policy()`
Inquires the current PowerShell execution policy. Log the policy setting.

### `check_vm()`
Identifies if the system is running within a virtual machine. Log the detection result.

### `cmd_raw(command, check)`
Executes a raw command and handles its output or logs it to the `DEBUG.md` file based on the specified check parameter.

### `main()`
 Orchestrates the execution flow, calling all other functions in sequence to perform the necessary checks and operations.

## Usage

Run this script as part of the Logicytics project setup process. It ensures that the project environment meets all prerequisites and logs all findings in the `DEBUG.md` file, facilitating easy troubleshooting and verification of the setup process.
