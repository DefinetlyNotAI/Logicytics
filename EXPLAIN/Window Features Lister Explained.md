# PowerShell Script for Managing Windows Features and Directory Creation

This PowerShell script is designed to manage Windows features and directory creation, focusing on ensuring the script runs with administrator privileges and creating a specified directory if it doesn't already exist. It also attempts to retrieve and save a list of enabled Windows features to a file. Below is a detailed explanation of its components and functionalities.

## Function Definition

### `Test-IsAdmin`
```powershell
function Test-IsAdmin {
    return ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
}
```
- Defines a function `Test-IsAdmin` that checks if the current PowerShell session is running with administrator privileges. It returns `True` if the script is running as an administrator and `False` otherwise.

## Variable Initialization

```powershell
$DataDir = "DATA"
```
- Sets the path to the `DATA` directory where the script will store its output.

## Directory Creation

```powershell
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Force -Path $DataDir
}
```
- Checks if the `DATA` directory exists. If not, it creates the directory using `New-Item`.

## Error Suppression

```powershell
$ErrorActionPreference = 'SilentlyContinue'
```
- Set the error action preference to silently continue, suppressing error messages.

## Administrator Privilege Check

```powershell
if (-not (Test-IsAdmin)) {
    Write-Host "ERROR: This script must be run as an Administrator."
    exit
}
```
- Check if the script is running as an administrator. If not, it displays an error message and exits the script.

## Retrieving and Saving Enabled Windows Features

```powershell
try {
    Get-WindowsOptionalFeature -Online | Out-File -FilePath "$DataDir\Enabled_Window_Features.txt" -Encoding utf8
    Write-Host "INFO: The list of enabled Windows features has been successfully saved to $DataDir\Enabled_Window_Features.txt"
} catch {
    # Do nothing if there's an error, effectively suppressing the error
}
```
- Attempts to retrieve the list of enabled Windows features using `Get-WindowsOptionalFeature` and saves it to a file in the `DATA` directory. If successful, it logs a success message. Errors are caught and suppressed.

## Exit Code Check

```powershell
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Unexpected Code Crash Occured, may be due to permissions."
}
```
- Check the last exit code of the previous command. If it indicates an error (`$LASTEXITCODE -ne 0`), it logs an error message suggesting a possible issue with permissions.

## Conclusion

This script is a utility for managing Windows features and directory creation, emphasizing the importance of running with administrator privileges. It demonstrates effective use of PowerShell for directory management, error suppression, and interaction with Windows features.