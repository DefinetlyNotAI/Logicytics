# PowerShell Script to Automatically Delete 'DATA' Directories in the Current Working Directory

# Function to check if the current working directory exists
function Test-CurrentDirectoryExists {
    param (
        [Parameter(Mandatory=$true)]
        [string]$currentWorkingDir
    )
    return (Test-Path $currentWorkingDir)
}

# Function to delete a directory safely
function Remove-SafeDirectory {
    param (
        [Parameter(Mandatory=$true)]
        [string]$directoryPath
    )
    try {
        Remove-Item -Recurse -Force $directoryPath -ErrorAction Stop
        Write-Host "INFO: '$directoryPath' has been successfully deleted."
    } catch {
        Write-Host "ERROR: Failed to delete '$directoryPath': $_"
    }
}

# Main script execution starts here

# Get the current working directory
$currentWorkingDir = Get-Location

# Validate if the current working directory exists
if (-not (Test-CurrentDirectoryExists -currentWorkingDir $currentWorkingDir)) {
    Write-Host "ERROR: The current working directory does not exist."
    exit
}

# Define the directory name to look for
$directoryName = "DATA"

# Get all directories in the current working directory
$directories = Get-ChildItem -Directory

# Loop through each directory
foreach ($dir in $directories) {
    # Check if the directory name matches the target
    if ($dir.Name -eq $directoryName) {
        # Safely attempt to delete the directory
        Remove-SafeDirectory -directoryPath $dir.FullName
    }
}

Write-Host "INFO: Script completed. 'DATA' directory deleted."
