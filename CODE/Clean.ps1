# PowerShell Script to Delete 'DATA' Directories and Move ZIP Files to ACCESS/DATA Directory

# Function to check if a directory exists
function Test-DirectoryExists
{
    param (
        [Parameter(Mandatory = $true)]
        [string]$directoryPath
    )
    return (Test-Path $directoryPath)
}

# Function to delete a directory safely
function Remove-SafeDirectory
{
    param (
        [Parameter(Mandatory = $true)]
        [string]$directoryPath
    )
    try
    {
        Remove-Item -Recurse -Force $directoryPath -ErrorAction Stop
        Write-Host "INFO: '$directoryPath' has been successfully deleted."
    }
    catch
    {
        Write-Host "ERROR: Failed to delete '$directoryPath': $_"
    }
}

# Function to move ZIP files to the specified directory
function Move-ZipFiles
{
    param (
        [Parameter(Mandatory = $true)]
        [string]$sourceDirectory,
        [Parameter(Mandatory = $true)]
        [string]$destinationDirectory
    )
    try
    {
        # Ensure the destination directory exists, create it if not
        if (-not (Test-DirectoryExists -directoryPath $destinationDirectory))
        {
            New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
        }

        # Find ZIP files in the source directory and move them
        $zipFiles = Get-ChildItem -Path $sourceDirectory -Filter "*.zip" -File
        foreach ($file in $zipFiles)
        {
            Move-Item -Path $file.FullName -Destination $destinationDirectory -Force
            Write-Host "INFO: Moved '$( $file.FullName )' to '$destinationDirectory'"
        }
    }
    catch
    {
        Write-Host "ERROR: Failed to move ZIP files: $_"
    }
}

# Main script execution starts here

# Get the current working directory
$currentWorkingDir = Get-Location

# Validate if the current working directory exists
if (-not (Test-DirectoryExists -directoryPath $currentWorkingDir))
{
    Write-Host "ERROR: The current working directory does not exist."
    exit
}

# Define the directory name to look for
$directoryName = "DATA"

# Get all directories in the current working directory
$directories = Get-ChildItem -Directory

# Loop through each directory
foreach ($dir in $directories)
{
    # Check if the directory name matches the target
    if ($dir.Name -eq $directoryName)
    {
        # Safely attempt to delete the directory
        Remove-SafeDirectory -directoryPath $dir.FullName
    }
}

# After deleting 'DATA' directories, move ZIP files to ACCESS/DATA
$sourceDirectory = Get-Location
$destinationDirectory = Join-Path -Path $sourceDirectory.Parent -ChildPath "ACCESS\DATA"

Move-ZipFiles -sourceDirectory $sourceDirectory -destinationDirectory $destinationDirectory

Write-Host "INFO: Script completed. 'DATA' directories deleted and ZIP files moved to ACCESS/DATA."
