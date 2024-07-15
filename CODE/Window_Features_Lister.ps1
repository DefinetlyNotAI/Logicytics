# Function to check if the script is running as Administrator
function Test-IsAdmin
{
    return ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator")
}

# Set the path to the DATA directory
$DataDir = "DATA"

# Check if the DATA directory exists, if not, create it
if (-not (Test-Path $DataDir))
{
    New-Item -ItemType Directory -Force -Path $DataDir
}

# Suppress PowerShell errors
$ErrorActionPreference = 'SilentlyContinue'

# Check if the script is running as Administrator
if (-not (Test-IsAdmin))
{
    Write-Host "ERROR: This script must be run as an Administrator."
    exit
}

# Attempt to get the list of enabled Windows features and save them to a file
try
{
    Get-WindowsOptionalFeature -Online | Out-File -FilePath "$DataDir\Enabled_Window_Features.txt" -Encoding utf8
    Write-Host "INFO: The list of enabled Windows features has been successfully saved to $DataDir\Enabled_Window_Features.txt"
}
catch
{
    # Do nothing if there's an error, effectively suppressing the error
}

# Check if the last command was successful
if ($LASTEXITCODE -ne 0)
{
    Write-Host "ERROR: Unexpected Code Crash Occured, may be due to permissions."
}
