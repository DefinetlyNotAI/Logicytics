# PowerShell Script to Automatically Delete 'DATA' Directories in the Current Working Directory

# Get the current working directory
$currentWorkingDir = Get-Location

# Define the directory name to look for
$directoryName = "DATA"

# Get all directories in the current working directory
$directories = Get-ChildItem -Directory

# Loop through each directory
foreach ($dir in $directories) {
    # Check if the directory name matches the target
    if ($dir.Name -eq $directoryName) {
        # Attempt to delete the directory
        try {
            Remove-Item -Recurse -Force $dir.FullName
            Write-Host "'$($dir.FullName)' has been deleted."
        } catch {
            Write-Host "Failed to delete '$($dir.FullName)': $_"
        }
    }
}

Write-Host "Script completed. All 'DATA' directories found have been deleted."
