# PowerShell Script to Automatically Delete 'DATA' Directories

This script is designed to automatically find and delete all directories named 'DATA' within the current working directory. It's a straightforward utility for cleaning up temporary or backup data stored in such directories.

## Code Breakdown

### Getting the Current Working Directory

```powershell
$currentWorkingDir = Get-Location
```

The script starts by determining the current working directory using `Get-Location` and storing it in the variable `$currentWorkingDir`.

### Defining the Target Directory Name

```powershell
$directoryName = "DATA"
```

It then sets a variable `$directoryName` to `"DATA"`, specifying that the script will focus on directories with this exact name.

### Listing All Directories in the Current Working Directory

```powershell
$directories = Get-ChildItem -Directory
```

Using `Get-ChildItem` with the `-Directory` parameter, the script retrieves a list of all directories within the current working directory and stores them in the `$directories` variable.

### Looping Through Each Directory

```powershell
foreach ($dir in $directories) {
   ...
}
```

The script iterates over each directory found in the previous step. For each directory, it performs a check to see if the directory's name matches the target name.

### Checking for Match and Deleting the Directory

```powershell
if ($dir.Name -eq $directoryName) {
   ...
}
```

If the current directory's name exactly matches the target name ("DATA"), the script proceeds to attempt deletion.

#### Attempting to Delete the Directory

```powershell
try {
    Remove-Item -Recurse -Force $dir.FullName
    Write-Host "INFO: '$($dir.FullName)' has been deleted."
} catch {
    Write-Host "ERROR: Failed to delete '$($dir.FullName)': $_"
}
```

Within a `try` block, the script uses `Remove-Item` with the `-Recurse` and `-Force` parameters to attempt to delete the directory and all its contents. If successful, it logs a success message. If an error occurs (e.g., the directory is in use), it catches the exception and logs an error message detailing the failure.

### Completion Message

```powershell
Write-Host "INFO: Script completed. All 'DATA' directories found have been deleted."
```

Upon completion of the loop, the script logs a final informational message indicating that all targeted 'DATA' directories have been successfully deleted.

## Conclusion

This script is a simple yet effective tool for managing cleanup tasks related to specific directory names within the current working directory. It demonstrates basic PowerShell techniques for directory traversal, conditional logic, and error handling when performing file system operations.