# PowerShell Script for Renaming and Moving Folders Based on User Data

This script automates the process of copying and renaming folders based on user data from various applications and system locations. It's particularly useful for backing up or analyzing user-specific data without manual intervention.

## Code Breakdown

### Defining Source Paths and Identifiers

```powershell
$sourcePaths = @(
    "C:\Users\{}\AppData\Local\Microsoft\Edge\User Data\Default\Network",
    "C:\Users\{}\AppData\Local\Google\Chrome\User Data\Default\Network",
    "C:\Users\{}\AppData\Roaming\Mozilla\Firefox\Profiles",
    "C:\Users\{}\AppData\Roaming\Opera Software\Opera Stable\Network",
    "C:\Users\{}\AppData\Roaming\Opera Software\Opera GX Stable\Network",
    'C:\\WINDOWS\\system32\\config\\SAM',
    'C:\\Windows\\System32\\config',
    'C:\\Windows\\System32\\GroupPolicy',
    'C:\\Windows\\System32\\GroupPolicyUsers',
    'C:\\Windows\\System32\\winevt\\Logs'
)
```

This section defines an array `$sourcePaths` containing strings representing the paths to various user data directories and system configuration files. Each path includes a placeholder `{}` that will be replaced with the current user's username.

### Defining Identifiers for Renaming

```powershell
$identifiers = @(
    "Edge",
    "Chrome",
    "Firefox",
    "OperaStable",
    "OperaGXStable",
    "SAM",
    "SystemConfig",
    "GroupPolicy",
    "GroupPolicyUsers",
    "WindowsEventLogs"
)
```

An array `$identifiers` is defined to hold the names by which the copied folders will be renamed. These names correspond to the types of data being copied, making it easier to identify the purpose of each folder later.

### Getting Current User Name

```powershell
$currentUser = $env:USERNAME
```

The script retrieves the current user's username from the environment variable `USERNAME` and stores it in the variable `$currentUser`.

### Base Directory for Destination

```powershell
$baseDirectory = "DATA"
```

A string `$baseDirectory` is set to `"DATA"`, indicating that the destination for the copied folders will be a directory named `DATA`.

### Main Loop Through Source Paths

```powershell
foreach ($sourcePath in $sourcePaths) {
   ...
}
```

The script iterates over each source path defined in `$sourcePaths`. For each path, it performs several operations:

#### Replacing Placeholder with Username

```powershell
$fullSourcePath = $sourcePath -replace '\{\}', $currentUser
```

The placeholder `{}` in each source path is replaced with the actual username of the current user, creating a fully qualified path to the user's data.

#### Checking Path Existence and Readability

```powershell
if (-not (Test-Path $fullSourcePath -PathType Container -ErrorAction SilentlyContinue)) {
   ...
}
```

The script checks if the constructed path exists and is accessible. If not, it issues a warning and skips to the next iteration.

#### Extracting Identifier from Source Path

```powershell
$identifier = $sourcePath.Split('\')[-1].Split('\\')[-1]
```

By splitting the source path and extracting the last segment, the script determines the type of data being copied, which corresponds to an entry in the `$identifiers` array.

#### Setting Destination Path and Creating If Necessary

```powershell
$destinationPath = Join-Path -Path $baseDirectory -ChildPath "USER_$identifier"
...
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
```

The script constructs the destination path by joining the base directory with a new child path that includes the identifier. If the destination directory does not exist, it creates it.

#### Copying and Renaming Folder

```powershell
try {
    Copy-Item -Path $fullSourcePath -Destination $destinationPath -Recurse -Force -ErrorAction SilentlyContinue
   ...
} catch {
   ...
}
```

Finally, the script attempts to copy the entire content of the source path to the destination path, including subdirectories and files. If successful, it logs the operation. In case of any error, it suppresses the error message and logs a generic error message instead.

## Conclusion

This script is a practical tool for managing user data backups or analyses by automating the process of copying and renaming folders based on user-specific data. It demonstrates effective use of PowerShell for file manipulation, path construction, and error handling.
