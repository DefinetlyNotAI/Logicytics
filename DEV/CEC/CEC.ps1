function Invoke-CrashReport {
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
    while (-not $Process.StandardOutput.EndOfStream) {
        $line = $Process.StandardOutput.ReadLine()
        Write-Host $line
    }

    # Wait for the process to exit
    $Process.WaitForExit()
}
