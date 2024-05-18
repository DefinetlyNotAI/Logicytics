# PowerShell Script Explanation

This script is designed to scan through the output of the `tree` command on Windows, searching for mentions of various antivirus software names within the directory structure. It then reports back on which antivirus names were found.

## Code Breakdown

### Defining Antivirus Names List

```powershell
$antivirusNames = @("Norton", "McAfee", "Avast", "AVG", "Bitdefender", "Kaspersky", "ESET", "Sophos", "TrendMicro", "Comodo", "Panda", "Avira", "F-Secure", "GData", "Malwarebytes", "Spybot", "ZoneAlarm", "Webroot", "IObit")
```

This section initializes an array `$antivirusNames` containing strings of popular antivirus software names. This list serves as the criteria against which the script will match the contents of directories listed by the `tree` command.

### Checking for `tree` Command Availability

```powershell
if (-not (Get-Command tree -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Tree command not found. Please install or use an alternative method."
    exit
}
```

The script checks if the `tree` command is available on the system. If not, it outputs an error message and exits. The `-ErrorAction SilentlyContinue` parameter ensures that any errors encountered during this check do not halt the execution of the script.

### Capturing and Processing `tree` Output

```powershell
$treeOutput = tree /f
$lines = $treeOutput -split "`n"
```

Here, the script executes the `tree` command with `/f` flag to display the directory structure in full format. The output is captured into the variable `$treeOutput`. Then, the output is split into individual lines using the newline character (`\n`) as a delimiter, storing these lines in the `$lines` array.

### Removing Duplicate Antivirus Names

```powershell
$antivirusNames = $antivirusNames | Sort-Object | Get-Unique
```

Before processing, the script sorts the `$antivirusNames` array and removes any duplicate entries using `Sort-Object` and `Get-Unique`, ensuring that each name is unique and processed only once.

### Progress Tracking and Matching

```powershell
$completedLines = 0
$foundAntivirus = @()
foreach ($line in $lines) {
    $completedLines++
    foreach ($name in $antivirusNames) {
        if ($line -match "\b$name\b") {
            $foundAntivirus += $name
        }
    }
}
```

For each line in the `$lines` array, the script increments a counter `$completedLines` to track progress. It then iterates over each antivirus name in the `$antivirusNames` array, checking if the current line contains the name as a whole word (ensured by `\b` word boundary markers). If a match is found, the name is added to the `$foundAntivirus` array.

### Reporting Results

```powershell
Write-Host "Processed $completedLines lines."
if ($foundAntivirus.Count -gt 0) {
    Write-Host "INFO: Found Antivirus:"
    $foundAntivirus | Sort-Object -Unique | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "INFO: No antivirus found."
}
```

Finally, the script prints out how many lines it has processed. If any antivirus names were found, it lists them uniquely; otherwise, it informs that no antivirus names were found.

## Conclusion

This script provides a basic example of text processing in PowerShell, demonstrating how to search for specific patterns across multiple lines of text, track progress, and report findings. It can be adapted for various text-based analysis tasks beyond just scanning for antivirus software names.
