# Define the list of antivirus names to search for
$antivirusNames = @("Norton", "McAfee", "Avast", "AVG", "Bitdefender", "Kaspersky", "ESET", "Sophos", "TrendMicro", "Comodo", "Panda", "Avira", "F-Secure", "GData", "Malwarebytes", "Spybot", "ZoneAlarm", "Webroot", "IObit")

# Check if the 'tree' command is available
if (-not (Get-Command tree -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Tree command not found. Please install or use an alternative method."
    exit
}

# Run the tree command and capture its output
$treeOutput = tree /f

# Split the output into lines
$lines = $treeOutput -split "`n"

# Remove duplicates from the antivirus names list
$antivirusNames = $antivirusNames | Sort-Object | Get-Unique

# Initialize variables for progress tracking
$completedLines = 0
$foundAntivirus = @()

# Process each line
foreach ($line in $lines) {
    $completedLines++

    # Check for antivirus names in the line, ensuring it's a complete word
    foreach ($name in $antivirusNames) {
        if ($line -match "\b$name\b") {
            $foundAntivirus += $name
        }
    }
}

# Print the total lines processed and what was found to the console
Write-Host "Processed $completedLines lines."
if ($foundAntivirus.Count -gt 0) {
    Write-Host "INFO: Found Antivirus:"
    $foundAntivirus | Sort-Object -Unique | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "INFO: No antivirus found."
}
