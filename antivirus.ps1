# Define the list of antivirus names to search for
$antivirusNames = @("Norton", "McAfee", "Avast", "AVG", "Bitdefender", "Kaspersky", "ESET", "Sophos", "TrendMicro", "Comodo", "Panda", "Avira", "F-Secure", "GData", "Malwarebytes", "Spybot", "ZoneAlarm", "Webroot", "McAfee", "Sophos", "Norton", "IObit")

# Run the tree command and capture its output
$treeOutput = tree /f

# Split the output into lines
$lines = $treeOutput -split "`n"

# Initialize variables for progress tracking
$totalLines = $lines.Count
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

# Output the total lines processed and what was found
Write-Host "Processed $completedLines lines."
if ($foundAntivirus.Count -gt 0) {
    Write-Host "Found Antivirus:"
    $foundAntivirus | Sort-Object -Unique | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "No antivirus found."
}

# Pause for 50 seconds
Start-Sleep -Seconds 50
