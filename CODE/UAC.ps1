# Define the path to the UAC settings in the registry
$UACPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System"

# Check the current UAC setting
$UACStatus = Get-ItemProperty -Path $UACPath -Name "EnableLUA" -ErrorAction SilentlyContinue

if ($null -eq $UACStatus) {
    Write-Host "ERROR: UAC status could not be determined. Please ensure the script is run with administrative privileges."
} elseif ($UACStatus.EnableLUA -eq 1) {
    # UAC is on, disable it
    Set-ItemProperty -Path $UACPath -Name "EnableLUA" -Value 0
    Write-Host "INFO: UAC has been disabled. Would you like to restart the computer now? (Y/N)"
} else {
    # UAC is off, enable it
    Set-ItemProperty -Path $UACPath -Name "EnableLUA" -Value 1
    Write-Host "INFO: UAC has been enabled. Would you like to restart the computer now? (Y/N)"
}

$confirmation = Read-Host

if ($confirmation -eq "Y" -or $confirmation -eq "y") {
    Write-Host "INFO: Restarting the computer..."
    Restart-Computer -Force
} else {
    Write-Host "INFO: Restart canceled by user."
}
