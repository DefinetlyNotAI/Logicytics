# Collect system information
$buildNumber = [System.Environment]::OSVersion.Version.Build
$physicalMemory = [System.Diagnostics.Process]::PhysicalMemorySize64 / 1MB
$virtualMemory = [System.Diagnostics.Process]::WorkingSet64 / 1MB
$userName = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$userSid = ([System.Security.Principal.WindowsIdentity]::GetCurrent().UserValue)
$userLanguageId = [System.Globalization.CultureInfo]::CurrentCulture.LCID
$computerName = [System.Net.Dns]::GetHostName()
$systemLanguageId = [System.Globalization.CultureInfo]::CurrentUICulture.LCID
$time = Get-Date -Format HH:mm:ss
$date = Get-Date -Format dd/MM/yyyy
$rootDrive = $env:SystemDrive

# Prepare the data to be written to the File
$data = @"
Property(C): Windows Build = $buildNumber
Property(C): Physical Memory = $( $physicalMemory -as [int] )
Property(C): Virtual Memory = $( $virtualMemory -as [int] )
Property(C): Log on User = $userName
Property(C): User SID = $userSid
Property(C): User Language ID = $userLanguageId
Property(C): Computer Name = $computerName
Property(C): System Language ID = $systemLanguageId
Property(C): Time = $time
Property(C): Date = $date
Property(C): Username = $userName
Property(C): Root Drive = $rootDrive
"@

# Write the data to a text File
$data | Out-File -FilePath ".\Extra_Data.txt"

# Optionally, display a message indicating success
Write-Host "INFO: Data successfully written to Extra_Data.txt"
