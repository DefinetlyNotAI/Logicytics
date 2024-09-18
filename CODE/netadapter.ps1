# Get all network details
Write-Output "INFO: Getting NetAdapter Info"
Get-NetAdapter | Select-Object Name, Status, MacAddress, ifIndex, InterfaceAlias, InterfaceDescription | Out-File -FilePath .\Network.txt
